from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict, contrastLoss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class AttentionHyperEdge(nn.Module):
    """超边注意力机制 - 修复维度问题"""
    def __init__(self, numhyperedges):
        super(AttentionHyperEdge, self).__init__()
        self.numhyperedges = numhyperedges
        # 修复：注意力向量应该与超边的特征维度匹配
        self.attention_vector = nn.Parameter(t.randn(args.hyperNum, 1))  # [hyperNum, 1]
        
    def forward(self, hyper_emb):
        # hyper_emb: [num_nodes, hyperNum]
        # 计算注意力分数
        scores = t.matmul(hyper_emb, self.attention_vector)  # [num_nodes, 1]
        attention_weights = F.softmax(scores, dim=0)
        
        # 应用注意力权重
        attended_hyper_emb = hyper_emb * attention_weights
        return attended_hyper_emb

class ProgressiveFeatureDisentanglement(nn.Module):
    """渐进式特征解耦学习"""
    def __init__(self):
        super(ProgressiveFeatureDisentanglement, self).__init__()
        self.latent_dim = args.latdim
        self.factor_dim = max(1, args.latdim // 4)
        
        self.feature_disentanglers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(args.leaky),
                nn.Linear(self.latent_dim, self.factor_dim)
            ) for _ in range(4)
        ])
        
        self.feature_recombiner = nn.Sequential(
            nn.Linear(self.factor_dim, self.latent_dim),
            nn.LeakyReLU(args.leaky),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.disentangle_weights = nn.Parameter(t.ones(4) / 4)
        self.layer_norm = nn.LayerNorm(self.latent_dim)
        
    def forward(self, embeds):
        disentangled_features = []
        for disentangler in self.feature_disentanglers:
            feature = disentangler(embeds)
            disentangled_features.append(feature)
        
        weights = F.softmax(self.disentangle_weights, dim=0)
        combined_features = sum(weights[i] * disentangled_features[i] for i in range(4))
        recombined_embeds = self.feature_recombiner(combined_features)
        output = self.layer_norm(embeds + recombined_embeds)
        
        return output, disentangled_features

class AdaptiveResidualGate(nn.Module):
    """自适应残差门控网络"""
    def __init__(self):
        super(AdaptiveResidualGate, self).__init__()
        
        # 门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(args.latdim * 2, args.latdim),
            nn.ReLU(),
            nn.Linear(args.latdim, args.latdim),
            nn.Sigmoid()
        )
        
        # 残差变换
        self.residual_transform = nn.Sequential(
            nn.Linear(args.latdim * 2, args.latdim),
            nn.LeakyReLU(args.leaky),
            nn.Linear(args.latdim, args.latdim)
        )
        
        self.layer_norm = nn.LayerNorm(args.latdim)
        
    def forward(self, current_emb, previous_emb):
        # 拼接当前和前一层的嵌入
        combined = t.cat([current_emb, previous_emb], dim=-1)
        
        # 计算门控权重
        gate_weights = self.gate_network(combined)
        
        # 计算残差变换
        residual = self.residual_transform(combined)
        
        # 应用门控残差连接
        output = gate_weights * residual + current_emb
        
        return self.layer_norm(output)

class Model(nn.Module):
    """三重创新组合模型"""
    def __init__(self):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayer = GCNLayer()
        self.hgnnLayer = HGNNLayer()
        self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
        self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
        
        # 三重创新组件
        self.uHyperAttention = AttentionHyperEdge(args.hyperNum)
        self.iHyperAttention = AttentionHyperEdge(args.hyperNum)
        self.gcn_disentanglers = nn.ModuleList([
            ProgressiveFeatureDisentanglement() for _ in range(args.gnn_layer)
        ])
        self.hyper_disentanglers = nn.ModuleList([
            ProgressiveFeatureDisentanglement() for _ in range(args.gnn_layer)
        ])
        self.residual_gates = nn.ModuleList([
            AdaptiveResidualGate() for _ in range(args.gnn_layer)
        ])

        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        # 创新1: 超边注意力机制
        uuHyper = self.uHyperAttention(self.uEmbeds @ self.uHyper)
        iiHyper = self.iHyperAttention(self.iEmbeds @ self.iHyper)
        
        embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        gnnLats = []
        hyperLats = []
        disentangled_features_list = []

        for i in range(args.gnn_layer):
            # GCN传播
            temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
            
            # HyperGNN传播
            hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keepRate), lats[-1][:args.user])
            hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keepRate), lats[-1][args.user:])
            hyperEmbeds = t.concat([hyperULat, hyperILat], dim=0)
            
            # 创新2: 渐进式特征解耦
            gnn_recombined, gnn_disentangled = self.gcn_disentanglers[i](temEmbeds)
            hyper_recombined, hyper_disentangled = self.hyper_disentanglers[i](hyperEmbeds)
            
            # 创新3: 自适应残差门控
            gated_gnn = self.residual_gates[i](gnn_recombined, lats[-1])
            gated_hyper = self.residual_gates[i](hyper_recombined, lats[-1])
            
            # 融合
            fused_embeds = gated_gnn + gated_hyper
            
            gnnLats.append(gated_gnn)
            hyperLats.append(gated_hyper)
            lats.append(fused_embeds)
            disentangled_features_list.append((gnn_disentangled, hyper_disentangled))
            
        embeds = sum(lats)
        return embeds, gnnLats, hyperLats, disentangled_features_list

    def calcLosses(self, ancs, poss, negs, adj, keepRate):
        embeds, gcnEmbedsLst, hyperEmbedsLst, disentangled_features = self.forward(adj, keepRate)
        uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
        
        # BPR损失 - 添加数值稳定性
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (t.clamp(scoreDiff.sigmoid(), min=1e-8, max=1-1e-8)).log().mean()

        # 对比学习损失
        sslLoss = 0
        for i in range(args.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp)
            sslLoss += contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)

        # 特征解耦正则化损失
        disentangle_loss = 0
        for i, (gnn_features, hyper_features) in enumerate(disentangled_features):
            for j in range(4):
                for k in range(j+1, 4):
                    gnn_sim = F.cosine_similarity(gnn_features[j], gnn_features[k], dim=-1)
                    hyper_sim = F.cosine_similarity(hyper_features[j], hyper_features[k], dim=-1)
                    disentangle_loss += (gnn_sim ** 2).mean() + (hyper_sim ** 2).mean()
        
        total_bpr_loss = bprLoss
        total_ssl_loss = sslLoss * args.ssl_reg
        total_disentangle_loss = disentangle_loss * 0.01
        
        return total_bpr_loss, total_ssl_loss, total_disentangle_loss
    
    def predict(self, adj):
        embeds, _, _, _ = self.forward(adj, 1.0)
        return embeds[:args.user], embeds[args.user:]

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

    def forward(self, adj, embeds):
        return (t.spmm(adj, embeds))

class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=args.leaky)
    
    def forward(self, adj, embeds):
        lat = (adj.T @ embeds)
        ret = (adj @ lat)
        return ret

class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)