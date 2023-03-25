'''
GAME1:无聊-消极-HN LN  情绪3
GAME2:平静-积极-HP LP  情绪4
GAME3:恐怖-消极-HN LN  情绪2
GAME4:有趣-积极-HP LP  情绪1

    1. 14个通道的功能连接矩阵 1
    2. 二值化获得图的邻接矩阵 1 
    3. 为每个图分配一个标签
'''
import torch
import numpy as np
import networkx as nx
from scipy import sparse

'''创建GCN输入'''
# 创建一个随机图
G = nx.erdos_renyi_graph(10, 0.5)
# 获取图中节点的度数
degrees = np.array([G.degree[i] for i in range(G.number_of_nodes())])
# 获取图中节点的特征（这里假设每个节点有一个随机特征）
features = np.random.rand(G.number_of_nodes(), 1)
# 获取图的邻接矩阵（稀疏格式）
adj = sparse.coo_matrix(nx.adjacency_matrix(G))
# 对邻接矩阵加上自环并归一化
adj = adj + sparse.eye(adj.shape[0])
rowsum = np.array(adj.sum(1))
d_inv_sqrt = np.power(rowsum, -0.5).flatten()
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
# 对特征矩阵进行标准化
features = features - features.mean(axis=0)
features = features / features.std(axis=0)
# 将输入数据转换为tensor对象
features = torch.FloatTensor(features)
adj = torch.FloatTensor(np.array(adj.todense()))
degrees = torch.LongTensor(degrees)

# 定义GCN模型（这里省略了具体细节）
# class GCN(torch.nn.Module):
    # 省略

# 创建GCN模型实例
# model = GCN(input_dim=1, hidden_dim=16, output_dim=2)

# 将输入数据传入GCN模型中，并得到输出结果（这里假设是节点分类任务）
# output = model(features, adj) # output是一个10*2的tensor，表示每个节点属于两个类别的概率