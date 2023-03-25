import FunctionConnectivity as fc
import CST
import Label
import torch
import dgl
import torch.nn as nn
from modelG import GCN
import torch.optim as optim
import numpy as np
from sklearn.model_selection import cross_val_score


FCMatrixList = fc.FC()
#二值化矩阵
binMatrixList = CST.cst_binarize(FCMatrixList) 
# binMatrixList = GPT_CST.cst_binarize(FCMatrixList) 


labels = Label.addLabel(binMatrixList)
# print(labels)
bgList = []
graphList = []
#用于存储元组(图, 标签)，collate的参数
samples = []
num = 0
for A in binMatrixList:
    binMatrix = binMatrixList[num]
    # print(f'binMatrix = {binMatrix}')
    feature = FCMatrixList[num]
    label = labels[num]
    num = num + 1
    # print(f'当前邻接矩阵：{num}')
    tar = []
    src = []
    for i in range(len(A)):
        for j in range(i+1):
            if A[i, j] == 1:
                tar.append(i)
                src.append(j)
    tar = torch.tensor(tar, dtype = torch.int)
    src = torch.tensor(src, dtype = torch.int)
    #当前A矩阵所对应的图
    g = dgl.graph((tar, src), num_nodes = 14)
    g = dgl.add_self_loop(g)
    #当前A矩阵所对应的图的特征
    # feature = np.array(binMatrix)
    feature = np.array(feature)
    # feature = feature*binMatrix
    # print(type(feature))
    feature = np.squeeze(feature).tolist()
    # print(len(feature))
    # print(type(feature))
    # print(feature)
    g.ndata['h'] = torch.tensor(feature)
    samples.append({g: label})
    graphList.append(g)

'''
k折交叉验证
7折
划分8个被试为测试集
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GCN(196, 133, 4)#196 16 4
model.to(device)
# 定义Adam优化器
opt = optim.Adam(model.parameters(),lr=0.001)   #SGD
lossF = nn.CrossEntropyLoss()

score = cross_val_score(model, graphList, labels, cv = 5)
print(score)

