import FunctionConnectivity as fc
import CST
import Label
import torch
import dgl
import torch.nn as nn
from modelG import GCN
import torch.optim as optim
import numpy as np
from kfold import kfold_split
import matplotlib.pyplot as plt
import EdgeFeature as EF


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
    feature = FCMatrixList[num]
    label = labels[num]
    num = num + 1
    # print(f'当前邻接矩阵：{num}')
    tar = []
    src = []
    # e_feature = np.eye(len(A), dtype = np.float64)
    e_feature = []
    for i in range(len(A)):
        for j in range(i):
            if A[i, j] == 1:
                tar.append(i)
                src.append(j)
                # e_feature[i, j] = feature[i, j]
                e_feature.append(((i, j), feature[i, j]))
                
    ef = EF.Edgefeature(e_feature, len(A))
    tar = torch.tensor(tar, dtype = torch.int)
    src = torch.tensor(src, dtype = torch.int)
    print(f'tar长度：{tar.shape}')
    print(f'src长度：{src.shape}')
    e_feature = torch.tensor(ef.get_edata(), dtype=float)
    g = dgl.graph((tar, src)) #num_nodes = 14
    
    print(f'e_feature长度：{e_feature.shape}')
    print(f'图的信息：{g}')
    g.edata['h'] = e_feature
    g = dgl.add_self_loop(g, fill_data = 1.0)
    
    samples.append({g: label})
    graphList.append(g)

'''
k折交叉验证
7折
划分8个被试为测试集
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

acc = 0
k_splits = 7 #7
train_dataloader_list, test_dataloader_list = kfold_split(graphList, labels, k_splits)
loss_list = [] #loss曲线
acc_list = []  #acc曲线

#训练集和验证集
for k in range(k_splits):
    model = GCN(196, 133, 4)#196 16 4
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.001)   #SGD
    lossF = nn.CrossEntropyLoss()

    train_dataloader = train_dataloader_list[k]
    test_dataloader = test_dataloader_list[k]
    for epoch in range(201):
        print(f"epoch:{epoch+1}")
        t_loss = 0
        for batched_graph, label in train_dataloader:
            batched_graph, label = batched_graph.to(device), label.to(device)
            # feats = batched_graph.edata['h']
            feats = ef.get_edataM()
            feats = feats.reshape(-1, 14*14)
            feats = torch.tensor(feats, dtype = torch.float32).to(device)
            # feats = feats.view(-1, 14*14)
            # print(type(batched_graph))
            print(type(feats))
            logits = model(batched_graph, feats)
            # print(logits.shape)
            loss = lossF(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_loss = t_loss+loss.item()
            print(f"----loss:{loss.item()}")
        loss_list.append(t_loss/len(train_dataloader))
    #每折的测试
    model.eval()

    correct = 0
    for it, (batched_graph, label) in enumerate(test_dataloader):     # 批遍历测试集数据集
        batched_graph, label = batched_graph.to(device), label.to(device)
        feats = batched_graph.edata['h']

        feats = feats.view(-1, 14*14)
        out = model(batched_graph, feats) # 一次前向传播
        pred = out.argmax(dim=1)                         # 使用概率最高的类别
        correct += int((pred == label).sum())           # 检查真实标签
    acc = acc+correct / len(test_dataloader)
    acc_list.append(correct / len(test_dataloader))
print(acc/k_splits)

x1 = range(len(loss_list))
y1 = loss_list
x2 = range(len(acc_list))
y2 = acc_list

plt.figure()
plt.plot(x1,y1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')

plt.figure()
plt.plot(x2,y2)
plt.xlabel('k')
plt.ylabel('acc')
plt.title('Accuracy')

plt.show()

#测试集
# Ltestdata = GraphDataset(LtestList, Llabels)
# Ltest = GraphDataLoader(Ltestdata)
# model.eval()
# Lacc = 0
# for it, (batched_graph, label) in enumerate(Ltest):     # 批遍历测试集数据集
#     batched_graph, label = batched_graph.to(device), label.to(device)
#     feats = batched_graph.edata['h']

#     feats = feats.view(-1, 14*14)
#     out = model(batched_graph, feats) # 一次前向传播
#     pred = out.argmax(dim=1)                         # 使用概率最高的类别
#     Lacc += int((pred == label).sum())           # 检查真实标签
# Lacc = Lacc / len(Ltest)

# print('=========================')
# print(f'最终测试集准确率：{Lacc}')