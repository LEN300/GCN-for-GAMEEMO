# 导入一些必要的库
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

from torch._C import device
import Label as L
import dgl



# 定义GCN模型类，继承自torch.nn.Module
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        nn.Dropout(0.5),
        self.relu = nn.ReLU()
        self.conv2 = dglnn.GraphConv(hidden_dim, n_classes)
        # self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g, h):
        # 应用图卷积和激活函数
        out = self.conv1(g, h)
        out = self.relu(out)
        out = self.conv2(g, out)
        # h = F.relu(h)
        with g.local_scope():
            g.ndata['h'] = out
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h')
            # return self.classify(hg)
            return self.softmax(hg)
    



# data_loader, test_loader, X, A = L.GraphDataset.addLabel()

# ''' 参数设置和模型定义'''
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 构造模型
# model = GCN(112, 28, 4) 

 
# model.train()
# for epoch in range(300):
#     optimizer.zero_grad()
#     out = model.forward(X, A)                               # edges 类型：int64
#     loss = F.nll_loss(out[idx_train], labels[idx_train])        # 损失函数
#     loss.backward()
#     optimizer.step()
#     print(f"epoch:{epoch+1}, loss:{loss.item()}")

# # 模型训练
# model.train()
# epoch_losses = []
# for epoch in range(100):
#     epoch_loss = 0
#     for iter, (batchg, label) in enumerate(data_loader):
#         batchg, label = batchg.to(DEVICE), label.to(DEVICE)
#         prediction = model(batchg)
#         loss = loss_func(prediction, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.detach().item()
#     epoch_loss /= (iter + 1)
#     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
#     epoch_losses.append(epoch_loss)

