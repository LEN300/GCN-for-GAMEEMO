from torch.utils.data import Dataset

# 定义一个自定义数据集类，用于存储和加载图数据和标签
class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs # 图数据列表
        self.labels = labels # 标签列表
    
    def __len__(self):
        return len(self.graphs) # 返回数据集大小
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx] # 返回第idx个样本