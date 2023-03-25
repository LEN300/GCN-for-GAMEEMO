

def addLabel(binMatrix):
    #游戏对应的情绪标签
    gameLabel = {'0':2, '1':3, '2':0, '3':1}
    labels = [] # 存储所有的标签
    # print(len(binMatrix))
    for i in range(len(binMatrix)):
        # print(i)
        key = str(i%4)
        labels.append(gameLabel[key])
    print(f'邻接矩阵数量：{len(binMatrix)}')
    print(f'标签数量：{len(labels)}')
    return labels


    # 创建自定义数据集对象并用DataLoader加载（这里假设批大小为16）
    # train_dataset = GraphDataset(train_graphs)
    # test_dataset = GraphDataset(test_graphs, test_labels)
    # train_loader = DataLoader(train_dataset, batch_size=16)
    # test_loader = DataLoader(test_dataset, batch_size=16)
    # return train_graphs, train_labels, test_graphs, test_labels