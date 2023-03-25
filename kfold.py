from GraphDataset import GraphDataset
from dgl.dataloading import GraphDataLoader

def kfold_split(dataset, labels, k_splits):
    data_size = len(dataset)
    fold_size = int(data_size/k_splits)
    split_counter = 1
    chunked_dataset = []
    chunked_labels = []
    start = 0
    #将数据集和标签分成k折
    while split_counter <= k_splits:
        chunked_dataset.append(dataset[start : (start+fold_size)])
        chunked_labels.append(labels[start : (start+fold_size)])
        start = start+fold_size
        split_counter = split_counter+1
    
    train_dataloader_list = []
    test_dataloader_list = []
    for index in range(k_splits):
        test_data = GraphDataset(chunked_dataset[index], chunked_labels[index])
        test_dataloader = GraphDataLoader(test_data, batch_size=1, shuffle=False)
        test_dataloader_list.append(test_dataloader)
        if index == 0:
            #使用sum进行降维
            train_data = sum(chunked_dataset[index+1:k_splits], [])
            train_labels = sum(chunked_labels[index+1:k_splits], [])
        elif index == k_splits-1:
            train_data = sum(chunked_dataset[0:index], [])
            train_labels = sum(chunked_labels[0:index], [])
        else:
            train_data = sum(chunked_dataset[0:index]+chunked_dataset[index+1:k_splits], [])
            train_labels = sum(chunked_labels[0:index]+chunked_labels[index+1:k_splits], [])

        train_dataset = GraphDataset(train_data, train_labels)
        train_dataloader = GraphDataLoader(train_dataset, batch_size=1, shuffle=False)
        train_dataloader_list.append(train_dataloader)
    return train_dataloader_list, test_dataloader_list    