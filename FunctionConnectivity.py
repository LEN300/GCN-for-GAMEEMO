import os
import pandas as pd
import mne
import mne_connectivity as mc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import Preprocess
'''归一化'''
def Normalize(self):
    m = np.mean(self)
    maxData = max(self)
    minData = min(self)
    return [(float(i) - m)/(maxData - minData) for i in self]

def FC():
    #功能连接矩阵
    FCMatrix = []
    #特征矩阵
    # X = []
    path = 'GAMEEMO/'
    fileName = os.listdir(path)

    '''
        0.16Hz - 43Hz
        beta, gamma
    '''
    # Freq_Bands = {"delta": [1.25, 4.0],"theta": [4.0, 8.0],
    #               "alpha": [8.0, 13.0],"beta": [13.0, 30.0],"gamma": [30.0, 49.0]}
    channel = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']
    sfreq = 128
    for i in fileName:
        print(f'当前被试：{i}')

        '''测试用'''
        # if n == 1:
        #     break
        # n = n+1
        '''测试用'''

        if i[0] != '(':
            break
        s_path = path + i + '/Preprocessed EEG Data/.csv format'
        # s_path = path + i + '/Preprocessed EEG Data/.mat format'
        dataPath = os.listdir(s_path)
        for dataName in dataPath:  
            print(f'----游戏：{dataName}')      
            data = pd.read_csv(s_path + '/' + dataName, header = None)
            data = data.drop(0, axis=0)
            data = data.drop(14, axis = 1)
            data = data.values.T.tolist()
            '''测试用'''
            # print(data.values.T.tolist())
            # if n == 1:
            #     break
            '''测试用'''
            info = mne.create_info(ch_names=channel, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)

            #预处理
            # raw = Preprocess(raw=raw)
            # print(raw.info)
            # plt.show()
            #创建等距事件，默认id为1
            events = mne.make_fixed_length_events(raw, duration=10., overlap=4 ) #15 5
            # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],first_samp=raw.first_samp)
            #通过events创建epoch
            epochs = mne.Epochs(raw,events)
            # epochs.plot(block = True)
            m = mc.spectral_connectivity_epochs(data = epochs, method = 'pli', sfreq = 128, fmin=12., fmax=43.,
                                                mode = 'multitaper')

            md = m.get_data()
            # print(md)
            # print(md.shape)
            # X.append(md)
            #spectral_connectivity_epochs得到的数据是196*41，将41维的频率求和
            md = md.sum(axis = 1)/md.shape[1]
            # print(md)
            #降维
            nmd = np.squeeze(md)
            # nor_nmd = Normalize(nmd)
            nor_nmd = nmd
            #转换为14*14的方阵，存储为下三角矩阵，为方便观察可用.astype(int)转化为整型数组

            sm = np.array(nor_nmd).reshape(14, 14)
            sm = sm + sm.T
            #画热力图
            # sns.heatmap(data=sm,cmap="RdBu_r") #viridis plasma
            # # 添加x轴标签
            # plt.xlabel("channel")
            # # 添加y轴标签
            # plt.ylabel("channel")
            # plt.show()

            FCMatrix.append(sm)
            # print(sm)
        print('========================================')
    return FCMatrix


# plt.show()

'''
    相干性
    x = y = raw.get_data()
    f, Cxy = ss.coherence(x, y, fs = sfreq)
    raw.plot(block = True)
'''


# raw.plot()
# plt.show()
# raw.copy().pick_types(meg=False, stim=True).plot(start=3, duration=6)
# plt.show()
# events = mne.find_events(raw, stim = None)  #寻找事件时间
# epochs = mne.Epochs(raw, events) 创建Epoch对象
        # data = mne.io.RawArray(data.values.T, info= mne.create_info(ch_names = channel, sfreq = 250))
        # data = scipy.io.loadmat(s_path + '/' + dataName)
        # print(dataName)
        # print(type(data))
        # print(data)
        # print('====================')
        # raw = mne.io.read_raw_bdf(s_path + '/' + dataName, eog=None, misc=None, stim_channel = None)

        # 计算每个文件的功能连接矩阵
        # m = mc.spectral_connectivity_epochs(data = data.values.T, method='pli', sfreq = 250, mode='multitaper', fmin = None,fmax = np.inf)
#构造符合数据结构的数据集



# 对功能连接矩阵进行情绪分类
# 定义GCN模型参数，比如输入特征维度、隐藏层维度、输出类别数、dropout率等
# input_dim = pli_matrix.shape[0]
# hidden_dim = 64
# output_dim = len(emotions)
# dropout_rate = 0.5
        