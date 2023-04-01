import mne
import os
import matplotlib.pyplot as plt
import pandas as pd

def Preprocess(raw):
    raw.set_montage(montage = 'standard_1020')
    #滤波
    raw_filter = raw.copy()
    # raw_filter = raw_ref.copy()
    raw_filter.filter(l_freq=12., h_freq=43.)
    return raw_filter
    
# path = 'GAMEEMO/'
# fileName = os.listdir(path)
# channel = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']
# sfreq = 128
# for i in fileName:
#     if i[0] != '(':
#             break
#     s_path = path + i + '/Preprocessed EEG Data/.csv format'
#     # s_path = path + i + '/Raw EEG Data/.csv format'
#     dataPath = os.listdir(s_path)
#     #对每一份脑电文件进行预处理
#     for dataName in dataPath:
#         data = pd.read_csv(s_path + '/' + dataName, header = None)
#         data = data.drop(0, axis = 0)
#         data = data.drop(14, axis = 1)
#         data = data.values.T.tolist()
#         info = mne.create_info(ch_names = channel, sfreq = sfreq, ch_types = 'eeg')
#         raw = mne.io.RawArray(data, info)
#         raw.plot(scalings = 'auto')
#         plt.show(block = True)

#         #导入数据并剔除无用channels
#         # raw.set_channel_types({'VEOG':'eog'})
#         # raw.pick(picks='all',exclude=['HEOG','EKG','EMG','Trigger'])

#         #电极定位
#         raw.set_montage(montage = 'standard_1020')

#         #插值坏导
#         # raw_cropped = raw.copy()
#         # raw_cropped.plot(scalings = 'auto')
#         # badflag = False
#         # if raw_cropped.info['bads']:
#         #     print('已选择坏导: ',raw_cropped.info['bads'], '开始进行插值')
#         #     badflag = True
#         # else:
#         #      print('无坏导，跳过插值')
#         # if badflag:
#         #     raw_cropped.load_data()
#         #     raw_cropped.interpolate_bads()
#         #     raw_cropped.plot()
#         #     plt.show()

#         #重参考
#         # raw_ref = raw_cropped.copy()
#         # raw_ref.load_data()
#         # raw_ref.set_eeg_reference(ref_channels = [])
#         # raw_ref.plot(block=True,title='重参考完成，无误请关闭窗口')
#         # plt.show()

#         # #滤波
#         raw_filter = raw.copy()
#         raw_filter.plot_psd()
#         plt.show(block = True)
#         # raw_filter = raw_ref.copy()
#         raw_filter.filter(l_freq=12, h_freq=43 )
#         # raw_filter.notch_filter(freqs=50)
#         raw_filter.plot_psd()
#         plt.show(block = True)
#         # plt.show(block=False)
#         # raw_filter.plot(start=20,duration=1,block=True,title='滤波完成，准备ICA，无误请关闭窗口')
        
        # #ICA
        # ica = mne.preprocessing.ICA(n_components=10, method='picard', max_iter=800)
        # ica.fit(raw_filter)

        # # 自动检测眼电伪影信号
        # eog_indices, eog_scores = ica.find_bads_eog(raw)
        # # 将检测到的伪影信号添加到exclude属性中
        # ica.exclude.extend(eog_indices)
        # # 应用ICA，去除伪影信号
        # ica.apply(raw)

        # raw_filter.load_data()
        # ica.plot_components()
        # ica.plot_sources(raw_filter, show_scrollbars=False, title='请选择需要去除的成分')
        # raw.plot(scalings = 'auto')
        # plt.show(block=True)
        # print(ica)
        # raw_recons = raw_filter.copy()
        # raw_recons = ica.apply(raw_recons)
        # raw_filter.plot(n_channels=14,title='ICA处理前, 确认请关闭')
        # raw_recons.plot(n_channels=14,title='ICA处理后, 确认请关闭')
        # raw.plot(scalings = 'auto')
        # plt.show(block=True)

''' 
    以下ICA过程由必应生成
'''
# 初始化ICA对象
# ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
# # 使用ICA对象拟合数据
# ica.fit(raw)

# # 自动检测眼电伪影信号
# eog_indices, eog_scores = ica.find_bads_eog(raw)
# # 自动检测心电伪影信号
# ecg_indices, ecg_scores = ica.find_bads_ecg(raw)

# # 将检测到的伪影信号添加到exclude属性中
# ica.exclude.extend(eog_indices)
# ica.exclude.extend(ecg_indices)

# # 应用ICA，去除伪影信号
# ica.apply(raw)