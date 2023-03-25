import numpy as np
import torch
from sklearn.preprocessing import Binarizer

def binarize(M):
    size = len(M)
    #np.eye 创建布尔类型的矩阵，对角线为true，其余元素为false
    #将矩阵M的对角线置0
    M[np.eye(len(M), dtype = np.bool)] = 0
    #矩阵M每列按照升序排列
    A = np.sort(M,axis=0)

    #上限
    a = (0.85*(size-2)*(size-1)/2)
    a = round(a, 0)
    C_glob = []
    for i in range(1, int(a+1)):
        #使用item进行矩阵索引，并将取到的元素作为阈值
        edgeThresh = A.T.item(round(0.15*(size-2)*(size-1)/2)+2*i)
        #构造Binarizer对象
        binarizer = Binarizer(threshold = edgeThresh)
        #使用选取的阈值进行二值化
        Propi = binarizer.transform(M).astype(int)
        # print(Propi)
        # print('==========================')
        #转为tensor对象
        Propi = torch.from_numpy(Propi)
        # print(Propi)
        
        #diag获得矩阵对角线
        Dsq = torch.diag(torch.diag(Propi^2))
        C_glob.append(sum(torch.diag( Propi^3))/sum(sum(Propi^2 - Dsq)))
    # print('**************************')
    C_glob = np.array(C_glob)
    C_glob = abs(C_glob - 0.5)
    # print(C_glob)
    C_glob = C_glob.tolist()
    index = C_glob.index(min(C_glob))
    CSThresh = A.T.item(round(0.15*(size-2)*(size-1)/2)+index) #A(round(.15*(length(M)-2)*(length(M)-1)/2)+index)
    binarizer = Binarizer(threshold = CSThresh)
    CSTAdMat = binarizer.transform(M)
    return CSTAdMat.astype(int)


def cst_binarize(list):
    binMatrix = []
    # 获取CST分数矩阵 
    for i in list:
        matrix = i
        cst = binarize(matrix) 
        # print('=========================================')
        # print(f'cst = {cst}')
        # 返回根据条件判断后得到的01矩阵 
        binMatrix.append(cst)
    return binMatrix



