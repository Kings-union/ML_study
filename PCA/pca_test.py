# encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
def loadData(filename, delim='\t'):
    data = pd.read_csv(filename)
    x = data[list(range(4))]
    print(data)
    return np.mat(data)


def percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

            # step1. zero mean


def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, percentage):
    # step2. 求协方差矩阵
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # rowval=0，表明传入一行代表一个样本

    # step3. 对协方差矩阵做特征分解：求特征值、特征矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    n = percentage2n(eigVals, percentage)
    # step4. 选取前 n 个最大的特征对应的特征向量矩阵
    eigValidx = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValidx = eigValidx[-1:-(n + 1):-1]  # 最大的n 个特征值的下标
    n_eigVect = eigVects[:, n_eigValidx]  # 最大的n 个特征值对应的特征向量
    lowDdataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDdataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDdataMat, reconMat

def plotW(self, lowDataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(lowDataMat[:, 0], lowDataMat[:, 1], marker='*', s=90)
    ax.scatter(reconMat[:, 0], reconMat[:, 1], marker='*', s=50, c='red')
    plt.show()

if __name__ == "__main__":
    matdata = loadData("C://Users//xwchang//Desktop//train.csv")
    lowDdataMat, reconMat = pca(matdata, 0.99)
    plotW(lowDdataMat, reconMat)
