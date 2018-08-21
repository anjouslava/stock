import numpy as np
import pandas as pd
from sklearn import svm
def loadDataSet(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  特征矩阵
        labelMat 类标签
        row      行数
    """
    dataMat = []
    labelMat = []
    # fr = pd.read_csv(fileName,engine='python')
    fr = pd.read_csv(fileName)
    for row in range(0,fr.shape[0]):
        # dataMat.extend([fr.loc[row,'macd'],fr.loc[row,'rsi_6']])
        dataMat.extend([fr.loc[row,'macd'],fr.loc[row,'rsi_6'],fr.loc[row,'kdj'],fr.loc[row,'cci'],fr.loc[row,'sma10'],fr.loc[row,'ema10']])
        labelMat.append(fr.loc[row,'change'])
    row = fr.shape[0]
    return dataMat, labelMat,row
def getTest(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  特征矩阵
        labelMat 类标签，此处为真实对比值
        row      行数
    """
    dataMat = []
    labelMat = []
    # fr = pd.read_csv(fileName,engine='python')
    fr = pd.read_csv(fileName)
    for row in range(0,fr.shape[0]):
        # dataMat.extend([fr.loc[row,'macd'],fr.loc[row,'rsi_6']])
        dataMat.extend([fr.loc[row,'macd'],fr.loc[row,'rsi_6'],fr.loc[row,'kdj'],fr.loc[row,'cci'],fr.loc[row,'sma10'],fr.loc[row,'ema10']])
        labelMat.append(fr.loc[row,'change'])
    row = fr.shape[0]
    return dataMat, labelMat,row
clf = svm.SVC()
data,label,row = loadDataSet("train2.csv")
data = np.array(data)
data = data.reshape(row,6)
clf.fit(data,label)
test,real,row2 = getTest("test2.csv")
test = np.array(test).reshape(row2,6)
result = clf.predict(test)
# print(len(result))
# print(len(real))
count = 0
for i in range(row2):
    if result[i] == real[i]:
        count += 1
print( count/row2)
