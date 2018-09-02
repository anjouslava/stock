import numpy as np
import pandas as pd
from sklearn import svm
'''
svm.py和ann.py所采用的原始数据均来自hs300.csv（股指变化），以及经stockstat库计算来的技术分析指标。
'''
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
'''
获取数据并保存，ann.py中同样利用该数据
'''
df = pd.read_csv('hs300.csv')
stock = stockstats.StockDataFrame.retype(df)
# use macd,rsi_6,kdj,cci,sma,ema
macd = stock['macd']
macd.to_csv("macd.csv",index=False,sep=',')
rsi_6 = stock['rsi_6']
rsi_6.to_csv("rsi_6.csv",index=False,sep=',')
kdj = stock['kdjk']
kdj.to_csv("kdj.csv",index=False,sep=',')
cci = stock['cci']
cci.to_csv("cci.csv",index=False,sep=',')
sma10 = stock['close_10_sma']
sma10.to_csv("sma10.csv",index=False,sep=',')
ema10 = stock['close_10_ema']
ema10.to_csv("ema10.csv",index=False,sep=',')
'''
从sklearn库调用svm
'''
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
