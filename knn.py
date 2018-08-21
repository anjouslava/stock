#!/usr/bin python3
#-*- coding:utf-8 -*-
import tushare as ts
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
df_list_20164 = ts.get_report_data(2016,3)
df_list_20164.to_csv('20163.csv')
df_list_20171 = ts.get_report_data(2016,4)
df_list_20171.to_csv('20164.csv')
df_list_industry = ts.get_industry_classified()
df_list_industry.to_csv('industry.csv')
df_list_concept = ts.get_concept_classified()
df_list_concept.to_csv('concept.csv')

train1 = pd.read_csv('20163.csv',engine='python')
train1_index = train1.set_index('code')


#建立第一个训练集，全股票(全股票异常数据过多，实际未使用）
label1= []
list_train1 = []
count = 0
# for i in train1['code']:
#     i = str(i)
#     int_i = int(i)
#     df_price =ts.get_hist_data(i,start='2016-07-01',end='2016-09-29')
#     if df_price is None:
#         continue
#     elif df_price.empty:
#         continue
#     first_price = df_price.iloc[0,0]
#     last_price = df_price.iloc[-1,2]
#     if (last_price - first_price >= 0):
#
#         change = 1
#     else:
#         change = 0
#     try:
#         eps = train1_index.loc[int_i, 'eps']
#         roe = train1_index.loc[int_i, 'roe']
#     except KeyError:
#         continue
#     else:
#         try:
#             pe = last_price / (eps * 4)
#         except ZeroDivisionError:
#             continue
#         else:
#             pe = pe.tolist()
#             if (pe is not np.nan and isinstance(pe, float)) and (roe is not np.nan and isinstance(roe, float)):
#                 list_train1.append(pe)
#                 list_train1.append(roe)
#                 label1.append(change)
#                 count = count+1
#                 print(count)
# array_train1 = np.array(list_train1).reshape(len(list_train1) // 2,2)
# print("Finish1")
# count = 0
#建立第二个训练集，电子信息行业
label2 = []
list_train2 = []
train2 = pd.read_csv('industry.csv',engine='python')
train2_index = train2.set_index('code')
for i in train2['code']:
    i = str(i)
    int_i = int(i)
    df_price = ts.get_hist_data(i, start='2016-07-01', end='2016-09-30')
    if df_price is None:
        continue
    elif df_price.empty:
        continue
    first_price = df_price.iloc[0, 0]
    last_price = df_price.iloc[-1, 2]
    if (last_price - first_price >= 0):

        change = 1
    else:
        change = 0
    try:
        eps = train1_index.loc[int_i, 'eps']
        roe = train1_index.loc[int_i, 'roe']
    except KeyError:
        continue
    else:
        try:
            pe = last_price / (eps * 4)
        except ZeroDivisionError:
            continue
        else:
            pe = pe.tolist()
            if (pe is not np.nan and isinstance(pe, float)) and (roe is not np.nan and isinstance(roe, float)):
                list_train2.append(pe)
                list_train2.append(roe)
                label2.append(change)
                count = count+1
                print(count)
array_train2 = np.array(list_train2).reshape(len(list_train2) // 2,2)
print("Finish2")
count = 0
#建立第三个训练集，云计算概念分类
label3 = []
list_train3 = []
train3 = pd.read_csv('concept.csv',engine='python')
train3_index = train3.set_index('code')
for i in train3['code']:
    i = str(i)
    int_i = int(i)
    df_price = ts.get_hist_data(i, start='2016-07-01', end='2016-09-30')
    if df_price is None:
        continue
    elif df_price.empty:
        continue
    first_price = df_price.iloc[0, 0]
    last_price = df_price.iloc[-1, 2]
    if (last_price - first_price >= 0):
        change = 1
    else:
        change = 0
    try:
        eps = train1_index.loc[int_i, 'eps']
        roe = train1_index.loc[int_i, 'roe']
    except KeyError:
        continue
    else:
        try:
            pe = last_price / (eps * 4)
        except ZeroDivisionError:
            continue
        else:
            pe = pe.tolist()
            roe = roe.tolist()
            if (pe is not np.nan and isinstance(pe, float)) and (roe is not np.nan and isinstance(roe, float)):
                list_train3.append(pe)
                list_train3.append(roe)
                label3.append(change)
                count += 1
                print(count)

array_train3 = np.array(list_train3).reshape(len(list_train3)//2,2)
print('Finish3')
#测试集选取行业分类电子信息与概念分类云计算的交集
#存储三种方法分类的结果
result1 = []
result2 = []
result3 = []
select_stock = pd.read_csv('test.csv',engine='python')
select_stock_index = select_stock.set_index('code')
test = pd.read_csv('20164.csv',engine='python')
test_index = test.set_index('code')
#建立三种分类集
# neigh1 = KNeighborsClassifier()
# neigh1.fit(array_train1,label1)
neigh2 = KNeighborsClassifier()
neigh2.fit(array_train2,label2)
neigh3 = KNeighborsClassifier()
neigh3.fit(array_train3,label3)
for i in select_stock['code']:
    i = str(i)
    df_price = ts.get_hist_data(i, start='2016-10-10', end='2016-12-29')
    if df_price is None:
        continue
    elif df_price.empty:
        continue
    first_price = df_price.iloc[0,0]
    last_price = df_price.iloc[-1,2]
    if (last_price - first_price >= 0):
        change = 1
    else:
        change = 0
    int_i = int(i)
    try:
        eps = test_index.loc[int_i, 'eps']
        roe = test_index.loc[int_i,'roe']
    except KeyError:
        continue
    else:
        try:
            pe = first_price / ( eps* 4)
        except ZeroDivisionError:
            continue
        else:
            if (pe is not np.nan and isinstance(pe, float)) and (roe is not np.nan and isinstance(roe, float)):
                temp = []
                temp.append(pe)
                temp.append(roe)
                temp_array = np.array(temp).reshape(1,2)
                print(temp_array)
                # predict_change1 = int(neigh1.predict(temp_array))
                # if (predict_change1 == change):
                #     result1.append(1)
                # else:
                #     result1.append(0)
                predict_change2 = int(neigh2.predict(temp_array))
                print(predict_change2)
                predict_change3 = int(neigh3.predict(temp_array))
                print(predict_change3)
                if (predict_change2 == change):
                    result2.append(0)
                else:
                    result2.append(1)
                if (predict_change3 == change):
                    result3.append(0)
                else:
                    result3.append(1)
# Prob1 = result1.count(1) / len(result1)
Prob2 = result2.count(1) / len(result2)
Prob3 = result3.count(1) / len(result3)
print('result=')
# print(Prob1)
print(Prob2)
print(Prob3)


