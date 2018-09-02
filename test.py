#!/usr/bin python3
#-*- coding:utf-8 -*-
import tushare as ts
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
df_list_industry = ts.get_industry_classified()
df_list_industry.to_csv('industry1.csv')
df_list_concept = ts.get_concept_classified()
df_list_concept.to_csv('concept1.csv')