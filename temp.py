# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import pandas as pd
# import numpy as np

# data=pd.read_csv("iris.csv")
# x=data.iloc[:,1:2]

# from sklearn.preprocessing import MinMaxScaler

# mmc=MinMaxScaler()
# X_train=mmc.fit_transform(x)

# from sklearn.preprocessing import StandardScaler

# sc= StandardScaler()
# X_train2= sc.fit_transform(x)

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

#veri kümesini oku
df= [[1,10,8],[23,4,np.nan],[8,7,np.nan]]

#eksik verileri KNN algoritması ile doldur
imputer =KNNImputer(n_neighbors=2) #n_neighbors en yakını yapıştırıyor.
df=imputer.fit_transform(df)
2-4-6-8-150-10-13-15
2-4-6-8-10-13-15-150


def iqr(data):
   data.sort()
   q1, q3 = np.percentile(data, [25,75])
   iqr = q3 - q1
   return iqr

data = [2, 4, 6, 8, 10, 13, 15, 150]
data.sort()
q1, q3 = np.percentile(data, [25, 75])
a= q3 - q1
alt = q1 - (1.5 * a)
üst = q3 + (1.5 * a)


liste= [2, 4, 6, 8, 10, 13, 15, 150]
data=pd.DataFrame(liste)

q25=np.percentile(liste,25)
q75=np.percentile(liste,75)
iqr=q75-q25
sabit=1,5*iqr
alt=q25-sabit
ust=q75+sabit

# liste2=[]
# for i in data.values :
#     if(i>alt)&(i<üst):alt<i<üst
#        liste2.append(i)
    
# veri=pd.DataFrame(liste2)

liste2=[]
for i in data.values:
    if np.logical_and(i > alt, i < ust).any():
        liste2.append(i)
        
        
veri=pd.DataFrame(liste2)
print(veri)
