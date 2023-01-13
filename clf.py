import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


from sklearn.decomposition import PCA


import pandas as pd
from pandas import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder  
import warnings
warnings.filterwarnings("ignore")

def sxauDateSet(X, y, standardscaler=True):
    # # X保存的是特征，对pd对象进行切片
    # X = dataFrame.iloc[:,1:-1].values # 性状
    # # y保存的是对应的标签数据
    # y = dataFrame.iloc[:,-1].values # 地理分组（标签）
    
    
    # # X保存的是特征，对pd对象进行切片
    # y = dataFrame.values[:,1] # 性状
    # # y保存的是对应的标签数据
    # X = dataFrame.values[:,2:]  # 地理分组（标签）

    # print(y)
    # 使用LabelEncoder对标签（本示例为地理分组）进行编码
    # 相关教程：https://blog.csdn.net/weixin_39450145/article/details/114156682
    label = LabelEncoder()
    y_label = label.fit_transform(y) 

    # 对数据进行归一化
    # 概率模型（树形模型）不需要归一化，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率，如决策树、RF。
    # 而像Adaboost、SVM、LR、Knn、KMeans之类的最优化问题就需要归一化。
    if standardscaler==True:
        X = StandardScaler().fit_transform(X)

    # train_test_split返回四个值，分别为X_train, X_test, y_train, y_test
    return train_test_split(X, y_label, test_size=0.25, random_state=0)
    

filePath = "D:\\work\\project\\py\\code\\HSI\\nirpyresearch\\data\\milk-powder.csv"

data = pd.read_csv(filePath)
print(data.shape)



def Pca(X, nums=20):
    """
       :param X: raw spectrum data, shape (n_samples, n_features)
       :param nums: Number of principal components retained
       :return: X_reduction：Spectral data after dimensionality reduction
    """
    pca = PCA(n_components=nums)  # 保留的特征数码
    pca.fit(X)
    X_reduction = pca.transform(X)

    return X_reduction


# 实例化分类器
svm_classifier = svm.SVC(kernel='poly')
knn_classifier = KNeighborsClassifier(n_neighbors=9)
# 使用GridSearchCV搜索最佳参数
params_grid = {"n_neighbors":[1,3,5,7,9,11]}
knn_classifier = GridSearchCV(knn_classifier,param_grid=params_grid,cv=5)
# 实例化分类器
rf_classifier = RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)

def train(X, y, clfType):
    X_train, X_test, y_train, y_test = sxauDateSet(X, y)
    
    if clfType == 6:
        classifier = knn_classifier
    elif clfType == 7:
        classifier = svm_classifier
    elif clfType == 8:
        classifier = rf_classifier
    
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)
    return acc
            



# print(train(data=data, clfType=7))