import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]
#使用RBF(Radial basis function)核函数处理
def K(x,z,sigma=1.5):
    return np.exp(np.dot((x-z),(x-z).T)/-(2*sigma**2))
#对应课本147页的g(x_i)，该函数助于验证KKT条件
def g(i,x,y,alpha,b):
    sum=b
    for j in range(len(y)):
        sum+=alpha[j]*y[j]*K(x[i],x[j])
    return sum
#验证第i个样本点是否满足KKT条件
def isKKT(alpha,i,x,y,b,C):
    if alpha[i]==0 and y[i]*g(i,x,y,alpha,b)>=1:
        return True
    elif alpha[i]==C and y[i]*g(i,x,y,alpha,b)<=1:
        return True
    elif alpha[i]>0 and alpha[i]<C and y[i]*g(i,x,y,alpha,b)==1:
        return True
    else:
        return False
#验证第i个样本点违反KKT条件的程度。由于KKT条件和y_i*g(x_i)与1的不等式有关
#因此计算y_i*g(x_i)与1之间差值的绝对值作为衡量违反程度的标准
def vioKKT(alpha,i,x,y,b):
    return abs(y[i]*g(i,x,y,alpha,b)-1)
#在数组x中找到值为a的元素第一次出现的位置
def findindex(x,a):
    for i in range(len(x)):
        if x[i]==a:
            return i
#某个样本点分类误差函数
def E(w,b,x_k,y_k):
    predi_k=int(np.sign(np.dot(w,x_k.T)+b))
    return predi_k-y_k
#计算样本点与分割直线的距离
def distance_count(x,w,b):
    return abs(w[0]*x[0]+w[1]*x[1]+b) / np.sqrt(w[0]**2 + w[1]**2)