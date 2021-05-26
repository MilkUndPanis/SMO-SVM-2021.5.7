from functions import *
from numpy import *
from random import *
from matplotlib import pyplot as plt
def train(C=1.0):
    #获得数据集
    x,y=create_data()
    #设定迭代次数为100次
    iter=100
    #样本容量也就是标签的个数
    N=len(y)
    #alpha的初始值取全0
    alpha=zeros(len(y))
    #设置i,j的初始值（对应alpha1和alpha2）
    i,j=randint(0,N-1),randint(0,N-1)
    #保证i≠j
    while i==j:
        i=randint(0,N-1)
    for k in range(iter):
        #x的尺寸为一个1×2行向量
        x_i,x_j=x[i],x[j]
        #y的取值为+1或-1
        y_i,y_j=y[i],y[j]
        #计算ita，为计算a2_newunc做准备
        ita=K(x_i,x_i)+K(x_j,x_j)-2*K(x_i,x_j)
        if ita==0:
            continue
        #计算分割平面参数w与b
        #x:100×2矩阵，w：1×2矩阵
        #由于y-dot(w,x.T)是个与y等长的行向量，取其各元素平均值
        w=dot(alpha*y,x)
        b=mean(y-dot(w,x.T))
        #计算误差E1和E2
        E_i=E(w,b,x_i,y_i)
        E_j=E(w,b,x_j,y_j)
        #计算a2_ewunc
        a1_old=alpha[i]
        a2_old=alpha[j]
        a2_newunc=a2_old+y_j*(E_i-E_j)/ita
        #计算L与H
        L,H=0.0,0.0
        if y_i!=y_j:
            L=max(0,a2_old-a1_old)
            H=min(C,C+a2_old-a1_old)
        elif y_i==y_j:
            L=max(0,a2_old+a1_old-C)
            H=min(C,a2_old+a1_old)
        #计算剪辑后a2_new与a1_new的值
        a2_new=max(L,min(H,a2_newunc))
        a1_new=a1_old+y_i*y_j*(a2_old-a2_new)
        #更新alpha
        alpha[i],alpha[j]=a1_new,a2_new
        #violation表示每个元素违反KKT条件的程度
        violation=zeros(N)
        #对每一个样本点检验KKT条件，在violation内记录每个样本点违反KKT的程度
        for k in range(N):
            if isKKT(alpha,k,x,y,b,C)==False:
                violation[k]=float(vioKKT(alpha,k,x,y,b))
            #如果没有违反KKT条件，则违反程度是0
            else:
                violation[k]=0.0
        #找到violation中违反程度最大的点，设定为i，对应alpha_1
        i=findindex(violation,max(violation))
        #这里设置j（对应alpha_2）为不等于i的随机数。
        #原本alpha_2的选取应该是令abs(E_i-E_k)最大的k值对应的alpha点
        #经过测试，在大多数情况下，abs(E_i-E_k）(1×100向量)的所有元素都是0
        #即预测每个元素都准确，每个元素的分类误差都是0，误差的差值也是0
        #只有少数情况下，会有一个误差差值不等于0
        #对于前一种情况，无所谓“最大的误差差值”（因为都是0），因此只能设置j为随机数
        #对于后一种情况，由于出现的次数少，并且那一个不为0的差值的元素出现的位置具有随机性
        #因此总是将j设定为随机数
        j=randint(0,N-1)
        while j==i:
            j = randint(0, N - 1)
    #计算最终（迭代100次）分割平面参数
    w = dot(alpha * y, x)
    b = mean(y - dot(w, x.T))
    draw_x, draw_y, draw_label = [], [], []
    #在散点图上标记样本点的位置，样本点第一个元素作为x坐标，第二个元素作为y坐标
    for p in x:
        draw_x.append(p[0])
        draw_y.append(p[1])
    #画散点图，其中支持向量呈现绿色，正类呈现红色，负类呈现蓝色
    #样本点离分割直线最近的为支持向量
    distance=zeros(len(y))
    for i in range(len(y)):
        distance[i]=distance_count(x[i],w,b)
    vector=findindex(distance,min(distance))
    for i in range(len(y)):
        if i==vector:
            draw_label.append('g')
        else:
            if y[i] > 0:
                draw_label.append('r')
            else:
                draw_label.append('b')
    plt.scatter(draw_x, draw_y, color=draw_label)
    #画分割平面（直线）
    #由于样本点的横坐标大概分布在4.3-7.0，纵坐标大概分布在1.8-4.5
    #因此将分割直线的横坐标范围设定为4-7
    plain_x = range(4, 8, 1)
    plain_y = []
    for i in plain_x:
        temp = double(-(w[0] * i + b) / w[1])
        plain_y.append(temp)
    plt.plot(plain_x, plain_y)
    #最终绘图
    plt.savefig('SMO.jpg')
    plt.show()
if __name__ == '__main__':
    train()