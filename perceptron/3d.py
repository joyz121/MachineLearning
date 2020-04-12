import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#数据集
training_set = [[(3,4,2),1],[(3,3,5),1],[(4,3,8),1],[(1, 2,1), 1], [(2, 3,4), 1],[(0,1.5,3),-1], [(3, 1,-1), -1], [(4, 2,2), -1],[(3,2,10),-1]]

#数据获取
data,label=[],[]

for i in training_set:
    data.append(i[0])
    label.append(i[1])

dataMat=np.mat(data)
#获得矩阵大小
m, n = np.shape(dataMat)
#参数初始化
w = np.zeros((1, np.shape(dataMat)[1]))
b=0
n=0.01#学习率

#data为1xn的矩阵，w为1xn的矩阵
def preceptron(dataset,labels,times):
    global w,b,n
    a=0
    print('start to train')
    #梯度下降
    for k in range(times):
        flag=False
        for i in range(m):
            xi=dataset[i]
            yi=labels[i]
            #错误分类
            if yi*(w*xi.T+b)[0,0]<=0:
                w=w+n*yi*xi
                b=b+n*yi
                flag=True
                
                a=a+1
                print('第 %d 次迭代'%a)
                print('w=',w,'b=',b)
        if not flag:
            break;   

    return w,b

#可视化
def show(w,b):
    x,x_,y,y_,z,z_=[],[],[],[],[],[]
    for p in training_set:
        if p[1] > 0:
            x.append(p[0][0])  # 存放yi=1的点的x1坐标
            y.append(p[0][1])  # 存放yi=1的点的x2坐标
            z.append(p[0][2])
        else:
            x_.append(p[0][0])  # 存放yi=-1的点的x1坐标
            y_.append(p[0][1])  # 存放yi=-1的点的x2坐标
            z_.append(p[0][2])
        
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_alpha(.4)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z=  -(b + w[0,0] * X+w[0,1]*Y) / w[0,2]
    ax.plot_surface(X,Y,Z,rstride = 1, cstride = 1,alpha=0.5)
    
    ax.scatter(x, y, z,c='r',marker='o', label='正样本')
    ax.scatter(x_, y_, z_,c='b',marker='x', label='负样本')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


if __name__ == '__main__':
    preceptron(dataMat,label,200)
    show(w,b)