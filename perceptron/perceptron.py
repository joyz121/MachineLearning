import numpy as np
import matplotlib.pyplot as plt

#数据集
training_set = [[(3,4),1],[(3,3),1],[(4,5),1],[(1, 2), 1], [(2, 3), 1], [(3, 1), -1], [(4, 2), -1],[(3,2),-1]]

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
    x,x_,y,y_=[],[],[],[]
    for p in training_set:
        if p[1] > 0:
            x.append(p[0][0])  # 存放yi=1的点的x1坐标
            y.append(p[0][1])  # 存放yi=1的点的x2坐标
        else:
            x_.append(p[0][0])  # 存放yi=-1的点的x1坐标
            y_.append(p[0][1])  # 存放yi=-1的点的x2坐标

    x1 = -6
    y1 = -(b + w[0,0] * x1) / w[0,1]
    x2 = 6
    y2 = -(b + w[0,0] * x2) / w[0,1]
    plt.plot([x1, x2], [y1, y2])  # 设置线的两个点

    plt.plot(x, y, 'bo', x_, y_, 'rx')  # 在图里yi=1的点用点表示，yi=-1的点用叉表示
    plt.axis([-6, 6, -6, 6])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Perceptron Algorithm')
    plt.show()

if __name__ == '__main__':
    preceptron(dataMat,label,i)
    show(w,b)