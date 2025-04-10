import numpy as np
from matplotlib import pyplot as plt

np.random.seed(100)
ALGHA = 0.9#algha值
NUM_LENGTH_SIZE = 5#图像的横轴大小
NUM_LENGTH_SIZE_T = 5 #图像的纵轴大小
NUMBER = 5 #要分类的目的种类多少
training_inputs = np.zeros((NUM_LENGTH_SIZE,NUM_LENGTH_SIZE_T,NUMBER))
training_inputs[:,:,0] = [[0, 1, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0]]
training_inputs[:,:,1] = [[1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 1],
                          [0, 1, 1, 1, 0],
                          [1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1]]
training_inputs[:,:,2] = [[1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 0]]
training_inputs[:,:,3] = [[0, 0, 0, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 1, 0, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 0, 0, 1, 0]]
training_inputs[:,:,4] = [[1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 0]]
#答案
d = np.array([[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1]])
N1 = NUM_LENGTH_SIZE * NUM_LENGTH_SIZE_T
N2 = N1 * 2#第二层节点数
N3 = NUMBER#第三层（输出层节点数）
W1 = np.random.uniform(-1,1,(N2, N1))#假设隐层节点数为输入层的两倍
W2 = np.random.uniform(-1,1,(N3, N2))
error = []#储存误差
error_abs = []#绝对误差
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def Softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex)
    
def MultiClass(W1,W2,training_inputs,D):
    for i in range(NUMBER):
        temp = training_inputs[:,:,i]
        ans = np.array([D[i]])#将ans二维向量化
        col = temp.ravel().reshape(-1,1)
        v1 = np.matmul(W1,col)
        y1 = sigmoid(v1)
        v2 = np.matmul(W2,y1)
        y = Softmax(v2)#归一化
        e = ans.T - y
        delta = e
        # print(f"ans是{ans}")
        # print(f"y是{y}")
        # print(f"e是{e}")
        # print(f"v2是{v2}")
        e1 = np.matmul(W2.T,delta)
        delta1 = np.multiply(sigmoid_derivative(y1),e1)
        # print(f"delta1是{delta1}")
        # print(f"sigmoid_derivative(y1)是{sigmoid_derivative(y1)}")
        # print(f"e1是{e1}")
        W1 = W1 + ALGHA*np.matmul(delta1,col.T)
        W2 = W2 + ALGHA*np.matmul(delta,y1.T)
        error.append(np.sum(e))
        error_abs.append(np.sum(np.abs(e)))
    return W1, W2
        
            

for i in range(10000):
    W1, W2 = MultiClass(W1,W2,training_inputs,d)


# n = range(1,len(error)+1)
# #创建图形和坐标轴
# plt.figure(figsize=(16,9))
# #plt.plot(n,error,label = 'error',alpha = 0.5)
# plt.plot(n,error_abs,label = 'error_abs')
# plt.legend()
# plt.show()

np.savetxt(r"D:\txt\W1.txt",W1)
np.savetxt(r"D:\txt\W2.txt",W2)