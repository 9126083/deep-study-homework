import numpy as np
import matplotlib.pyplot as plt

#第一层个数
NUM_1 = 3
#第二层个数
NUM_2 = 4
alpha = 0.9
beta = 0.9
training_inputs = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
W1 = np.random.uniform(-1,1,(NUM_2,NUM_1))#第一层权值,(5,3)
W2 = np.random.uniform(-1,1,(1,NUM_2))#第二层权值
W3 = W1
W4 = W2
training_answers = np.array([[0,1,1,0]]).T
flag = 1
myself_list = []#我自己的数据
teacher_list = []#老师的数据

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def myself(training_inputs,W1,W2,answer,flag):
    mmt1 = np.zeros_like(W1)
    mmt2 = np.zeros_like(W2)
    if flag == 1:
        answer = reversed(answer)
        training_inputs = reversed(training_inputs)
    for k, ans in zip(training_inputs, answer):
        v1 = np.matmul(W1,k)#矩阵乘法，第一层神经网络输入（NUM_2*1矩阵）
        y1 = sigmoid(v1)#第二层神经网络输出
        v2 = np.matmul(W2,y1)#第三层神经网络输入
        y2 = sigmoid(v2)#第三层神经网络输出
        e2 = ans - y2#最终偏差
        δ2 = e2 * sigmoid_derivative(y2)#第三层对第二层的δ值
        e1 = np.matmul(δ2, W2)#传播至第一层的误差（1*NUM_2的矩阵）
        δ1 = np.multiply(e1.T, sigmoid_derivative(y1))#第二层对第一层的δ值(NUM_2,1)
        dW1 = np.outer(δ1, k) * alpha
        mmt1 = np.multiply(mmt1, beta) +dW1
        W1 = W1 + mmt1
        dW2 = np.outer(δ2, y1) * alpha
        mmt2 = beta * mmt2 +dW2
        W2 = W2 + mmt2
        myself_list.append(e2)
    return W1,W2,flag*(-1)

def teacher(training_inputs,W1,W2,answer):
    mmt3 = np.zeros_like(W1)
    mmt4 = np.zeros_like(W2)
    for k, ans in zip(training_inputs, answer):
        v1 = np.matmul(W1,k)#矩阵乘法，第一层神经网络输入（NUM_2*1矩阵）
        y1 = sigmoid(v1)#第二层神经网络输出
        v2 = np.matmul(W2,y1)#第三层神经网络输入
        y2 = sigmoid(v2)#第三层神经网络输出
        e2 = ans - y2#最终偏差
        δ2 = e2 * sigmoid_derivative(y2)#第三层对第二层的δ值
        e1 = np.matmul(δ2, W2)#传播至第一层的误差（1*NUM_2的矩阵）
        δ1 = np.multiply(e1.T, sigmoid_derivative(y1))#第二层对第一层的δ值(NUM_2,1)
        dW1 = np.outer(δ1, k) * alpha
        mmt3 = np.multiply(mmt3, beta) +dW1
        W1 = W1 + mmt3
        dW2 = np.outer(δ2, y1) * alpha
        mmt4 = beta * mmt4 +dW2
        W2 = W2 + mmt4
        teacher_list.append(e2)
    return W1,W2

for i in range(10000):
    W1, W2, flag = myself(training_inputs,W1,W2,training_answers,flag)
    W3, W4 = teacher(training_inputs,W3,W4,training_answers)

print(f"W1的权值为{W1},W2的权值为{W2}")
for i, j in zip(training_inputs, training_answers):
    v1 = np.matmul(W1,i)#矩阵乘法，第一层神经网络输入（NUM_2*1矩阵）
    y1 = sigmoid(v1)#第二层神经网络输出
    v2 = np.matmul(W2,y1)#第三层神经网络输入
    y2 = sigmoid(v2)#第三层神经网络输出
    print(f"输入的是{i}\n结果为{y2}")
    print(f"正确的答案为{j}")
    
print(f"W3的权值为{W3},W4的权值为{W4}")
for i, j in zip(training_inputs, training_answers):
    v1 = np.matmul(W3,i)#矩阵乘法，第一层神经网络输入（NUM_2*1矩阵）
    y1 = sigmoid(v1)#第二层神经网络输出
    v2 = np.matmul(W4,y1)#第三层神经网络输入
    y2 = sigmoid(v2)#第三层神经网络输出
    print(f"输入的是{i}\n结果为{y2}")
    print(f"正确的答案为{j}")

n = range(1,len(myself_list)+1)
#创建图形和坐标轴
plt.figure(figsize=(16,9))
plt.plot(n,myself_list,label = 'Me',alpha = 0.8)
plt.plot(n,teacher_list,label = 'Teacher',alpha = 0.5)

plt.legend()
plt.show()