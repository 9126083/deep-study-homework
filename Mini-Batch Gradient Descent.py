#coding=utf-8
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

training_inputs = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])#4行3列
training_outputs = np.random.uniform(-1,1,(3,1))#3行一列
training_answers = np.array([[0,0,1,1]]).T#4行一列

def Delta_Mini_BGD(training_outputs,training_inputs,training_answers):
    alpha = 0.9
    for i in range(0,3,2):
        v=np.zeros((2,1))#列向量
        y=np.zeros((2,1))#列向量
        delta = np.zeros((2,1))#列向量
        v=np.matmul(training_inputs[i:i+2,] , training_outputs)#前两组数据影响权值
        for j in range(2):
            y[j] = sigmoid(v[j])
            delta[j] = sigmoid_derivative(y[j])
        e = training_answers[i:i+2]-y
        delta = np.multiply(e,delta)/2
        for j in range(3):
            training_outputs[j] += alpha * np.matmul(delta.T,training_inputs[i:i+2,j].T)
    print(f"调整后的权值为：{training_outputs}")
    return training_outputs

for i in range(2000):
    training_outputs = Delta_Mini_BGD(training_outputs,training_inputs,training_answers)

for i in range(4):
    x = training_inputs[i]
    v = np.matmul(x , training_outputs)
    y = sigmoid(v)
    print(f"输入值为：{x}")
    print(f"输出值为：{y}")
    print("\n")

print(f"最终的权值为：{training_outputs}")