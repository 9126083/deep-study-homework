#coding=utf-8
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

training_inputs = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
training_outputs = np.random.uniform(-1,1,(3,1))
training_answers = np.array([[0,0,1,1]]).T

def DeltaBGD(training_outputs,training_inputs,training_answers):
    alpha = 0.9
    v=np.zeros((4,1))
    y=np.zeros((4,1))
    delta = np.zeros((4,1))
    v=np.matmul(training_inputs , training_outputs)
    for i in range(4):
        y[i] = sigmoid(v[i])
        delta[i] = sigmoid_derivative(y[i])

    e = training_answers-y
    delta = np.multiply(e,delta)/4
    for i in range(3):
        training_outputs[i] += alpha * np.matmul(delta.T,training_inputs[:,i].T)
    print(f"调整后的权值为：{training_outputs}")
    return training_outputs

for i in range(4000):
    training_outputs = DeltaBGD(training_outputs,training_inputs,training_answers)

for i in range(4):
    x = training_inputs[i]
    v = np.matmul(x , training_outputs)
    y = sigmoid(v)
    print(f"输入值为：{x}")
    print(f"输出值为：{y}")
    print("\n")

print(f"最终的权值为：{training_outputs}")