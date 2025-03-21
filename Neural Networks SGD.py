#coding=utf-8
import numpy as np
import random as rd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def DeltaSGD(training_outputs,training_inputs,training_answers):
    alpha = 0.9
    v=0
    y=0
    x = training_inputs
    v = np.matmul(x , training_outputs)
    y = sigmoid(v)
    e = training_answers-y
    delta = e*sigmoid_derivative(y)
    for i in range(len(x)):
        training_outputs[i] += alpha*delta*x[i]
    print(f"调整后的权值为：{training_outputs}")
    print(f"输出值为：{y}")
    return training_outputs

training_inputs = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
training_outputs = np.array([[rd.uniform(-1,1),rd.uniform(-1,1),rd.uniform(-1,1)]]).T
training_answers = np.array([[0,0,1,1]]).T

for i in range(2500):
    for j in range(4):
        training_outputs=DeltaSGD(training_outputs,training_inputs[j],training_answers[j])

for i in range(4):
    x = training_inputs[i]
    v = np.matmul(x , training_outputs)
    y = sigmoid(v)
    print(f"输入值为：{x}")
    print(f"输出值为：{y}")
    print("\n")

print(f"最终的权值为：{training_outputs}")