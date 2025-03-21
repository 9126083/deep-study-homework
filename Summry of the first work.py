#coding=utf-8
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# 定义sigmoid函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 定义训练数据
training_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
training_answers = np.array([[0, 0, 1, 1]]).T

# 批量梯度下降
def DeltaBGD(training_outputs, training_inputs, training_answers):
    alpha = 0.9
    v = np.zeros((4, 1))
    y = np.zeros((4, 1))
    delta = np.zeros((4, 1))
    v = np.matmul(training_inputs, training_outputs)
    for i in range(4):
        y[i] = sigmoid(v[i])
        delta[i] = sigmoid_derivative(y[i])
    e = training_answers - y
    delta = np.multiply(e, delta) / 4
    for i in range(3):
        training_outputs[i] += alpha * np.matmul(delta.T, training_inputs[:, i].T)
    return training_outputs

# 小批量梯度下降
def Delta_Mini_BGD(training_outputs, training_inputs, training_answers):
    alpha = 0.9
    for i in range(0, 3, 2):
        v = np.zeros((2, 1))
        y = np.zeros((2, 1))
        delta = np.zeros((2, 1))
        v = np.matmul(training_inputs[i:i + 2, ], training_outputs)
        for j in range(2):
            y[j] = sigmoid(v[j])
            delta[j] = sigmoid_derivative(y[j])
        e = training_answers[i:i + 2] - y
        delta = np.multiply(e, delta) / 2
        for j in range(3):
            training_outputs[j] += alpha * np.matmul(delta.T, training_inputs[i:i + 2, j].T)
    return training_outputs

#单次一轮
def DeltaSGD(training_outputs, training_inputs, training_answers):
    alpha = 0.9
    v = 0
    y = 0
    x = training_inputs
    v = np.matmul(x, training_outputs)
    y = sigmoid(v)
    e = training_answers - y
    delta = e * sigmoid_derivative(y)
    for i in range(len(x)):
        training_outputs[i] += alpha * delta * x[i]
    return training_outputs

# 初始化权值，让三种方法使用相同的初始化权值
initial_weights = np.random.uniform(-1, 1, (3, 1))
training_outputs_bgd = initial_weights.copy()
training_outputs_mbgd = initial_weights.copy()
training_outputs_sgd = initial_weights.copy()

# 训练轮数
epochs_bgd = 4000
epochs_mbgd = 2000
epochs_sgd = 2500

# 存储每一轮的误差
errors_bgd = []
errors_mbgd = []
errors_sgd = []

# 批量梯度下降训练
for i in range(epochs_bgd):
    training_outputs_bgd = DeltaBGD(training_outputs_bgd, training_inputs, training_answers)
    v = np.matmul(training_inputs, training_outputs_bgd)
    y = sigmoid(v)
    e = training_answers - y
    # 计算误差平方和
    error = np.sum(np.square(e))
    errors_bgd.append(error)

# 小批量梯度下降训练
for i in range(epochs_mbgd):
    training_outputs_mbgd = Delta_Mini_BGD(training_outputs_mbgd, training_inputs, training_answers)
    v = np.matmul(training_inputs, training_outputs_mbgd)
    y = sigmoid(v)
    e = training_answers - y
    # 计算误差平方和
    error = np.sum(np.square(e))
    errors_mbgd.append(error)

# 随机梯度下降训练
for i in range(epochs_sgd):
    for j in range(4):
        training_outputs_sgd = DeltaSGD(training_outputs_sgd, training_inputs[j], training_answers[j])
    v = np.matmul(training_inputs, training_outputs_sgd)
    y = sigmoid(v)
    e = training_answers - y
    # 计算误差平方和
    error = np.sum(np.square(e))
    errors_sgd.append(error)

# 可视化“轮 - 误差”曲线
plt.figure(figsize=(10, 6))
plt.plot(range(epochs_bgd), errors_bgd, label='BGD')
plt.plot(range(epochs_mbgd), errors_mbgd, label='BGD_Mini')
plt.plot(range(epochs_sgd), errors_sgd, label='SGD')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Gradient Descent Methods')
plt.legend()
plt.grid(True)
plt.show()

# 输出最终权值
print(f"批量梯度下降最终的权值为：{training_outputs_bgd}")
print(f"小批量梯度下降最终的权值为：{training_outputs_mbgd}")
print(f"随机梯度下降最终的权值为：{training_outputs_sgd}")