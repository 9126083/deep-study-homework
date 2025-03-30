import numpy as np

# 第一层个数
NUM_1 = 3
# 第二层个数
NUM_2 = 2
alpha = 0.9
training_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
W1 = np.random.uniform(-1, 1, (NUM_2, NUM_1))  # 第一层权值
W2 = np.random.uniform(-1, 1, (1, NUM_2))  # 第二层权值
training_answers = np.array([[0, 1, 1, 0]]).T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def SGD_double_neural_network(training_inputs, W1, W2, answer):
    for k, ans in zip(training_inputs, answer):
        v1 = np.matmul(W1, k)  # 矩阵乘法，第一层神经网络输入（NUM_2*1矩阵）
        y1 = sigmoid(v1)  # 第二层神经网络输出
        v2 = np.matmul(W2, y1)  # 第三层神经网络输入
        y2 = sigmoid(v2)  # 第三层神经网络输出
        e2 = ans - y2  # 最终偏差
        δ2 = e2 * sigmoid_derivative(y2)  # 第三层对第二层的δ值
        e1 = np.matmul(δ2, W2)  # 传播至第一层的误差（1*NUM_2的矩阵）
        δ1 = e1.T * sigmoid_derivative(y1)  # 第二层对第一层的δ值(NUM_2*1)
        dW1 = alpha * np.outer(δ1, k)
        W1 = W1 + dW1
        dW2 = alpha * np.outer(δ2, y1)
        W2 = W2 + dW2
    return W1, W2


for i in range(999999):
    W1, W2 = SGD_double_neural_network(training_inputs, W1, W2, training_answers)

print(f"W1的权值为{W1},W2的权值为{W2}")
for i, j in zip(training_inputs, training_answers):
    v1 = np.matmul(W1, i)  # 矩阵乘法，第一层神经网络输入（NUM_2*1矩阵）
    y1 = sigmoid(v1)  # 第二层神经网络输出
    v2 = np.matmul(W2, y1)  # 第三层神经网络输入
    y2 = sigmoid(v2)  # 第三层神经网络输出
    print(f"输入的是{i}\n结果为{y2}")
    print(f"正确的答案为{j}")
    