import numpy as np
from matplotlib import pyplot as plt
import find_number as fn

try:
    # 尝试加载权重矩阵
    W1 = np.loadtxt(r"D:\txt\W1.txt")
    W2 = np.loadtxt(r"D:\txt\W2.txt")
except FileNotFoundError:
    print("权重文件未找到，请检查文件路径。")
    exit(1)

x = np.zeros((5,5,5))#测试数据
for i in range(5):
    x[:,:,i] = np.random.randint(0, 2, size=(5, 5))

def text(x):
    y = np.zeros((5, 1, 5))
    for i in range(5):
        text_th = x[:,:,i]
        col = text_th.ravel().reshape(-1,1)
        v1 = np.matmul(W1,col)
        y1 = fn.sigmoid(v1)
        v2 = np.matmul(W2,y1)
        y[:,:,i] = fn.Softmax(v2)
        print(f"样本：\n{text_th}")
        print(f"识别出来为1的可能性为{y[0,0,i]}")
        print(f"识别出来为2的可能性为{y[1,0,i]}")
        print(f"识别出来为3的可能性为{y[2,0,i]}")
        print(f"识别出来为4的可能性为{y[3,0,i]}")
        print(f"识别出来为5的可能性为{y[4,0,i]}")
    return y

text(x)