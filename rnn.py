"""
@Author: yu_qian_ran

@note: RNN demo
"""
import numpy as np
sentence = []
len_x = 0
w = np.random.random((1, 4))*2 - 1
wy = np.random.random((1, 2))*2 - 1
lb = np.array([[0., 0., 0., 0., 0., 1., 1., 1., 1.]])
lr = 0.0097


# 读取汉字对应表
def convert_sentence(some_word):
    global sentence, len_x, w
    table_word = {}
    with open("word_list.txt", "r", encoding="utf-8") as word_list:
        for m in word_list:
            tmp = m.split(" ")
            table_word[tmp[2].strip()] = [float(tmp[0]), float(tmp[1])]
    for n in some_word:
        unit = table_word.get(n)
        sentence.append(unit)
    len_x = len(some_word)
    sentence = np.array(sentence)


# logistic 函数
def sigmoid(value):
    return 1/(1+np.exp(-value))


def dsigmoid(value):
    return sigmoid(value)*(1 - sigmoid(value))


# RELU 函数
def relu(value):
    tmp = value.copy()
    tmp[tmp < 0] = 0.1*tmp[tmp < 0]
    return tmp


def drelu(value):
    tmp = value.copy()
    tmp[tmp > 0] = 1
    tmp[tmp < 0] = 0.1
    return tmp


# back propagation through time
def bptt():
    global sentence, y, wy, w, ax_array, at_array
    # 向前传播
    t = 0
    a = {0: np.array([[0., 0.]])}
    ax_array = []
    at_array = []
    y = []
    for _ in range(len_x):
        ax = np.concatenate((a[t], np.array([sentence[t]])), axis=1).T
        ax_array.append(ax[:, 0])
        at = relu(np.dot(w, ax))
        at = np.concatenate((at, np.array([[1]])), axis=1)
        at_array.append(at[0])
        t += 1
        a[t] = at
        yt = sigmoid(np.dot(wy, at.T))
        y.append([yt[0][0]])
    ax_array = np.array(ax_array)
    at_array = np.array(at_array)
    y = np.array(y)
    # 反向传播
    wy -= lr*(y.T - lb).dot(at_array)
    w -= lr*((y.T - lb)*drelu(np.dot(w, ax_array.T))).dot(ax_array)
    print(np.mean(np.abs(y.T-lb)))


def operate(num):
    global y
    for _ in range(num):
        bptt()


def main():
    convert_sentence("她的名字叫绚丽多彩")
    operate(1000)
    print(y)


main()
