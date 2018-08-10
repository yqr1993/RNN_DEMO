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
lr = 0.011


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
    tmp[tmp < 0] = 0.01*tmp[tmp < 0]
    return tmp


def drelu(value):
    tmp = value.copy()
    tmp[tmp > 0] = 1.0
    tmp[tmp < 0] = 0.01
    return tmp


# back propagation through time
def bptt():
    global sentence, y, wy, w, ax_array, at_array
    # 向前传播,各个参数
    t = 0
    a = {0: np.array([[0., 0.]])}
    ax_array = []
    at_array = []
    y = []
    # 计算各个参数，并转化为矩阵
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
    by = np.ones((9, 1), "float64")
    dat = np.concatenate((drelu(np.dot(ax_array, w.T)), by), axis=1)
    w -= lr*((y.T - lb)*np.dot(wy, dat.T)).dot(ax_array)
    print(np.mean(np.abs(y.T-lb)))


def operate(num):
    global y
    for _ in range(num):
        bptt()


# 训练权值写入磁盘
def memory(*para):
    index = 1
    for ww in para:
        file_w = open("w" + str(index) + ".txt", "w")
        file_w_buffer = ww.tolist()
        for i in file_w_buffer:
            for j in i:
                file_w.write(str(j))
                file_w.write(" ")
            file_w.write("\n")
        file_w.close()
        index += 1


# 从磁盘读取神经网络
def load_param(file_name):
    file = open(file_name)
    lines = file.readlines()
    nl = len(lines)
    line = lines[0].split(" ")
    line.pop()
    nr = len(line)
    weight = np.ones((nl, nr))
    for m, i in zip(lines, range(nl)):
        weight_line = m.split(" ")
        weight_line.pop()
        for n, j in zip(weight_line, range(nr)):
            weight[i][j] = n
    file.close()
    return weight


# 验证函数
def test(some_word):
    convert_sentence(some_word)
    # 向前传播
    w_test = load_param("w1.txt")
    wy_test = load_param("w2.txt")
    t = 0
    a = {0: np.array([[0., 0.]])}
    ax_array_test = []
    at_array_test = []
    y_test = []
    for _ in range(len_x):
        ax = np.concatenate((a[t], np.array([sentence[t]])), axis=1).T
        ax_array_test.append(ax[:, 0])
        at = relu(np.dot(w_test, ax))
        at = np.concatenate((at, np.array([[1]])), axis=1)
        at_array_test.append(at[0])
        t += 1
        a[t] = at
        yt = sigmoid(np.dot(wy_test, at.T))
        y_test.append([yt[0][0]])
    y_test = np.array(y_test)
    name = []
    index = 0
    for n in y_test:
        if n >= 0.5:
            name.append(some_word[index])
        index += 1
    print("".join(name))


# 进行模型训练
def main():
    convert_sentence("她的名字叫绚丽多彩")
    operate(8000)
    memory(w, wy)
    print(y)


if __name__ == "__main__":
    # main()
    # test("他叫牛小二")
    test("他牛小二")
    # test("他的名字叫牛小二")
