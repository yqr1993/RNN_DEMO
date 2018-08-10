"""
@Author: yu_qian_ran

@note: BIRNN demo

@note：我是中国人
      [1., 1., 1., 1., 1.]
"""
import numpy as np
sentence = np.array([[1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]], "float64")
sentence_r = np.array([[0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0]], "float64")
len_x = 5
w = np.random.random((1, 7))*2 - 1
w_r = np.random.random((1, 7))*2 - 1
wy = np.random.random((1, 4))*2 - 1
# w = np.array()
# w_r = np.array()
# wy = np.array()
# print(w, w_r, wy)
lb = np.array([[0., 0., 1., 1., 0.]])
lr = 0.11


# 读取汉字对应表
def convert_sentence(some_word):
    global sentence, sentence_r, len_x, w
    table_word = {}
    with open("word_list.txt", "r", encoding="utf-8") as word_list:
        for m in word_list:
            tmp = m.split(" ")
            table_word[tmp[2].strip()] = [float(tmp[0]), float(tmp[1])]
    for n in some_word:
        unit = table_word.get(n)
        sentence.append(unit)
        sentence_r.append(unit)
    len_x = len(some_word)
    sentence = np.array(sentence)
    sentence_r.reverse()
    sentence_r = np.array(sentence_r)


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
    tmp[tmp > 0] = 1
    tmp[tmp < 0] = 0.01
    return tmp


# back propagation through time
def bptt():
    global sentence, sentence_r, y, wy, w, w_r, ax_array, at_array
    # 输出层参数
    y = []
    a_concatenate_array = []
    # 向前传播,从左往右方向参数
    t = 0
    a = {0: np.array([[0., 0.]])}
    ax_array = []
    at_array = []
    # 向前传播,从右往左方向参数
    t_r = 0
    a_r = {0: np.array([[0., 0.]])}
    ax_array_r = []
    at_array_r = []
    for _ in range(len_x):
        # 从左往右方向,计算正方向at
        ax = np.concatenate((a[t], np.array([sentence[t]])), axis=1).T
        ax_array.append(ax[:, 0])
        at = relu(np.dot(w, ax))
        at = np.concatenate((at, np.array([[1]])), axis=1)
        at_array.append(at[0])
        t += 1
        a[t] = at
        # 从右往左方向，计算反方向at_r
        ax_r = np.concatenate((a_r[t_r], np.array([sentence_r[t_r]])), axis=1).T
        ax_array_r.append(ax_r[:, 0])
        at_r = relu(np.dot(w_r, ax_r))
        at_r = np.concatenate((at_r, np.array([[1]])), axis=1)
        at_array_r.append(at_r[0])
        t_r += 1
        a_r[t_r] = at_r
    # 输出层，计算合并a与a_r得到y
    for index in range(len_x):
        a_concatenate = np.concatenate((a[index + 1], a_r[len_x - index]), axis=1)
        a_concatenate_array.append(a_concatenate[0])
        yt = sigmoid(np.dot(wy, a_concatenate.T))
        y.append([yt[0][0]])
    # 将各个参数转化为矩阵
    ax_array = np.array(ax_array)
    ax_array_r = np.array(ax_array_r)
    a_concatenate_array = np.array(a_concatenate_array)
    y = np.array(y)
    by = np.ones((5, 1), "float64")
    # 反向传播
    wy -= lr*(y.T - lb).dot(a_concatenate_array)
    dat = np.concatenate((drelu(np.dot(ax_array, w.T)), by), axis=1)
    w -= lr*((y.T - lb)*np.dot(wy[:, 0:2], dat.T)).dot(ax_array)
    dat_r = np.concatenate((drelu(np.dot(ax_array_r, w_r.T)), by), axis=1)
    w_r -= lr*((y.T - lb)*np.dot(wy[:, 2:4], dat_r.T)).dot(ax_array_r)
    print(np.mean(np.abs(y.T-lb)))


def operate(num):
    global y
    for _ in range(num):
        bptt()


# 训练权值写入磁盘
def memory(*para):
    index = 1
    for ww in para:
        file_w = open("bw" + str(index) + ".txt", "w")
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
    w_test = load_param("bw1.txt")
    w_r_test = load_param("bw2.txt")
    wy_test = load_param("bw3.txt")
    # 输出层参数
    y_test = []
    a_concatenate_array_test = []
    # 向前传播,从左往右方向参数
    t = 0
    a = {0: np.array([[0., 0.]])}
    ax_array_test = []
    at_array_test = []
    # 向前传播,从右往左方向参数
    t_r = 0
    a_r = {0: np.array([[0., 0.]])}
    ax_array_r = []
    at_array_r = []
    for _ in range(len_x):
        # 从左往右方向,计算正方向at
        ax = np.concatenate((a[t], np.array([sentence[t]])), axis=1).T
        print(a[t])
        ax_array_test.append(ax[:, 0])
        at = relu(np.dot(w_test, ax))
        at = np.concatenate((at, np.array([[1]])), axis=1)
        at_array_test.append(at[0])
        t += 1
        a[t] = at
        # 从右往左方向，计算反方向at_r
        ax_r = np.concatenate((a_r[t_r], np.array([sentence_r[t_r]])), axis=1).T
        ax_array_r.append(ax_r[:, 0])
        at_r = relu(np.dot(w_r_test, ax_r))
        at_r = np.concatenate((at_r, np.array([[1]])), axis=1)
        at_array_r.append(at_r[0])
        t_r += 1
        a_r[t_r] = at_r
    # 输出层，计算合并a与a_r得到y
    for index in range(len_x):
        a_concatenate = np.concatenate((a[index + 1], a_r[len_x - index]), axis=1)
        a_concatenate_array_test.append(a_concatenate[0])
        yt = sigmoid(np.dot(wy_test, a_concatenate.T))
        y_test.append([yt[0][0]])
    y_test = np.array(y_test)
    name = []
    index = 0
    for n in y_test:
        if n >= 0.5:
            name.append(some_word[index])
        index += 1
    print(y_test)
    print("".join(name))


# 进行模型训练
def main():
    # convert_sentence("她叫丽多彩这名字")
    operate(600)
    memory(w, w_r, wy)
    print(y)


if __name__ == "__main__":
    main()
    # test("他叫牛二这名字")
