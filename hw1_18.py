# Author: Wenxing Deng
# CreateTime: 2020/7/13
# FileName: hw1_18
# Description: Run the pocket algorithm with a total of 50 updates on D, and verify the performance of
# wpocket using the test set. Please repeat your experiment for 2000 times, each with a different
# random seed. What is the average error rate on the test set? Plot a histogram to show error rate
# versus frequency.
# 进行2000轮，每一次都在训练集上得到w_pocket,然后在随机的测试集上测试一轮。
# 2000轮与naive cycle不同，naive要一直循环到无错，最后的循环次数与2000的大小关系不确定。

import numpy as np
import pandas as pd
# from numpy import random
import matplotlib.pyplot as plt
import copy


def PLA_training(train_data, row_num, col_num):
    round_count = 50
    # 初始化权值函数，全部权值为0。传递进PLA函数的列数没包括x0列，而权值的个数正好是所有x变量的个数，不包括y。
    w = np.zeros([col_num, 1])
    w_best = np.zeros([col_num, 1])
    # 初始化更新次数为数据行数
    best_update = row_num

    while True:
        for i in range(row_num):
            # 取出当前计算的数据行
            data = train_data[i]
            y = data[-1]
            x = data[0:-1]
            # dot的两个对象必须是维数相同的，所以把w变为了一个行向量
            if w.reshape(1, col_num).dot(x) * y <= 0:
                # w_best = w
                w += 0.5 * float(y) * x.reshape(col_num, 1)
                round_count -= 1
                update = 0
                for j in range(row_num):
                    data_1 = train_data[j]
                    y_1 = data_1[-1]
                    x_1 = data_1[0:-1]
                    if w.reshape(1, col_num).dot(x_1) * y_1 <= 0:
                        update += 1
                if update < best_update:
                    # w_best与w都改变
                    w_best = copy.deepcopy(w)
                    best_update = update

                if round_count == 0:
                    break
        if round_count == 0:
            break
    return w_best


# 将w_pocket取来用，但是每一轮仍要更新w。
def PLA_testing(test_data, w, row_num, col_num):
    # 初始化权值函数，全部权值为0。传递进PLA函数的列数没包括x0列，而权值的个数正好是所有x变量的个数，不包括y。
    error_count = 0

    for i in range(row_num):
        # 取出当前计算的数据行
        data = test_data[i]
        y = data[-1]
        x = data[0:-1]
        # dot的两个对象必须是维数相同的，所以把w变为了一个行向量
        if w.reshape(1, col_num).dot(x) * y <= 0:
            w += float(y) * x.reshape(col_num, 1)
            error_count += 1

    return error_count


def main():
    sum = 0
    errors = []

    # random.seed(j)
    cols = ['x1', 'x2', 'x3', 'x4', 'y']
    # \s代表正则表达式中的一个空白字符(可能是空格、制表符、其他空白)
    df_train = pd.read_csv('hw1_18_train.dat', header=None, sep='\s', engine='python', \
                           dtype={'x1,x2,x3,x4,y': np.float})
    row_train, col = df_train.shape
    # 任何合并函数都只能有一个输入，因此两个合并的array外面还要加一个括号
    train_data = np.column_stack((np.ones([row_train, 1]), np.array(df_train)))

    cols = ['x1', 'x2', 'x3', 'x4', 'y']
    # \s代表正则表达式中的一个空白字符(可能是空格、制表符、其他空白)
    df_test = pd.read_csv('hw1_18_test.dat', header=None, sep='\s', engine='python', \
                          dtype={'x1,x2,x3,x4,y': np.float})
    row_test, col = df_test.shape
    # 任何合并函数都只能有一个输入，因此两个合并的array外面还要加一个括号
    test_data = np.column_stack((np.ones([row_test, 1]), np.array(df_test)))

    for j in range(2000):
        np.random.shuffle(train_data)
        # np.random.shuffle(train_data)
        w_pocket = PLA_training(train_data, row_train, col)
        # np.random.shuffle(test_data)
        error = PLA_testing(test_data, w_pocket, row_test, col)

        errors.append(error / row_test)
        sum += error / row_test
    average = sum / 2000
    print("The average number of updates before the algorithm halts is %f" % average)
    min_update = min(errors)
    max_update = max(errors)
    x_axis = [x for x in np.arange(min_update, max_update + 0.001, 0.001)]
    bins = (max_update - min_update) / 0.001
    # y_axis = [0]*len(x_axis)
    # for i in range(2000):
    # y_axis[ updates[i] - min_update] += 1
    x_data = np.array(errors)
    # y_data = np.array(y_axis)

    # bins指定的是直方图的条数
    plt.hist(x=x_data, bins=int(bins), normed=1, rwidth=0.1)
    plt.xticks(x_axis, rotation='vertical', size=5)
    plt.xlabel('Error Rate')
    plt.ylabel('Frequency')
    plt.title('Error Rate Versus Frequency')
    plt.show()
    # 这句语句在直方图关闭后执行。
    print("The average error rate on the test set %f" % average)


if __name__ == '__main__':
    main()
