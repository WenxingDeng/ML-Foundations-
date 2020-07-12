# Author: Wenxing Deng
# CreateTime: 2020/7/8
# FileName: hw1_15
# Description:  initialize with w = 0 , sign(0) as 1 and add x0 = 1 as always
# Implement a version of PLA by visiting examples in the naive cycle using the order of examples
# in the data set. Run the algorithm on the data set. What is the number of updates before the
# algorithm halts? What is the index of the example that results in the last mistake?
# 根据题目要求，最后一轮循环时，需要在所有的数据上都判断正确。

import numpy as np
import pandas as pd


def PLA_naive(train_data,row_num,col_num):
    round_count = 0
    fault_num = 0
    #初始化权值函数，全部权值为0。传递进PLA函数的列数没包括x0列，而权值的个数正好是所有x变量的个数，不包括y。
    w = np.zeros([col_num,1])
    while True:
        k = 0
        for i in range(row_num):
           # 取出当前计算的数据行
            data = train_data[i]
            y = data[-1]
            x = data[0:-1]
           # dot的两个对象必须是维数相同的，所以把w变为了一个行向量
            if w.reshape(1,col_num).dot(x) * y <= 0:
                w +=  float(y) * x.reshape(col_num, 1)
                fault_num = i+1
                round_count += 1
            else:
                k += 1
        #round_count += 1
        if k == row_num:
            break
    # 一开始理解错了题意，以为update是指在所有数据条目上扫描过的遍数，所以一开始一直输出为3。
    print("The number of updates before the algorithm halts is %d" %round_count)
    print("The index of the example that results in the last mistake is %d" % fault_num)

cols = ['x1','x2','x3','x4','y']
# \s代表正则表达式中的一个空白字符(可能是空格、制表符、其他空白)
df = pd.read_csv('hw1_15_train.dat',header = None,sep = '\s' ,engine = 'python', \
                 dtype={'x1,x2,x3,x4,y': np.float})
row,col = df.shape
#x_data = float(df[:,:-1])
#y_data = float(df[:,-1])
#任何合并函数都只能有一个输入，因此两个合并的array外面还要加一个括号
train_data = np.column_stack( (np.ones([row,1]),np.array(df)) )
PLA_naive(train_data,row,col)




