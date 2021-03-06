# Author: Wenxing Deng
# CreateTime: 
# FileName: 
# Description:Implement a version of PLA by visiting examples in fixed, pre-determined random cycles
# throughout the algorithm, while changing the update rule to be wt+1 <- wt + ηyn(t)xn(t)
# with η = 0.5. Note that your PLA in the previous problem corresponds to η = 1. Please repeat
# your experiment for 2000 times, each with a different random seed. What is the average number
# of updates before the algorithm halts? Plot a histogram to show the number of updates versus
# frequency. Compare your result to the previous problem and briefly discuss your findings.

import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt

def PLA_random(train_data,row_num,col_num):
    # 初始化权值函数，全部权值为0。传递进PLA函数的列数没包括x0列，而权值的个数正好是所有x变量的个数，不包括y。

    w = np.zeros([col_num, 1])
    update_count = 0
    while True:
        k = 0
        for i in range(row_num):
           # 取出当前计算的数据行
            data = train_data[i]
            y = data[-1]
            x = data[0:-1]
           # dot的两个对象必须是维数相同的，所以把w变为了一个行向量
            if w.reshape(1,col_num).dot(x) * y <= 0:
                w +=  0.5 * float(y) * x.reshape(col_num, 1)
                update_count += 1
            else:
                k += 1
        if k == row_num:
            break

    return update_count



def main():
    cols = ['x1','x2','x3','x4','y']
    # \s代表正则表达式中的一个空白字符(可能是空格、制表符、其他空白)
    df = pd.read_csv('hw1_15_train.dat',header = None,sep = '\s' ,engine = 'python', \
                     dtype={'x1,x2,x3,x4,y': np.float})
    row,col = df.shape
    #任何合并函数都只能有一个输入，因此两个合并的array外面还要加一个括号
    train_data = np.column_stack( (np.ones([row,1]),np.array(df)) )
    sum = 0
    updates = []
    for j in range(2000):
        random.seed(j)
        #这里发现一定要在前面加上np，否则后面的循环是会发现测试数据出现大量重复的数据。
        random.shuffle(train_data)
        update = PLA_random(train_data,row,col)
        updates.append(update)
        sum += update
    average = sum/2000
    min_update = min(updates)
    max_update = max(updates)
    x_axis = [x for x in range(min_update, max_update+1)]
    bins = len(x_axis)

    x_data = np.array(updates)
    #y_data = np.array(y_axis)

    # bins指定的是直方图的条数
    plt.hist(x = x_data, bins = bins, density=1, rwidth= 0.1)
    plt.xticks(x_axis, rotation = 'vertical', size = 5)
    plt.xlabel('Updates')
    plt.ylabel('Frequency')
    plt.title('The Number of Updates Versus Frequency')
    plt.show()
    # 这句语句在直方图关闭后执行。
    print("The average number of updates before the algorithm halts is %f" %average)

if __name__ == '__main__':
    main()