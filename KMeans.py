# coding=utf-8
import random
import numpy
import matplotlib.pyplot as plt
import time

start = time.clock()  # 计时

# 准备数据集
dataSet = []  # 数据集
file = open("数据集.txt")

for line in file.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])


# K-Means算法函数
def KMeans(dataSet, k):
    firstVector = random.sample(dataSet, k)  # 准备初始均值向量
    c_last = []
    mTimes = 1000000  # 最大迭代次数
    for turns in range(mTimes):
        c = [[] for i in range(k)]  # 本轮聚类的结果集

        distance = [[1 for i in range(k)] for i in range(len(dataSet))]  # 记录每个点到每个聚类中心的距离

        for i in range(len(dataSet)):
            for j in range(k):
                # 计算欧氏距离
                length_x_sq = pow(dataSet[i][0] - firstVector[j][0], 2)
                length_y_sq = pow(dataSet[i][1] - firstVector[j][1], 2)
                distance_temp = numpy.sqrt(length_x_sq + length_y_sq)
                distance[i][j] = distance_temp

        print(distance)
        for i in range(len(dataSet)):
            minValue = min(distance[i])
            c[distance[i].index(minValue)].append(dataSet[i])

        print(c)
        # 更新均值向量
        j = 0
        while j < k:
            firstVector[j][0] = sum(c[j][0]) / len(c[j][0])
            firstVector[j][1] = sum(c[j][1]) / len(c[j][1])
            j += 1

        # 如果结果集不再更新，证明已经收敛，退出循环
        if c == c_last:
            print(turns)
            break
        c_last = c
    return c


result = KMeans(dataSet, 4)
# print(len(result[0]))
# print(len(result[1]))
# print(len(result[2]))
# print(len(result[3]))

# 画图
markers = ['x', '*', '+', '^', 'o']
for i in range(4):
    for j in range(len(result[i])):
        plt.scatter(result[i][j][0], result[i][j][1], s=60, marker=markers[i], c='b', alpha=0.5)
plt.title('K-Means')
plt.show()
plt.savefig('KMeans.png')

elapsed = (time.clock() - start)
print(elapsed)
