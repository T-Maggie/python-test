"""
2.3.4    高级的通用函数特征
"""
import numpy as np

# 1、指定输出
x1 = np.arange(5)   # 构建等差数组
print(x1)
y1 = np.empty(5)    # 定义空数组
np.multiply(x1, 10, out=y1)    # 将数组x1*10，结果存放到y1
print(y1)

x2 = np.zeros(10)
np.power(2, x1, out=x2[::2])     # 将2^x1存放到x2
print(x2)

# 2、聚合
x3 = np.arange(1, 6)
y2 = np.add.reduce(x3)      # 对 add 通用函数调用 reduce 方法会返回数组中”所有元素的和“
y3 = np.multiply.reduce(x3)     # 对 multiply 通用函数调用 reduce 方法会返回数组中”所有元素的乘积“
y4 = np.add.accumulate(x3)      # 使用 accumulate 方法存储每次计算的中间结果
y5 = np.multiply.accumulate(x3)
print(y4, y5)

# 3、外积
x3 = np.arange(1, 6)
print(np.multiply.outer(x3, x3))     # 两个矩阵的乘积

# 2.4 聚合：最小值、最大值的其他值
"""
2.4.1   数组值求和
"""
L = np.random.random(100)
l1 = sum(L)    # 列表求和函数
print(l1)
l2 = np.sum(L)    # numpy 的 sum 函数求和
print(l2)


"""
2.4.2    最大值和最小值
"""
big_array = np.random.rand(1000000)
print(min(big_array))
print(max(big_array))

# Numpy 对应的函数语法（执行速度更快）
print(np.min(big_array))
print(np.max(big_array))
print(np.sum(big_array))

# 1、多维度聚合
M = np.random.random((3, 4))       # 生成3x4的数组，数组的元素是范围（0，1）的随机数
print(M)
print(M.sum())
print(M.min(axis=0))        # 通过指定axis=0 找到每一列的最小值
print(M.max(axis=1))        # 通过指定axis=0 找到每一行的最大值

"""
2.4.3   实例：美国总统的身高是多少
"""
import pandas as pd
data = pd.read_csv("president_heights.csv")
heights = np.array(data['height(cm)'])
print(heights)

# import matplotlib.pyplot as plt
# import seaborn; seaborn.set()
# plt.hist(heights)
# plt.title('height distribution of US presidents')
# plt.xlabel('height(cm)')
# plt.ylabel('number')
# plt.show()



"""
5、  数值的计算：广播 
"""
# 2.5.1 广播的计算
a1 = np.array([0, 1, 2])
b1 = np.array([5, 5, 5])
print(a1+b1)
print(a1+6)        # a矩阵里所有的元素都+6

M = np.ones((3, 3))
print(M+a1)     # a分别加到M中的每一行中（称 a 这个一维数组被扩展或广播了）

a2 = np.arange(3)
b2 = np.arange(3)[:, np.newaxis]     # 输出列向量
print(a2+b2)      # 将a2和b2都扩展成3x3的矩阵，然后再相加

# 广播的规则
m1 = np.ones((2, 3))
m2 = np.arange(3)
print(m1+m2)     # 由于两个矩阵大小不匹配，因此需要将m2这个1x3的矩阵扩展为2x3的矩阵后再相加

a3 = np.arange(3).reshape((3, 1))
b3 = np.arange(3)
print(a3+b3)     # a3与b3都扩展成3x3的矩阵再相加

X = np.random.random((10, 3))
X_mean = X.mean(0)      # 通过mean函数沿着第一个维度聚合，(此处计算每列的均值)
print(X_mean)
X_centered = abs(X - X_mean)    # 通过从x数组的元素中减去这个均值实现归一化
print(X_centered)

# ############
# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 50)[:, np.newaxis]     # [:,np.newaxis]表示的是转置的意思
# z = np.sin(x)**10+np.cos(10+y*x)*np.cos(x)
#
# import matplotlib.pyplot as plt
# plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
#
# plt.show()
# ############


"""
2.6.2   和通用函数类似的比较操作
"""
x = np.array([1, 2, 3, 4, 5])
print(x < 3)












