L3 = [True, "3", 3.0, 3]
X = [type(x)for x in L3]
print(X)
"""
2.1.5 从头创建数组
"""
import numpy as np
A = np.array([1, 4, 2, 5, 3])
print(A)

B = np.arange(0, 20, 2)    # 创建一个线性序列数组，从0开始，20结束，步长为2
print(B)

C = np.linspace(0,1,5)    # 创建一个5个元素的数组，这5个数均匀的分配到0~1
print(C)

D = np.random.random((3,3))   # 创建一个3x3的、在0~1均匀分布的随机数组组成的数组
print(D)

# 创建一个3x3的、均值为0、标准差为1
# 正态分布的随机数数组
E = np.random.normal(0, 1, (3, 3))
print(E)

F = np.random.randint(0, 10, (3, 3))   # 创建一个3x3的、[0，10)区间的随机整型数组
print(F)

G = np.eye(3)     # 创建一个3x3的单位矩阵
print(G)

"""
2.2.1 Numpy数组的属性
"""
import numpy as np
np.random.seed(0)             # 设置随机数种子（以确保每次程序执行时都可以生成同样的随机数组）
x1 = np.random.randint(10, size=6)             # 一维数组
x2 = np.random.randint(10, size=(3, 4))        # 二维数组
x3 = np.random.randint(10, size=(3, 4, 5))     # 三维数组
print(x1)
print(x2)
print(x3)
print(x3.dtype)   # 查看x3数组的数据类型
print("itemsize:", x3.itemsize)  # 每个数组元素字节大小
print("nbytes:", x3.nbytes)    # 数组总字节大小

"""
2.2.2 数组索引：获取单个元素
"""
x1 = np.array([1, 2, 3, 4, 5])
print(x1[4])       # 元素索引

x2 = np.array([[3, 5, 2, 4],
               [7, 6, 8, 8],
               [1, 6, 7, 7]])
x2[0, 0] = 6    # 修改第一行第一列的元素值
print(x2)
print(x2[0])

"""
2.2.3 数组切片：获取子数组
      x[start:stop:step]
"""

#  1、一维子数组
x = np.arange(10)    # arange函数用于创建等差数组 （可指定开始值、终值和步长）
a1 = x[:5]        # 获取前5个元素,为[0,1,2,3,4]
a2 = x[5:]        # 索引5之后的元素(包含第五位)，为[5,6,7,8,9]
a3 = x[2:7]       # 中间的子数组(2指索引位置，第一个元素的索引位置是0)，a3的输出：[2，3，4，5，6]
a4 = x[::2]       # 索引整个x数组，步长为2，即隔一个元素
a5 = x[1::2]      # 每隔一个元素，从1开始

in1 = x[::-1]     # 索引所有元素，逆序
a6 = in1[5::-2]
print(a6)


# 2、多维子数组
r = np.array([[12, 5, 2, 4],
              [7, 6, 8, 8],
              [1, 6, 7, 7]])
r1 = r[:2, :3]    # 两行，三列  (切割交叉区域)
print(r1)
r2 = r[:3, ::2]     # 三行（即所有行,此处3可有可无），隔一列
print(r2)

r3 = r[::-1, ::-1]    # 子数组维度逆序
print(r3)


# 3、获取数组的行和列
print(r[:, 0])      # r的第一列
print(r[0, :])      # r的第一行 ，在获取行时，出于简介考虑，可省略空切片，r[0]=r[0,:]

# 4、非副本试图的子数组
r = np.array([[12, 5, 2, 4],
              [7, 6, 8, 8],
              [1, 6, 7, 7]])


r_sub = r[:2, :2]    # 切取2x2的矩阵
r_sub[0, 0] = 99     # 将r_sub子数组的第一行第一列元素值替换成99
print(r_sub)
print(r)          # 修改子数组r_sub的同时，原始数组r也被修改

# 4、创建数组的副本
r_sub_copy = r[:2, :2].copy()       # 通过copy()方法实现保留原数组不改变
r_sub_copy[0, 0] = 42
print(r_sub_copy)
print(r)


"""
2.2.4   数组的变形
"""

b = np.array([1, 2, 3])
b1 = b.reshape((1, 3))      # 通过变形获得行向量(变形为1行3列)
print(b1)
b2 = b[np.newaxis, :]      # 通过 newaxis 获得的行向量
print(b2)
b3 = b[:, np.newaxis]      # 通过 newaxis 获得的列向量
print(b3)


"""
2.2.5   数组的拼接和分裂
"""
# 1、数组的拼接
b = np.array([1, 2, 3])
grid1 = np.array([[9, 8, 7],
                [6, 5, 4]])
b4 = np.vstack([b, grid1])     # 垂直栈数组（垂直拼接）
print(b4)
grid2 = np.array([[99],
                  [99]])
b5 = np.hstack([grid1, grid2])    # 水平栈数组（水平拼接）
print(b5)

# 2、数组的分裂
c = np.array([1, 2, 3, 99, 99, 3, 2, 1])
c1, c2, c3 = np.split(c, [3, 5])     # 索引记录的是分裂点的位置，此处分裂点分别为3和5
print(c1, c2, c3)

c4 = np.arange(16).reshape((4, 4))
upper, lower = np.vsplit(c4, [2])   # 垂直分裂
print(upper)
print(lower)
left, right = np.hsplit(c4, [2])     # 水平分裂
print(left)
print(right)

"""
2.3.3    探索Numpy的通用函数
"""
# 2、绝对值
d = np.array([-2, -1, 0, 1, 2])
print(abs(d))
d1 = np.array([3-4j, 4-3j, 2+0j, 0+1j])
print(abs(d1))

# 3、三角函数
theta = np.linspace(0, np.pi, 3)     # 定义角度数组
print(theta)
print("sin(theta)=", np.sin(theta))        # 计算数组的三角函数
d2 = np.array([-1, 0, 1])
print("arcsin(d2)=", np.arcsin(d2))        # 计算数组的反三角函数

# 4、指数和对数
d3 = np.array([1, 2, 3])
print("e^d3 = ", np.exp(d3))
print("2^d3 = ", np.exp2(d3))
print("3^d3 = ", np.power(3, d3))

d4 = np.array([1, 2, 4, 10])
print("ln(d4) = ", np.log(d4))
print("log2(d4) = ", np.log2(d4))
print("log10(d4) = ", np.log10(d4))












