# -*- coding:UTF-8 -*-
import log_utils as log

# Python算术运算符
log.loge("Python算术运算符")

a = 21
b = 10
c = 0

c = a + b
print("1 -> c的值为:", c)

c = a - b
print("2 -> c的值为:", c)
c = a * b
print("3 -> c的值为:", c)
c = a / b
print("4 -> c的值为:", c)

# 取余数
c = a % b
# 1
print("5 -> c的值为:", c)

# a的n次幂
a = 2
b = 3
c = a ** b
# 8
print("6 -> c的值为:", c)

# 取整除 (商的整数部分)
a = 9
b = 4
c = a // b
# 2
print("7 -> c的值为:", c)

log.loge("Python算术运算符")

# Python比较运算符
log.loge("Python比较运算符")
a = 21
b = 10
c = 0

if (a == b):
    print("1 - a 等于 b")
else:
    print("1 - a 不等于 b")

if (a != b):
    print("2 - a 不等于 b")
else:
    print("2 - a 等于 b")

if (a != b):
    print("3 - a 不等于 b")
else:
    print(
        "3 - a 等于 b")

if (a < b):
    print("4 - a 小于 b")
else:
    print("4 - a 大于等于 b")

if (a > b):
    print("5 - a 大于 b")
else:
    print("5 - a 小于等于 b")

# 修改变量 a 和 b 的值
a = 5
b = 20
if (a <= b):
    print("6 - a 小于等于 b")
else:
    print("6 - a 大于  b")

if (b >= a):
    print("7 - b 大于等于 a")
else:
    print("7 - b 小于 a")

log.loge("Python比较运算符")

log.loge("Python赋值运算符")
a = 21
b = 10
c = 0

c = a + b
print("1 - c 的值为：", c)

c += a
print("2 - c 的值为：", c)

c *= a
print("3 - c 的值为：", c)

c /= a
print("4 - c 的值为：", c)

c = 2
print("5 - a 的值为：a = ", a)
print("5 - c 的值为：c = ", c)
c %= a
# a % 比它小的数，结果还是a
print("5 - c 的值为：", c)
print("5 - c2 的值为：", 2 % 21)

# a的n次幂
c **= a
print("6 - c 的值为：", c)
# print("6 - c2 的值为：", 2 ** 3)

c //= a
print("7 - c 的值为：", c)

log.loge("Python赋值运算符")

log.loge("Python位运算符")
a = 60  # 60 = 0011 1100
b = 13  # 13 = 0000 1101
c = 0

# 两个都要为1才为1
c = a & b;  # 12 = 0000 1100
print("1 - c 的值为：", c)

# 有一个为1就为1
c = a | b;  # 61 = 0011 1101
print("2 - c 的值为：", c)

c = a ^ b;  # 49 = 0011 0001
print("3 - c 的值为：", c)

# -a-1=-61 (这里还需要再计算一下)

''' 
~60 
        0011 1100
(取反)  1100 0011
-1      0000 0001
----------------
        1100 0010
(取反)  0011 1101
----------------
            61    
'''
c = ~a;  # -61 = 1100 0011
print("4 - c 的值为：", c)

c = a << 2;  # 240 = 1111 0000
print("5 - c 的值为：", c)

c = a >> 2;  # 15 = 0000 1111
print("6 - c 的值为：", c)

log.loge("Python位运算符")

log.loge("Python逻辑运算符")
a = 10
b = 20
print(a and b)
print(a & b)
if (a and b):
    print("1 - 变量 a 和 b 都为 true")
else:
    print("1 - 变量 a 和 b 有一个不为 true")

print(a or b)

if (a or b):
    print("2 - 变量 a 和 b 都为 true，或其中一个变量为 true")
else:
    print("2 - 变量 a 和 b 都不为 true")

# 修改变量 a 的值
a = 0
print(a or b)
if (a and b):
    print("3 - 变量 a 和 b 都为 true")
else:
    print(
        "3 - 变量 a 和 b 有一个不为 true")

if (a or b):
    print("4 - 变量 a 和 b 都为 true，或其中一个变量为 true")
else:
    print("4 - 变量 a 和 b 都不为 true")

if not (a and b):
    print("5 - 变量 a 和 b 都为 false，或其中一个变量为 false")
else:
    print("5 - 变量 a 和 b 都为 true")

log.loge("Python逻辑运算符")

log.loge("Python成员运算符")
a = 10
b = 20
list = [1, 2, 3, 4, 5];

if (a in list):
    print("1 - 变量 a 在给定的列表中 list 中")
else:
    print("1 - 变量 a 不在给定的列表中 list 中")

if (b not in list):
    print("2 - 变量 b 不在给定的列表中 list 中")
else:
    print("2 - 变量 b 在给定的列表中 list 中")

# 修改变量 a 的值
a = 2
if (a in list):
    print("3 - 变量 a 在给定的列表中 list 中")
else:
    print("3 - 变量 a 不在给定的列表中 list 中")

log.loge("Python成员运算符")

log.loge("Python身份运算符")
a = 20
b = 20

if (a is b):
    print("1 - a 和 b 有相同的标识")
else:
    print("1 - a 和 b 没有相同的标识")

if (a is not b):
    print("2 - a 和 b 没有相同的标识")
else:
    print(
        "2 - a 和 b 有相同的标识")

# 修改变量 b 的值
b = 30
if (a is b):
    print("3 - a 和 b 有相同的标识")
else:
    print("3 - a 和 b 没有相同的标识")

if (a is not b):
    print("4 - a 和 b 没有相同的标识")
else:
    print("4 - a 和 b 有相同的标识")

log.loge("Python身份运算符")

"""
这里 is 和 == 类似编译原理中传值与传地址。又或者说是 is 只是传递的指针，判断是否指向同一个地址块，这样 is 两边的参数指向内存中同个地址块，毕竟我家电视跟你电视不是同一个东西。而 == 则是仅仅判断值相同


"""
