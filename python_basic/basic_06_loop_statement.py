# -*- coding:UTF-8 -*-
import log_utils as log

# 循环语句

# while判断条件:
#  执行语句
log.loge("while判断条件")
count = 0
while (count < 9):
    print("the count is :", count)
    count += 1;
print("good bye!")
print()

"""
while 语句时还有另外两个重要的命令 continue，break 来跳过循环，continue 用于跳过该次循环，break 则是用于退出循环，
此外"判断条件"还可以是个常值，表示循环必定成立，具体用法如下：
"""

# continue 和 break 用法
i = 1
while i < 10:
    i += 1;
    if i % 2 > 0:  # 非双数时跳过输出
        continue
    print(i)
print()

i = 1
while 1:
    print(i)
    i += 1
    if i > 10:
        break

# 无限循环
# 如果条件判断语句永远为 true，循环将会无限的执行下去

var = 1
# while var == 1:
#     var = input("Enter a number :")
#     print("your entered : ", var)

print("good bye")

"""
循环使用 else 语句
在 python 中，while … else 在循环条件为 false 时执行 else 语句块：
"""
count = 0
while count < 5:
    print(count, " is less then 5")
    count += 1
else:
    print(count, "is not less then 5")
log.loge(" while判断条件")

log.loge(" for 循环语句")
# Python for 循环语句
for letter in 'python':
    print('当前字母:', letter)

fruits = ['banna', 'apple', 'mango']
for fruit in fruits:
    print('当前水果:', fruit)

# 通过序列索引迭代
# 另外一种执行循环的遍历方式是通过索引
"""
使用了内置函数 len() 和 range(),函数 len() 返回列表的长度，即元素的个数。 range返回一个序列的数。
"""
for index in range(len(fruits)):
    print('当前水果:', fruits[index])

"""
循环使用 else 语句
在 python 中，for … else 表示这样的意思，for 中的语句和普通的没有区别，else 中的语句会在循环正常执行完（即 for 不是通过 
break 跳出而中断的）的情况下执行，while … else 也是一样。
"""

"""
%s 字符串
%d 整型
%f 浮点型
"""
for num in range(10, 20):
    for i in range(2, num):
        if num % 2 == 0:
            j = num / i
            print('%d 等于 %d * %d' % (num, i, j))
            break
    else:
        print(num, '是一个质数')

log.loge(" for 循环语句")

log.loge(" Python 循环嵌套")
"""
Python 语言允许在一个循环体里面嵌入另一个循环。
Python for 循环嵌套语法：
for iterating_var in sequence:
   for iterating_var in sequence:
      statements(s)
   statements(s)
Python while 循环嵌套语法：
while expression:
   while expression:
      statement(s)
   statement(s)
你可以在循环体内嵌入其他的循环体，如在while循环中可以嵌入for循环， 反之，你可以在for循环中嵌入while循环。
"""

# 实例： 以下实例使用了嵌套循环输出2~100之间的素数：
i = 2
while (i < 100):
    j = 2
    while (j <= (i / j)):
        if not (i % j): break
        j += 1;
    if (j > i / j): print(i, " 是素数")
    i += 1;
log.loge(" Python 循环嵌套")

log.loge(" Python break 语句")
"""
Python break语句，就像在C语言中，打破了最小封闭for或while循环。
break语句用来终止循环语句，即循环条件没有False条件或者序列还没被完全递归完，也会停止执行循环语句。
break语句用在while和for循环中。
如果您使用嵌套循环，break语句将停止执行最深层的循环，并开始执行下一行代码
"""
for letter in 'python':
    if letter == 'h':
        break
    print('当前字母:', letter)

var = 10
while var > 0:
    print('当前变量值:', var)
    var -= 1;
    if var == 5:
        break
log.loge(" Python break 语句")

log.loge(" Python continue 语句")
"""
Python continue 语句跳出本次循环，而break跳出整个循环。
continue 语句用来告诉Python跳过当前循环的剩余语句，然后继续进行下一轮循环。
continue语句用在while和for循环中。
"""

for letter in 'python':
    if letter == 'h':
        continue
    print('当前字母:', letter)

var = 10
while var > 0:
    var -= 1;
    if var == 5:
        continue
    print('当前变量值:', var)
print()

# continue 语句是一个删除的效果，他的存在是为了删除满足循环条件下的某些不需要的成分:
var = 10
while var > 0:
    var -= 1;
    if var == 5 or var == 8:
        continue
    print('当前变量值:', var)

# 我们想只打印0-10之间的奇数，可以用continue语句跳过某些循环：
n = 0
while n < 10:
    n += 1;
    if n % 2 == 0:
        continue
    print(n, " 为基数")
log.loge(" Python continue 语句")

log.loge(" Python pass 语句")
"""
Python pass 语句
Python pass是空语句，是为了保持程序结构的完整性。
pass 不做任何事情，一般用做占位语句。
"""
for letter in 'python':
    if letter == 'h':
        pass
        print('这是pass块')
    print('当前字母:', letter)

log.loge(" Python pass 语句")

# 猜大小的游戏
log.loge("猜大小的游戏")
import random

s = int(random.uniform(1, 10))
# print(s)
# m = int(input('输入整数:'))

# while m != s:
#     if m > s:
#         print('大了')
#         m = int(input('输入整数:'))
#     if m < s:
#         print('小了')
#         m = int(input('输入整数:'))
#     if m == s:
#         print("OK")
#         break
log.loge("猜大小的游戏")

log.loge("摇筛子游戏")
import sys
import time

result = []
# while True:
#     result.append(int(random.uniform(1, 7)))
#     result.append(int(random.uniform(1, 7)))
#     result.append(int(random.uniform(1, 7)))
#     print(result)
#     count = 0
#     index = 2
#     pointStr = ""
#     while index >= 0:
#         currPoint = result[index]
#         count += currPoint
#         index -= 1
#         pointStr += " "
#         pointStr += str(currPoint)
#     if count <= 11:
#         sys.stdout.write(pointStr + " -> " + "小" + "\n")
#         time.sleep(1)  # 睡眠一秒
#     else:
#         sys.stdout.write(pointStr + " -> " + "大" + "\n")
#         time.sleep(1)  # 睡眠一秒
#     result = []

log.loge("摇筛子游戏")

# 输出 2 到 100 简的质数
prime = []
for num in range(2, 100):  # 迭代 2 到 100 之间的数字
    for i in range(2, num):  # 根据因子迭代
        if num % i == 0:  # 确定第一个因子
            break  # 跳出当前循环
    else:  # 循环的 else 部分
        prime.append(num)
print(prime)

# 冒泡排序# 定义列表 list
arays = [1, 8, 2, 6, 3, 9, 4]
for i in range(len(arays)):
    for j in range(i + 1):
        if arays[i] < arays[j]:
            # 实现连个变量的互换
            arays[i], arays[j] = arays[j], arays[i]
print(arays)
