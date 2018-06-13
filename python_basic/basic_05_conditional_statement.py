# -*- coding:UTF-8 -*-
import os

# 条件语句

"""
if 判断条件：
    执行语句……
else：
    执行语句……
"""
flag = False

name = 'luren'

# 例1：if 基本用法
if name == 'python':
    flag = True
    print(
        'welcome boss')
else:
    print(name)

"""
if 语句的判断条件可以用>（大于）、<(小于)、==（等于）、>=（大于等于）、<=（小于等于）来表示其关系。
当判断条件为多个值时，可以使用以下形式：
if 判断条件1:
    执行语句1……
elif 判断条件2:
    执行语句2……
elif 判断条件3:
    执行语句3……
else:
    执行语句4……
"""

# 例2：elif用法
num = 5
if num == 3:
    print("boass")
elif num == 2:
    print("user")
elif num == 2:
    print("worker")
elif num < 0:
    print("error")
else:
    print("roadman")

"""
由于 python 并不支持 switch 语句，所以多个条件判断，只能用 elif 来实现，如果判断需要多个条件需同时判断时，可以使用 or （或），
表示两个条件有一个成立时判断条件成功；使用 and （与）时，表示只有两个条件同时成立的情况下，判断条件才成功。
"""

# 例3：if语句多个条件
num = 9
if (num >= 0 and num <= 10):
    print("hello")

num = 10
if (num < 0 or num > 10):
    print("hello")
else:
    print("undefine")

num = 8
if (num >= 0 and num <= 5) or (num >= 10 and num <= 15):
    print("hello")
else:
    print("undefine")

"""
当if有多个条件时可使用括号来区分判断的先后顺序，括号中的判断优先执行，此外 and 和 or 的优先级低于>（大于）、<（小于）等判断符
号，即大于和小于在没有括号的情况下会比与或要优先判断。
"""


# 一个简单的条件循环语句实现汉诺塔问题
def my_print(args):
    print(args)


my_print('汉诺塔问题')


def move(n, a, b, c):
    my_print((a, '-->', c)) if n == 1 else (move((n - 1), a, c, b)) or move(n - 1, b, a, c)


move(3, 'a', 'b', 'c')
my_print('汉诺塔问题')

import os


def zero():
    return "zero"


def one():
    return "one"


def two():
    return "two"


def num2Str(arg):
    switcher = {
        0: zero, 1: one, 2: two, 3: lambda: "three"
    }
    func = switcher.get(arg, lambda: "northing")
    return func()


print(num2Str(1))
