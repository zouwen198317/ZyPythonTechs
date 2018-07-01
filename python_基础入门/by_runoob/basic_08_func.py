# -*- coding:UTF-8 -*-
import log_utils as log

'''
python 函数
'''

"""
语法
def functionname( parameters ):
   "函数_文档字符串"
   function_suite
   return [expression]
默认情况下，参数值和参数名称是按函数声明中定义的顺序匹配起来的。
语法
def functionname( parameters ):
   "函数_文档字符串"
   function_suite
   return [expression]
默认情况下，参数值和参数名称是按函数声明中定义的顺序匹配起来的。
"""


# 定义函数
def printme(str):
    print(str)
    return


printme("我要调用用户自定义函数!")
printme("再次调用同一函数")

# python 传不可变对象实例

log.loge("传不可变对象实例")


def changeInt(a):
    a = 10


b = 2
print(b)
changeInt(b)
print(b)

"""
实例中有 int 对象 2，指向它的变量是 b，在传递给 ChangeInt 函数时，按传值的方式复制了变量 b，a 和 b 都指向了同一个 Int 对象，
在 a=10 时，则新生成一个 int 值对象 10，并让 a 指向它。
"""

log.loge("传不可变对象实例")

# 传可变对象实例

log.loge("传可变对象实例")

"""
    修改列表的值
"""


def changeme(mylist):
    print("arg 原始数据", mylist)
    mylist.append([1, 2, 3, 4])
    print("arg 修改之后的数据", mylist)
    return


myNames = [10, 20, 30]

print("myNames原始数据", myNames)
changeme(myNames)
print("myNames修改之后的数据", myNames)
# 实例中传入函数的和在末尾添加新内容的对象用的是同一个引用
log.loge("传可变对象实例")

log.loge("必备参数")


# 必备参数
# 必备参数须以正确的顺序传入函数。调用时的数量必须和声明时的一样。
def printstr(str):
    print(str)
    return


printstr("1")
log.loge("必备参数")

# 关键字参数
log.loge("关键字参数")
printstr(str="my func is print sth")

print()


# 下例能将关键字参数顺序不重要展示得更清楚
def print_info(name, age):
    print("Name:", name)
    print("Age:", age)
    return


print_info(age=50, name="miki")
log.loge("关键字参数")

log.loge("缺省参数")


# 缺省参数
# 调用函数时，缺省参数的值如果没有传入，则被认为是默认值

def print_info2(name, age=35):
    print("Name:", name)
    print("Age:", age)
    return


print_info2(age=55, name="lihai")
print_info2(name="macao")

log.loge("缺省参数")

log.loge("可变参数")


# 不定长参数(可变参数)
# 你可能需要一个函数能处理比当初声明时更多的参数。这些参数叫做不定长参数

def print_hinfo(arg1, *vartuple):
    print("输出")
    print(arg1)
    for var in vartuple:
        print(var)
    return


print_hinfo(10)
print_hinfo(10, 50, 100, 120)
log.loge("可变参数")

log.loge("匿名函数")
# 匿名函数
# lambda函数的语法只包含一个语句

sum = lambda arg1, arg2: arg1 + arg2

print("相加后的值:", sum(10, 20))
print("相加后的值:", sum(20, 150))
log.loge("匿名函数")

# return语句
log.loge("return语句")


def sum2(arg1, arg2):
    total = arg1 + arg2
    print("函数内: ", total)
    return total


print("计算结果：", sum2(10, 15))

log.loge("return语句")

# 变量作用域
log.loge("变量作用域")
total = 0


def sum3(arg1, arg2):
    total = arg1 + arg2
    print("函数内的局部变量total: ", total)
    return total


print("计算结果: ", sum3(5, 12))
print("全局变量total: ", total)

log.loge("变量作用域")

# 笔记 -> 全局变量想作用于函数内，需加 global
globvar = 0


def set_globvar_to_one():
    # 使用global声明全局变量
    global globvar
    globvar = 1


def print_globvar():
    print(globvar)


set_globvar_to_one()
print(globvar)
print_globvar()

"""
1、global---将变量定义为全局变量。可以通过定义为全局变量，实现在函数内部改变变量值。
2、一个global语句可以同时定义多个变量，如 global x, y, z。
"""

# 列表反转函数
log.loge("列表反转函数")


def reverse(ListInput):
    RevList = []
    for i in range(len(ListInput)):
        RevList.append(ListInput.pop())
    return RevList


l = [1, 2, 3, 4, 5]
print(l)

print(reverse(l))
log.loge("列表反转函数")



