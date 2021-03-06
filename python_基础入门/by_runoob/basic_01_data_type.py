# -*- coding: UTF-8 -*-

# python数据类型

# 整数
i = 10
print("整数", i)
type1 = type(i)
print("整数", type1)

# 浮点数
f = 1.123
print("浮点数", f)
type2 = type(f)
print("浮点数", type2)

# 布尔类型
b = (10 > 100)
print("布尔类型", b)
type3 = type(b)
print("布尔类型", type3)

# 字符串

s = "这是一个传说"
print("字符串", s)
type4 = type(s)
print("字符串", type4)

str = 'Hello World!'
# 输出完整字符串
print(str)
# 输出字符串中的第一个字符
print(str[0])
'''
python的字串列表有2种取值顺序:
从左到右索引默认0开始的，最大范围是字符串长度少1
从右到左索引默认-1开始的，最大范围是字符串开头

如果你要实现从字符串中获取一段子字符串的话，可以使用变量 [头下标:尾下标]，就可以截取相应的字符串，其中下标是从 0 开始算起，
可以是正数或负数，下标可以为空表示取到头或尾。
'''
# 输出字符串中第三个至第五个之间的字符串
print(str[2:5])
'''
当使用以冒号分隔的字符串，python返回一个新的对象，结果包含了以这对偏移标识的连续的内容，左边的开始是包含了下边界。
上面的结果包含了s[1]的值l，而取到的最大范围不包括上边界，就是s[5]的值p。
'''
# 输出从第三个字符开始的字符串
print(str[2:])
# 输出字符串两次
'''
加号（+）是字符串连接运算符，星号（*）是重复操作
'''
print(str * 2)
# 输出连接的字符串
print(str + "TEST")

"""
区别就是:
 type()不会认为子类是一种父类类型。
 isinstance()会认为子类是一种父类类型。
"""
a = 115
print(type(1))
print(isinstance(1, int))


class A:
    pass


class B:
    pass


print(isinstance(A(), A))
print(type(A()) == A)

print(isinstance(B(), A))
print(type(B()) == A)

"""
变量赋值简单粗暴不需要声明类型, 灵活多变,非常好用。
数字数据类是不可改变的数据类型，改变数字数据类型会分配一个新的对象。
字符串的操作有基本的功能不需要再自己进行拼接遍历的操作。
列表用 "[ ]" 标识类似 C 语言中的数组。
元组用 "( )" 标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表。
字典用 "{ }" 标识。字典由索引 key 和它对应的值 value 组成。

字符串表示方式：str="hello,world"
列表表示方式：list=['hello',2,3,4,'world']
元组：tuple=('hello',2,3,4,'world')
截取方式相同：名称[头下标:尾下标]
下标是从0开始算起，可以是正数或者负数，下标为空则表示取到头或者尾
开始截取时，包含了下边界，而截取到最大范围不包括上边界。
元组不能二次赋值，列表可以

Python变量类型：
（1）Numbers
（2）String
（3）List  []
（4）Tuple（元祖）(),相当于只读列表，不可以二次赋值
（5）dictionary（字典）{}，key值对
"""
