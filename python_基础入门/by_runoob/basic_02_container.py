# -*- coding:UTF-8 -*-

# 容器

# 容器是用来记录数据的，Python中的常用容器有列表、集合、元组、字典。

# 列表（list）是一种有序的容器，可以对元素进行增、删、改操作，例如：

'''
List（列表） 是 Python 中使用最频繁的数据类型。
列表可以完成大多数集合类的数据结构实现。它支持字符，数字，字符串甚至可以包含列表（即嵌套）。
列表用 [ ] 标识
'''

print("------ 列表 ------\n")
# 建立一个列表
names = ["Lily", "Jack", "Lucy", "LiYang"]
print(names)

# 查看元素个数
num = len(names)
print(num)

# 增加元素
names.append("Xiaohu")
print(names)
num = len(names)
print(num)

# 减少元素
names.remove("Jack")
print(names)
num = len(names)
print(num)

# 元素反转
names.reverse()
print(names)

'''
列表中值的切割也可以用到变量 [头下标:尾下标] ，就可以截取相应的列表，从左到右索引默认 0 开始，从右到左索引默认 -1 开始，下标
可以为空表示取到头或尾。
加号 + 是列表连接运算符，星号 * 是重复操作
'''

list = ['runoob', 786, 2.23, 'john', 70.2]
tinylist = [123, 'jonh']
# 输出完整列表
print(list)
# 输出列表的第一个元素
print(list[0])
# 输出第二个至第三个元素
print(list[1:3])
# 输出从第三个开始至列表末尾的所有元素
print(list[2:])
# 输出列表两次
print(tinylist * 2)
# 打印组合的列表
print(list + tinylist)
print("------ 列表 ------\n")

'''
    集合set
    
    集合（set）是一种无序的而且元素不重复的容器。对上面的列表用set()做一个转化，会发现，之前重复的Lily没有了。
'''

print("------ 集合 ------\n")
names.append("Lucy")
print(names)

# set中的元素是唯一
name_set = set(names)
print(name_set)
print("------ 集合 ------\n")

'''
    元组
    元组（tuple）和list非常类似，但是tuple一旦初始化就不能修改。因为tuple不可变，所以代码更安全。如果可能，能用tuple代替list就尽量用tuple。
    
    元组用"()"标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表。
'''
print("------ 元组 ------\n")
t = (1, 2)
print(t)
print('\n')

tuple = ('runoob', 786, 2.23, 'jonh', 70.2)
tintuple = (123, 'jonh')

# 输出完整元组
print(tuple)
# 输出元组的第一个元素
print(tuple[0])
# 输出第二个至第三个元素
print(tuple[1:3])
# 输出从第三个开始至元组末尾的所有元素
print(tuple[2:])
# 输出元组两次
print(tintuple * 2)
# 打印组合的元组
print(tuple + tintuple)
print("------ 元组 ------\n")

'''
    字典（dict）在其他语言中也称为map，使用键-值（key-value）存储，具有极快的查找速度。
    
    字典是另一种可变容器模型，且可存储任意类型对象。
字典的每个键值 key=>value 对用冒号 : 分割，每个键值对之间用逗号 , 分割，整个字典包括在花括号 {} 中 ,格式如下所示：
d = {key1 : value1, key2 : value2 }
'''

print("------ 字典 ------\n")
d = {'Jack': 93, "Lily": 85, "Lucy": 81}
print(d)

# 查询某个数据
print(d['Lily'])

# 添加元素
d['Luyang'] = 99
print(d)

# 修改元素
d['Jack'] = 99
print(d)

# 移出元素:将元素出栈
d.pop('Lily')
print(d)

'''
删除字典元素
能删单一的元素也能清空字典，清空只需一项操作。
显示删除一个字典用del命令，如下实例：
'''
# 删除键是'Name'的条目
del d['Jack']
print(d)

# 清空词典所有条目
d.clear()
print(d)

# 删除词典
del d

"""
字典(dictionary)是除列表以外python之中最灵活的内置数据结构类型。列表是有序的对象集合，字典是无序的对象集合。
两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。
字典用"{ }"标识。字典由索引(key)和它对应的值value组成。
"""

dict = {}
dict['one'] = "this is one"
dict[2] = "this is two"
tinydict = {'name': 'jonh', 'code': 5537, 'dept': 'sales'}

print(dict)
# 输出键为'one' 的值
print(dict['one'])
# 输出键为 2 的值
print(dict[2])
# 输出完整的字典
print(tinydict)
# 输出所有键
print(tinydict.keys())
# 输出所有值
print(tinydict.values())
print("------ 字典 ------\n")