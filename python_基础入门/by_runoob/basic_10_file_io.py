# -*- coding:UTF-8 -*-

# 打印到屏幕
print("python 是一个非常棒的语言,不是吗?")

# 读取键盘输入
# input函数
# input([prompt]) 函数和 raw_input([prompt]) 函数基本类似，但是 input 可以接收一个Python表达式作为输入，并将运算结果返回。

str = input("请输入：")
# 这里会有出入
print("你输入的内容是: ", str)

# 打开和关闭文件
# open 函数
# 如果文件不存在会创建
fo = open("foo.txt", "w")
print("文件名: ", fo.name)
print("是否关闭: ", fo.closed)
print("访问模式: ", fo.mode)

# close()方法
fo.close()
print("是否关闭: ", fo.closed)

# write()方法
# write()方法不会在字符串的结尾添加换行符('\n')：
fo_01 = open("foo.txt", "w")
fo_01.write("http://www.baidu.com good site\n")

# 关闭打开的文件
fo_01.close()

# read()方法
# 在这里，被传递的参数是要从已打开文件中读取的字节计数。该方法从文件的开头开始读入，如果没有传入count，它会尝试尽可能多地读取更多的
# 内容，很可能是直到文件的末尾。
fo_02 = open("foo.txt", "r+")
# str = fo_02.read(10)
str = fo_02.read()
print("读取的字符串的内容是:", str)
fo_02.close()

# 文件定位

fo_03 = open("foo.txt", "r+")
str = fo_03.read(10)
print("读取的字符串是:", str)

# 查找当前位置
current_pos = fo_03.tell()
print("当前文件位置:", current_pos)

# 把指针再次重新定位到文件头
position = fo_03.seek(0, 0)
str = fo_03.read(10)
print("重新读取到的字符串为:", str)
# 关闭打开的文件
fo_03.close()

# 重命名和删除文件
import os

# 重命名文件
os.rename("foo.txt", "foo1.txt")

# 删除文件
os.remove("foo1.txt")

# mkdir()方法
os.mkdir("test")

# chdir()方法

# 给出当前的目录
# print(os.getcwd())
#
# # 将当前目录改为/home/newdir
# os.chdir("/home/newdir")
# print(os.getcwd())

# rmdir()方法 未生效
os.rmdir("test")