import common_header

print("Hello,Python!")

# 行和缩进
if True:
    print("True")
else:
    print("False")

if True:
    print("Answer")
    print("Ture")

else:
    print("Answer")
    # 没有严格缩进，在执行时保持
    print("False")

# 多行语句
item_one = 1
item_two = 2
item_three = 3

total = item_one + \
        item_two + \
        item_three
print(total)

print(['Monday', "Tuesday", 'Wedneday', 'Thursday', 'Friday'])

# 引号
word = 'word'
sentence = "这是一个句子."
paragraph = """这是一个段落。
包含了多个语句的段落"""

print(word)
print(sentence)
print(paragraph)

# 注释
# 第一个注释
print("Hello,Python!")

# 第二个注释
name = "Madisetti"  # 这是一个注释

'''
    多行注释，单引号结构
'''
"""
    多行注释，双引号结构
"""

# 空行
str = input("\n\nPress the enter key to exit.")

import sys

x = 'foo'
sys.stdout.write(x + "\n")
