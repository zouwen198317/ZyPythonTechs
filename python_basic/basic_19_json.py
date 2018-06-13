# -*- coding:UTF-8 -*-

'''
    json解析与生成
'''

# -----------------     json.dumps 与 json.loads 实例     -----------------
import log_utils as log

log.loge("对数据进行编码")
'''
json.dumps(): 对数据进行编码。
json.loads(): 对数据进行解码。
'''
import json

# python字典类型转换为Json对象
data = {
    'no': 1,
    'name': 'test01',
    'url': 'http://www.baidu.com'
}

json_str = json.dumps(data)
print("Python 原始数据: ", repr(data))
print("Json对象: ", json_str)

log.loge("对数据进行编码")

log.loge("对数据进行解码")
# 将Json对象转换为Python字典
data_json_dict = json.loads(json_str)
print("dict['name']: ", data_json_dict['name'])
print("dict['url']: ", data_json_dict['url'])
log.loge("对数据进行解码")
# -----------------     json.dumps 与 json.loads 实例     -----------------


# -----------------     json.dump 与 json.load 实例     -----------------
# 如果你要处理的是文件而不是字符串，你可以使用 json.dump() 和 json.load() 来编码和解码JSON数据。例如：

# 写入json数据
with open('data.json', 'w') as f:
    json.dump(data, f)

# 读取数据
with open('data.json', 'r') as f:
    data = json.load(f)
    print(data)

# -----------------     json.dump 与 json.load 实例     -----------------
