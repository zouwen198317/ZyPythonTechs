# -*- coding:UTF-8 -*-
import log_utils as log

import time

ticks = time.time()

print('当前时间戳', ticks)

"""
获取当前时间
"""
localtime = time.localtime(time.time())
print("本地时间为:", localtime)

"""
获取格式化的时间
"""
localtime = time.asctime(localtime)
print("本地时间为:", localtime)

"""
格式化日期
"""

# 我们可以使用 time 模块的 strftime 方法来格式化日期

# 格式化成2016-03-20 11:45:39形式
time_strftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("格式化后的本地时间为:", time_strftime)
time_strftime = time.strftime("%y-%m-%d %H:%M:%S", time.localtime())
print("格式化后的本地时间为:", time_strftime)

# 格式化成Sat Mar 28 22:24:24 2016形式
time_strftime = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
print("格式化后的本地时间为:", time_strftime)

#  将格式字符串转换为时间戳
a = "Fri Jun 08 15:30:55 2018"
print("将a格式字符串转换为时间戳:", time.mktime(time.strptime(a, "%a %b %d %H:%M:%S %Y")))

"""
获取某月日历
Calendar模块有很广泛的方法用来处理年历和月历
"""

log.loge("获取某月日历")
import calendar

cal = calendar.month(2018, 7)
print("以下是2018年7月份的日历:")
print(cal)
log.loge("获取某月日历")

# 使用datetime模块来获取当前的日期和时间
log.loge("datetime")
import datetime

i = datetime.datetime.now()
print("当前的日期和时间是 %s" % i)
print("ISO格式的日期和时间是 %s" % i.isoformat())
print("当前的年份是 %s" % i.year)
print("当前的月份是 %s" % i.month)
print("当前的日期是  %s" % i.day)
print("dd/mm/yyyy 格式是  %s/%s/%s" % (i.day, i.month, i.year))
print("当前小时是 %s" % i.hour)
print("当前分钟是 %s" % i.minute)
print("当前秒是  %s" % i.second)
log.loge("datetime")
