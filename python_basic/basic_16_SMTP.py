# -*- coding:UTF-8 -*-

''''
  win10环境不支持smtp无法难暂不做测试

'''
'''
实例
以下执行实例需要你本机已安装了支持 SMTP 的服务，如：sendmail。
'''
import smtplib
from email.mime.text import MIMEText
from email.header import Header

sender = 'xfgcjy@163.com'
# 接收邮件，可设置为你的QQ邮箱或者其他邮箱
receivers = ['xfgczzg@163.com', 'xfgczzg@qq.com']

# 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
message = MIMEText('Python 邮件发送测试...', 'plain', 'utf-8')
# 发送者
message['From'] = Header("菜鸟教程", 'utf-8')
# 接收者
message['To'] = Header("测试", 'utf-8')

subject = 'Python SMTP 邮件测试'
message['Subject'] = Header(subject, 'utf-8')

try:
    smtpObj = smtplib.SMTP('localhost')
    smtpObj.sendmail(sender, receivers, message.as_string())
    print('邮件发送成功')
except smtplib.SMTPException:
    print("Error： 无法发送邮件")
