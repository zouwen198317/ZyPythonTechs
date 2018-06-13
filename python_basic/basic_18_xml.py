# -*- coding:UTF-8 -*-

'''
    XML解析
'''

# -----------------     sax解析     -----------------

import xml.sax


class MovieHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.type = ""
        self.format = ""
        self.year = ""
        self.rating = ""
        self.stars = ""
        self.description = ""

    # 元素开始调用
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "movie":
            print("*****Movie*****")
            title = attributes["title"]
            print("Title:", title)

    # 元素结束调用
    def endElement(self, tag):
        if self.CurrentData == "type":
            print("Type:", self.type)
        elif self.CurrentData == "format":
            print("Format:", self.format)
        elif self.CurrentData == "year":
            print("Year:", self.year)
        elif self.CurrentData == "rating":
            print("Rating:", self.rating)
        elif self.CurrentData == "stars":
            print("Stars:", self.stars)
        elif self.CurrentData == "description":
            print("Description:", self.description)
        self.CurrentData = ""

    # 读取字符时调用
    def characters(self, content):
        if self.CurrentData == "type":
            self.type = content
        elif self.CurrentData == "format":
            self.format = content
        elif self.CurrentData == "year":
            self.year = content
        elif self.CurrentData == "rating":
            self.rating = content
        elif self.CurrentData == "stars":
            self.stars = content
        elif self.CurrentData == "description":
            self.description = content


def parseBySax():
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # 关闭命名空间
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler(Handler)
    parser.parse("sample_data.xml")


# if (__name__ == "__main__"):
#     parseBySax()
# -----------------     sax解析     -----------------


# -----------------     dom解析     -----------------
from xml.dom.minidom import parse
import xml.dom.minidom

# 使用minidom解析器打开XML文档
dom_tree = xml.dom.minidom.parse("sample_data.xml")
collection = dom_tree.documentElement
if collection.hasAttribute("shelf"):
    print("Root element : %s" % collection.getAttribute("shelf"))

# 在集合中获取所有电影
movies = collection.getElementsByTagName("movie")

# 打印每部电影的详细信息
for movie in movies:
    print("****** Movie ***** ")

    if movie.hasAttribute("title"):
        print("Title: %s" % movie.getAttribute("title"))

    type = movie.getElementsByTagName("type")[0]
    print("Type: %s" % type.childNodes[0].data)
    format = movie.getElementsByTagName("format")[0]
    print("format: %s" % format.childNodes[0].data)
    rating = movie.getElementsByTagName("rating")[0]
    print("rating: %s" % rating.childNodes[0].data)
    description = movie.getElementsByTagName("description")[0]
    print("description: %s" % description.childNodes[0].data)
# -----------------     dom解析     -----------------
