# -*- coding: utf-8 -*-
# defaultencoding = 'utf-8'

"""
爬取美团的商家评论页面评论内容
http://www.meituan.com/meishi/api/poi/getMerchantComment?uuid=ee9c6e40420d6624b95b.1528784004.1.0.1&platform=1&partner=126&originUrl=http%3A%2F%2Fwww.meituan.com%2Fmeishi%2F159397446%2F&riskLevel=1&optimusCode=1&id=159397446&userId=&offset=0&pageSize=10&sortType=1
"""

import json
import requests
import random
import csv
import logging; logging.basicConfig(level=logging.INFO)
from icecream import ic



def getJson(n):
  '''
  获取json数据
  '''
  headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
  }

  url = 'http://www.meituan.com/meishi/api/poi/getMerchantComment?uuid=095e5a{}50d417a92d9.1526123380.1.0.0&platform=1&partner=126&originUrl=http%3A%2F%2Fwww.meituan.com%2Fmeishi%2F4955158%2F&riskLevel=1&optimusCode=1&id=159397446&userId=&offset={}&pageSize=10&sortType=1'.format(random.randint(100,999),n*30)
  response = requests.get(url, timeout=10, headers=headers)
  res = json.loads(response.text)
  return res


def getPageNum(json_data):
  '''
  计算页数
  '''
  total = json_data['data']['total']
  num = int(total / 10 + 1)
  return num


def getResult(json_data):
  '''
  获取数据
  '''
  all_comment = json_data['data']['comments']
  res = []
  for i in range(0, 10):
    prs = all_comment[i]
    user_id = prs['userId']
    user_name = prs['userName']
    comment = prs['comment']
    star = prs['star']
    res.append([user_id, user_name, comment, star])
  return res


def mainProcess(page_num):
  '''
  主逻辑函数
  '''
  res = []
  headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
  }
  for i in range(1, page_num):
    logging.info('爬取第{}页.....'.format(i))
    url = 'http://www.meituan.com/meishi/api/poi/getMerchantComment?uuid=095e5a{}d417a92d9.1526123380.1.0.0&platform=1&partner=126&originUrl=http%3A%2F%2Fwww.meituan.com%2Fmeishi%2F4955158%2F&riskLevel=1&optimusCode=1&id=159397446&userId=&offset={}&pageSize=10&sortType=1'.format(random.randint(10000,99999),(i-1)*10)
    try:
      response = requests.get(url, timeout=10, headers=headers)
      res_get = json.loads(response.text)
    except:
      continue

    res_tmp = getResult(res_get)
    res.append(res_tmp)
  logging.info('Finish')
  return res


def fwrite(filename, res, label):
  '''
  写入csv文件
  '''
  with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(label)
    for data in res:
      for row in data:
        try:
          writer.writerow(row)
        except UnicodeEncodeError as e1:
          writer.writerow(['UnicodeEncodeError'])


if __name__ == '__main__':
  get_json = getJson(0)
  page_num = getPageNum(get_json)
  ic(page_num)
  res = mainProcess(page_num)
  label = ['userId', 'userName', 'comment', 'star']
  file_name = 'comment2.csv'
  fwrite(file_name, res, label)





