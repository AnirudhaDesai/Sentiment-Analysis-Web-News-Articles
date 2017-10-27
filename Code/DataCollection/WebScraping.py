# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 22:11:04 2017

@author: Anirudha Desai
"""

import scrapy
import pandas as pd
from bs4 import BeautifulSoup

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    contentArray = []         
	
    def start_requests(self):
        filePath = 'urls_trial.txt'
        urls = self.readFile(filePath)
        for url in urls:
            request = scrapy.Request(url=url, callback=self.parse)
            yield request
	
    def parse(self, response):
        df = pd.DataFrame()
        text = self.cleanPage(response)
        self.contentArray.append(text)
        df = df.append(self.contentArray)
        df.to_csv('FinalData.csv')       
    	
    def readFile(self,filePath):
        urls = []
        textFile = open(filePath, 'r')
        for line in textFile.readlines():
            urls.append(line)
        return urls
        
    def cleanPage(self,response):
        page = response.body
        soup = BeautifulSoup(page,'html.parser')
        page_text = soup.find_all('p')
        page_text = str(page_text)
        return page_text
        