#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:22:49 2017

@author: gadde
"""
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from nltk import word_tokenize

def main():
    print "Hello World"
    df = pd.read_csv('rawTrainLineDataSet.csv',delimiter=',')
    print df.shape
    df_cleaned = removeAlphabets(df)
    df_cleaned2 = removeOtherStuff(df_cleaned)
    print df_cleaned2['article']
    df_cleaned2.to_csv('cleanedTestSentencesDataSet.csv',sep=',')    

def Stemmer(df):
    porter = PorterStemmer()
    df_stemmed  = DataFrame(columns=['sentimentValue','article'])
    
    for i in range(0,len(df)):
        row = df.loc[i]
        art = row['sentence']
        art_tokens = word_tokenize(art)
        art_tokens_stemmed = [porter.stem(t) for t in art_tokens]
        art_stemmed = " ".join(art_tokens_stemmed)
        df_stemmed.loc[len(df_stemmed)]=[row['sentimentValue'],art_stemmed]
    return df_stemmed
    
def removeAlphabets(df):
    df_new = DataFrame(columns=['sentimentValue','sentence'])
    for i in range(0,len(df)):
        print i
        row = df.loc[i]
        article = row['sentence']
        article_tokens= word_tokenize(article)
        to_remove = ['n',',','u','.','[',']','advertisement']
        article_tokens_cleaned = [item for item in article_tokens if not item in to_remove]
        article_cleaned = " ".join(article_tokens_cleaned)
        df_new.loc[len(df_new)]=[row['sentimentValue'],article_cleaned]
    return df_new

def removeOtherStuff(df):
    print 'in remove other stuff'
    df2 = DataFrame(columns=['sentimentValue','article'])
    for i in range(0,len(df)):
        row = df.loc[i]
        sentiment= row['sentimentValue']
        article = row['sentence']
        article = article.replace('\\n','')
        article = article.replace('\\u','')
        article = article.replace('1','')
        article = article.replace('2','')
        article = article.replace('3','')
        article = article.replace('4','')
        article = article.replace('5','')
        article = article.replace('6','')
        article = article.replace('7','')
        article = article.replace('8','')
        article = article.replace('9','')
        article = article.replace('0','')
        df2.loc[len(df2)]=[sentiment,article]
    return df2
        
if(__name__=='__main__'):
    main()