#!/usr/bin/env python

import cgi
from goose import Goose
from textblob import TextBlob
import urllib
from nltk.tokenize import sent_tokenize
import scrapy
from bs4 import BeautifulSoup
from sklearn.externals import joblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

form = cgi.FieldStorage()
url = form["url"].value
url = urllib.unquote(url)

g = Goose()
article = g.extract( url = url)

if len(article.cleaned_text[:])!=0:
	article = article.cleaned_text[:]

else:
	url2 = 'http://timesofindia.indiatimes.com/elections/assembly-elections/goa/news/manohar-parrikar-appointed-goa-chief-minister-as-bjp-sews-up-numbers/articleshow/57610803.cms'
	article = g.extract(url=url)
	article = article.cleaned_text[:]

clf = joblib.load('./clf.pkl')
vectorizer = joblib.load('./vector.pkl')

# df = pd.read_csv('./cleanedTrainSentencesDataSet.csv',delimiter=',')
# articles=df['article']

sentence_list = sent_tokenize(article)
length_list = len(sentence_list)
polarity = 0.0

features = vectorizer.transform(sentence_list)
sentiments = clf.predict(features)
polarity = sum(sentiments)
length_list = len(sentence_list)

if length_list>0:
	val1 = float(polarity)/length_list
else:
	val1=5

print """Content-type: text/plain

%s""" % val1