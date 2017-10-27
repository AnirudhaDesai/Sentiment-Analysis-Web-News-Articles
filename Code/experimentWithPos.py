import os
import re
import nltk
import pandas as pd
from pandas import DataFrame
from nltk import word_tokenize
from nltk import pos_tag
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.svm import LinearSVC

def main():
	print "Hello World"
	df_train = pd.read_csv('../Data/TrainSentencesDataSet.csv',delimiter=',')
	df_test = pd.read_csv('../Data/TestSentencesDataSet.csv',delimiter=',')
	
	df_train = df_train.dropna(how='any')
	df_test = df_test.dropna(how='any')

	df_train = df_train.reset_index()
	df_test = df_test.reset_index()

	trainY = df_train['sentimentValue']
	testY = df_test['sentimentValue']

	trainX = createfeatures(df_train)
	testX = createfeatures(df_test)

	classifier(trainX,trainY,testX,testY)

def createfeatures(df):

	df2 = DataFrame(columns=['nouns','verbs','adjectives','verbsSenti_pos','verbsSenti_neg','verbsSenti_neu','adjsSenti_pos','adjsSenti_neg','adjsSenti_neu'])
	Yvalues = df['sentimentValue'].values

	sid = SentimentIntensityAnalyzer()

	for i in range(0,len(df)):
		print "i is : ",i
		row = df.loc[i]
		sentimentValue = row['sentimentValue']
		article = row['article']
		text = word_tokenize(article)
		pos_tags = pos_tag(text)

		nouns = [w for w in pos_tags if w[1]=='NN' or w[1]=='NNS' or w[1]=='NNP']
		verbs = [w for w in pos_tags if w[1]=='VB' or w[1]=='VBD']
		adjectives = [w for w in pos_tags if w[1]=='JJ']

		verbs_sentiment_neg = 0
		verbs_sentiment_neu = 0
		verbs_sentiment_pos = 0

		adjectives_sentiment_neg = 0
		adjectives_sentiment_neu = 0
		adjectives_sentiment_pos = 0

		for i in range(0,len(verbs)):
			tup = verbs[i]
			word = tup[0]
			polarity = sid.polarity_scores(word)
			verbs_sentiment_neg = verbs_sentiment_neg + polarity['neg']
			verbs_sentiment_pos = verbs_sentiment_pos + polarity['pos']
			verbs_sentiment_neu = verbs_sentiment_neu + polarity['neu']

		for i in range(0,len(adjectives)):
			tup = adjectives[i]
			word = tup[0]
			polarity = sid.polarity_scores(word)
			adjectives_sentiment_neg = adjectives_sentiment_neg + polarity['neg']
			adjectives_sentiment_pos = adjectives_sentiment_pos + polarity['pos']
			adjectives_sentiment_neu = adjectives_sentiment_neu + polarity['neu']
		
		if len(verbs)>0:
			verbs_sentiment_pos = float(verbs_sentiment_pos)/len(verbs)
			verbs_sentiment_neu = float(verbs_sentiment_neu)/len(verbs)
			verbs_sentiment_neg = float(verbs_sentiment_neg)/len(verbs)
		else:
			verbs_sentiment_pos=0
			verbs_sentiment_neg=0
			verbs_sentiment_neu=0

		if len(adjectives)>0:
			adjectives_sentiment_pos = float(adjectives_sentiment_pos)/len(adjectives)
			adjectives_sentiment_neu = float(adjectives_sentiment_neu)/len(adjectives)
			adjectives_sentiment_neg = float(adjectives_sentiment_neg)/len(adjectives)
		else:
			adjectives_sentiment_pos=0
			adjectives_sentiment_neg=0
			adjectives_sentiment_neu=0

		l = len(text)
		if l>0:
			df2.loc[len(df2)]=[float(len(nouns))/l,float(len(verbs))/l,float(len(adjectives))/l,verbs_sentiment_pos,verbs_sentiment_neu,verbs_sentiment_neg,adjectives_sentiment_neu,adjectives_sentiment_pos,adjectives_sentiment_neg]
	return df2.values

def classifier(trainX,trainY,testX,testY):

	clf = LinearSVC(multi_class='ovr', class_weight="balanced")
	clf = clf.fit(trainX,trainY)
	results = clf.predict(testX)

	value = 0

	for i in range(0,len(results)):
		if results[i]==testY[i]:
			value = value+1

	print 'results are : ',results
	print 'testY values are : ',testY

	print 'the accuracy score is :',float(value)/len(results)
	fp,fn,fne = mainMetric(testY,results)
	print 'The false positive is : ',1-fp
	print 'The false negative is : ',1-fn
	print 'The false neutral is : ',1-fne

def mainMetric(testY,results):
    false_positive = 0
    false_negative = 0
    false_neutral=0
    positives = 0
    negatives = 0
    neutrals=0
    for i in range(0,len(testY)):
        if testY[i]==1:
            positives = positives + 1
            if results[i]==1:
                false_positive = false_positive + 1
        elif testY[i]==-1:
            negatives = negatives + 1 
            if results[i]==-1:
                false_negative = false_negative + 1
        elif testY[i]==0:
            neutrals = neutrals+1
            if results[i]==0:
                false_neutral = false_neutral+1
               
    false_postives = float(false_positive)/positives
    false_negatives = float(false_negative)/negatives
    false_neutral = float(false_neutral)/neutrals
    return false_postives,false_negatives,false_neutral

if(__name__=='__main__'):
	main()