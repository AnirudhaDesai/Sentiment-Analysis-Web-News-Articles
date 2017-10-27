#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:35:41 2017
@author: gadde
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def main():
    print "Hello World"
    df_train = pd.read_csv('../Data/TrainSentencesDataSet.csv',delimiter=',')
    df_test = pd.read_csv('../Data/TestSentencesDataSet.csv',delimiter=',')
    
    df_train = df_train.dropna(how='any')
    df_test= df_test.dropna(how='any')

#    Implementing the first model using the tf-idf vectorizer for the classification
    runModel(df_train,df_test)
    PlotGraphs();    
"""
Removing all the alphabets from the document
"""

def KfoldCV(*args):
    kf = KFold(args[0].shape[0], n_folds=3)
    accuracy=[]
    for train,test in kf:     
        #print "Implementing kfold on : ", args[2]
        args[2].fit(args[0][train],args[1][train]) 
        #p = map(args[2].predict, args[0][test])
        p = args[2].predict(args[0][test])
        acc = accuracy_score(args[1][test],p)
        accuracy.append(acc)
     
    return (sum(accuracy)/len(accuracy))

    
def runModel(df_train,df_test):
    print 'This is in the obtainAccuracy method'
    trainX = df_train['article'].values
    trainY = df_train['sentimentValue'].values
    
    testX = df_test['article'].values
    testY = df_test['sentimentValue'].values

    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2,stop_words='english',
                                 ngram_range=(1,3))
      
    vec = vectorizer.fit(trainX)
    joblib.dump(vec, 'vectorSentencesTFIDF.pkl',compress=True)
    trainX = vectorizer.transform(trainX)
    testX = vectorizer.transform(testX)
    '''###########################################################
            HYPER PARAMETER OPTIMIZATION FOR RFC - n_estimators
       #########################################################'''
    accRFC = []
    for n in range(5,40):
        print ("HO loop ",n)
        clf = RandomForestClassifier(n_estimators=n, class_weight='balanced',random_state=50) 
        accRFC.append(KfoldCV(trainX,trainY,clf))
    best_hyperparameter = accRFC.index(max(accRFC))+4
    print ("Best Hyperparameter : " , best_hyperparameter)
    print ("Best accuracy : ",accRFC[best_hyperparameter-4])
    
    # Fit with best hyper parameter
    clf = RandomForestClassifier(n_estimators=best_hyperparameter+4, class_weight='balanced', random_state=50)
    clf = clf.fit(trainX, trainY)
    joblib.dump(clf, 'RFCTFIDF5000.pkl',compress=True)
    results = clf.predict(testX)
    
    print "Accuracy with RFC and best hyperparameter : ",identityFunc(results,testY)
    fp,fn,fne = mainMetric(testY,results)
    print 'False Negatives : ',1-fn
    print 'False Positives : ',1-fp
    print 'False Neutrals  : ',1-fne
    SaveList(accRFC)
    
def SaveList(accRFC):
    npAccuracy = np.asarray(accRFC)
    np.savetxt('npAccuracyRFCTFIDF.txt',npAccuracy)
    
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

def identityFunc(results,values):
    
    accuracyValue = 0
    for i in range(0,len(results)):
        if results[i]==values[i]:
            accuracyValue = accuracyValue + 1
    return float(accuracyValue)/len(results)

def plot_line(x,y):
    plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(x[:],y[:],'or-', linewidth=3) #Plot the first series in red with circle marker
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.xlabel("Minimum Number of Splits") #X-axis label
    plt.ylabel("Classifier Accuracy") #y-axis label
    plt.title("Minimum No. of Splits vs Classification Accuracy") #Plot title
    plt.xlim() #set x axis range
    plt.ylim() #Set yaxis range
    
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    
    #Displays the plots.
    plt.show()

def PlotGraphs():
    accRFC = np.loadtxt('npAccuracyRFCTFIDF.txt')
    plot_line(range(5, len(accRFC)+5), accRFC)
    
if(__name__=='__main__'):
    main()