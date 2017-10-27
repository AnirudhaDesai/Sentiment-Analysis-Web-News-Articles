import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from pandas import DataFrame
from pycorenlp import StanfordCoreNLP

# Main method
def main():
	print "This code is for cleaning the dataset"
	# Reading the scraped dataset.
	df = pd.read_csv('FinalDataLarge.csv',delimiter=',')

	print 'step1'
	# Method to obtain the cleaned articles.
	df_cleaned = dataCleaning(df)
     
	print 'step2'
	# Only the large articles which have a word count greater than 500 are kept in the dataframe and the remaining articles are removed from the dataframe.
	df_cleaned_only_large_articles = obtainArticles(df_cleaned)
  
	print 'step3'
	df_cleaned_only_large_articles = obtainSentencesChrLimit(df_cleaned_only_large_articles)

	print 'step4'
	# The below method s to get the sentiments form the stanford Core NLP Java API. 
	articleswithSentiment  = getSentimentsDf(df_cleaned_only_large_articles)
  
	print 'step5'	
	# Creation of the csv file for the training dataset is done here.
	articleswithSentiment.to_csv('trainDataSet_12000.csv',sep=',')
  
def dataCleaning(df):

	cleaned_df = DataFrame(columns=['sentiment','article'])

	for i in range(0,len(df)):
    
		# Check whether the name of the column is article in the csv file
		page_text = df.loc[i]['article']

		# To remove the <p></p> and similar tags fomr the page text.
		page_text = re.sub(r'<.+?>','',page_text)   

		# To remove the unicode strings from the page_text
		page_text = re.sub(r'[^\x00-\x7F]+',' ',page_text)

		# To remove number from the text file. Other 
		# page_text = re.sub("[^a-zA-Z]"," ",page_text)

		# We need to convert all the words or the text into lower case.
		page_text = page_text.lower()

		# Add the articles to the dataframe to return it to the main method.
		cleaned_df.loc[len(cleaned_df)]=[0,page_text]

	# Return the list of articles to the main method.
	return cleaned_df

# Remove articles from the dataset which has words less than 500
def obtainArticles(dataframe):
	print 'The columns are : ',dataframe.columns
   	df = DataFrame(columns=['sentiment','article'])

    # The below method iterates over all the rows in the dataframe and appends only articles which are larger than 500 words to the new dataframe.
   	for i in range(0,len(dataframe)):

   	    if i ==500:
   	        print 'i is equal to : 500'

   	    row = dataframe.loc[i]
   	    article = row.article

        # split method separates the text into words. 
   	    if len(article.split())>500:
   	        df.loc[len(df)]=[0,article]
   	return df

# To create a new dataframe where the number of characters are limited to 1000. 
# First 1000 charcaters from all the data are created and stored in a dataframe.
def obtainSentencesChrLimit(dataframe):
    
    df  = DataFrame(columns=['sentiment','article'])
    print dataframe.loc[1]

    for i in range(0,len(dataframe)):

        row = dataframe.loc[i]
        article = row.article

        if len(article)>1000:
            article = article[:1000]
            df.loc[len(df)]=[0,article]
        else:
            df.loc[len(df)]=[0,article]
                   
    return df

def getSentimentsDf(dataframe):

	print dataframe.shape
	print dataframe.loc[1]
	# nlp is an object of the stanford core nlp
   	nlp = StanfordCoreNLP("http://localhost:9000")

    # A new dataframe is created to append the sentiment values and return to the main method
   	df = DataFrame(columns=['sentimentValue','article'])

   	for i in range(0,len(dataframe)):
   	    print i
   	    row = dataframe.loc[i]
   	    article = row.article

        # Annotators are the method required to be given to the API. Other annotators are like, NER(Named Entity Recognition etc.).
   	    output =  nlp.annotate(article,properties={'annotators':'sentiment','outputFormat':'json'})
     	
     	# As the stanford Core NLP cannot annotate all the articles, as when the sentences are long enough, it would be difficult to establish dependencies and 
     	# eventually would lead to timeout and would result in an error.
   	    if type(output) is dict:
   	    	
        	# As the output from the stanfordCoreNLP is a list of objects within a dictionary.
   	        outputList = output['sentences']
   	        print 'The outputList is : ',len(outputList)
            # Sentiment is calculated by averaging over all the sentiments. 
   	        sentiment=0
   	        for i in range(0,len(outputList)):
   	            sentiment= sentiment+int(outputList[i]['sentimentValue'])

   	        print 'The sentiment is : ',sentiment
   	        print 'the lenght of the outputlist : ',len(outputList) 
   	        sentiment = (float(sentiment))/len(outputList)

   	        print "The final article sentiment is : ",sentiment

            # Appending the article and its entiment to the dataframe.
   	        df.loc[len(df)]=[sentiment,article]
   	return df
   	
if(__name__=='__main__'):
	main()
