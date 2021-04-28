#!/usr/bin/env python
# coding: utf-8

# librarys for local
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns
import time
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.corpus import stopwords
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')


# librarys for spark
import findspark
findspark.init('/opt/spark')
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit


# Load review data
review_data = pd.read_csv('Berlin/reviews.csv')
print('{} rows and {} columns.'.format(*review_data.shape))
review_data.head()
review_data.info()


# Load listing data
listing_data = pd.read_csv('Berlin/listings.csv')
print('{} rows and {} columns.'.format(*listing_data.shape))

listing_data.head()
listing_data.describe()
listing_data.info()


# merging full df_1 + add only specific columns from df2
df = pd.merge(review_data, listing_data[['neighbourhood_group',
                                         'host_id', 'latitude',
                                         'longitude', 'id', 'room_type']],
              left_on='listing_id', right_on='id', how='left')     
            
df.rename(columns= {'id_x': 'id'}, inplace=True)
df.drop(['id_y'], axis=1, inplace=True)

df.head()

df.info()


# checking shape
print('Dataset has {} rows and {} columns'.format(*df.shape))


# How many accommodations host has?
properties_per_host = pd.DataFrame(df.groupby('host_id')['listing_id'].nunique())
properties_per_host.sort_values(by=['listing_id'], ascending=False, inplace=True)
properties_per_host.head(20)


# --------------------------------------------------------------------


### Top1 host 
top1_host = df.host_id == 8250486
df[top1_host].neighbourhood_group.value_counts()
pd.DataFrame(df[top1_host].groupby('neighbourhood_group')['listing_id'].nunique())

pd.DataFrame(df[top1_host].groupby('room_type')['listing_id'].nunique())


### Top2 host
top2_host = df.host_id == 1625771
df[top2_host].neighbourhood_group.value_counts()
pd.DataFrame(df[top2_host].groupby('neighbourhood_group')['listing_id'].nunique())

pd.DataFrame(df[top2_host].groupby('room_type')['listing_id'].nunique())


### Top3 host
top3_host = df.host_id == 109995917
df[top3_host].neighbourhood_group.value_counts()

pd.DataFrame(df[top3_host].groupby('neighbourhood_group')['listing_id'].nunique())

pd.DataFrame(df[top3_host].groupby('room_type')['listing_id'].nunique())


# ---Top1-3 hosts may mange their accommodations for living.---


# Drop null
df.dropna(inplace=True)
df.info()


# Drop canceled reservation
i = df[df.comments.str.contains("The host canceled this reservation")].index
df.drop(i, inplace=True)


# --------------------------------------------------------------------


# Language Detection
def language_detection(text):
    try:
        return detect(text)
    except:
        return None

df['language'] = df['comments'].apply(language_detection)

df.to_csv('langdetec.csv')

df.language.value_counts().head(10)


# --------------------------------------------------------------------


### visualization
ax = df.language.value_counts().head(6).plot(kind='barh', figsize=(9, 5),
                                             color='lightcoral', fontsize=12)
ax.set_title('\nWhat are the modt frequent languages comments are written in?\n',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Total Number of Comments', fontsize=10)
ax.set_yticklabels(['English', 'German', 'French', 'Spain', 'Italian',
                    'Dutch'])

# create a list to collect the plt.patches data
totals = []
for i in ax.patches:
    totals.append(i.get_width())

# get total
total = sum(totals)

# set individual bar labels 
for i in ax.patches:
    ax.text(x=i.get_width(), y=i.get_y()+0.35,
            s=str(round((i.get_width() / total) * 100, 2)) + '%',
            fontsize=10, color='black')

ax.invert_yaxis()


# Create dataframe by language
df_eng = df[(df['language'] == 'en')]
df_de = df[(df['language'] == 'de')]
df_fr = df[(df['language'] == 'fr')]
df_esp = df[(df['language'] == 'es')]

df_loc = df[['listing_id', 'latitude', 'longitude']]
df_loc = df_loc.drop_duplicates(subset=['listing_id'], keep='last')


# Choose english and take only listing_id and comments to be simple
df_eng = df_eng[['listing_id', 'comments']]


# Preprocessing reviews
df_eng = df_eng.replace({'\n' : ''}, regex=True)
df_eng = df_eng.replace({'\r' : ''}, regex=True)
df_eng = df_eng.replace({'\t' : ''}, regex=True)
df_eng[['comments']] = df_eng[['comments']].apply(lambda x: x + ' ')


# Group by listing_id
df_eng = df_eng.groupby(['listing_id']).agg('sum')


# save file to analyze using pyspark
df_eng.to_csv('df_eng2.csv', sep='\t')


# --------------------------------------------------------------------
### Sentiment Analysis on "local"
start = time.time()

analyzer = SentimentIntensityAnalyzer()

def negative_score(text):
    negative_value = analyzer.polarity_scores(text)['neg']
    return negative_value

def neutral_score(text):
    neutral_value = analyzer.polarity_scores(text)['neu']
    return neutral_value

def positive_score(text):
    positive_value = analyzer.polarity_scores(text)['pos']
    return positive_value

def compound_score(text):
    compound_value = analyzer.polarity_scores(text)['compound']
    return compound_value

df_eng['neg'] = df_eng['comments'].apply(negative_score)
df_eng['neu'] = df_eng['comments'].apply(neutral_score)
df_eng['pos'] = df_eng['comments'].apply(positive_score)
df_eng['compound'] = df_eng['comments'].apply(compound_score)

end = time.time()
print("time : {}".format(end - start))

df_eng

df_eng.to_csv("df_senti_local.csv", sep='\t')


# --------------------------------------------------------------------
### Sentiment Analysis on "Spark"
# Set Spark
conf = SparkConf().setMaster("spark://10.10.20.53:7077").setAppName("sentiment_analy")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


# Create schema 
schema = StructType([ 
    StructField("linsting_id", IntegerType(), True), 
    StructField("comments", StringType(), True)])

df = sqlContext.read.format('com.databricks.spark.csv').options(header = 'true',
                                                                inferschema = 'true').options(delimiter = '\t').schema(schema).load('df_eng.csv')


# Turn into RDD
df_rdd = df.select("comments").rdd.flatMap(lambda x: x)
header = df_rdd.first()
df2 = df_rdd.filter(lambda row: row != header)


# Sentiment Analysis and check time
start = time.time()

analyzer = SentimentIntensityAnalyzer()


def negative_score(text):
    negative_value = analyzer.polarity_scores(text)['neg']
    return negative_value


def neutral_score(text):
    neutral_value = analyzer.polarity_scores(text)['neu']
    return neutral_value


def positive_score(text):
    positive_value = analyzer.polarity_scores(text)['pos']
    return positive_value


def compound_score(text):
    compound_value = analyzer.polarity_scores(text)['compound']
    return compound_value


def sentimentWordsFunct(x):
    senti_list = []
    neg = negative_score(x)
    neu = neutral_score(x)
    pos = positive_score(x)
    compound = compound_score(x)
    senti_list.append(float(neg))
    senti_list.append(float(neu))
    senti_list.append(float(pos))
    senti_list.append(float(compound))
        
    return senti_list

result = df2.map(sentimentWordsFunct)


# Create schema2 for dataframe
schema2 = StructType([StructField("neg", FloatType(), True), 
                      StructField("neu", FloatType(), True), 
                      StructField("pos", FloatType(), True), 
                      StructField("compound", FloatType(), True)])


# Turn to dataframe
sentiment_result = sqlContext.createDataFrame(result, schema2)


# Turn to pandas dataframe
pandas_sentiment = sentiment_result.toPandas()
df_local = pd.read_csv('df_eng.csv', sep='\t')


# Merge dataframe
df_final = pd.concat([df_local, pandas_sentiment], axis=1)

end = time.time()
print("time : {}".format(end - start))



# --------------------------------------------------------------------
# See sentiment reviews graph in Berlin
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

df_eng.hist('neg', bins=25, ax=axes[0, 0], color='lightcoral', alpha=0.6)
axes[0, 0].set_title('Negative Sentiment Score')

df_eng.hist('neu', bins=25, ax=axes[0, 1], color='lightsteelblue', alpha=0.6)
axes[0, 1].set_title('Neutral Sentiment Score')

df_eng.hist('pos', bins=25, ax=axes[1, 0], color='chartreuse', alpha=0.6)
axes[1, 0].set_title('Positive Sentiment Score')

df_eng.hist('compound', bins=25, ax=axes[1, 1], color='navajowhite', alpha=0.6)
axes[1, 1].set_title('Compound Score')

fig.text(0.5, 0.04, 'Sentiment Scores', fontweight='bold', ha='center')
fig.text(0.04, 0.5, 'Number of reviews', fontweight='bold', va='center', rotation='vertical')
plt.suptitle('\nSentiment Analysis of Airbnb Reviews for Berlin\n\n', fontsize=12, fontweight='bold')
plt.show()


# --------------------------------------------------------------------
### WordClouds
def plot_wordcloud(wordcloud, language):
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(language + 'Comments\n', fontsize=18, fontweight='bold')
    plt.show()

listing_data.sort_values(by='number_of_reviews', ascending=False, inplace=True)
review_sort_listing = listing_data[['id', 'number_of_reviews', 'host_id']]
review_sort_listing.head(10)


### Top1 review_number_accomodation
# Input specific listing_id to give WordClouds for host and customerst
review_by_listing1 = df_eng.loc[df_eng.index == 292864]

exword = ['room', 'nice', 'great', 'place', 'stay', 'host', 'Berlin', 'apartment', 'really', 'good', 'everything', 'also', 'hosts', 'us']
stword = stopwords.words('english') + list(exword)

wordcloud = WordCloud(max_font_size=None, max_words=200, background_color='lightgrey',
                      width=3000, height=2000, 
                      stopwords=stword).generate(str(review_by_listing1.comments.values))
wordcloud.to_file('Top1_listing.png')
plot_wordcloud(wordcloud, 'English')


### Top2 review_number_accomodation
# Input specific listing_id to give WordClouds for host and customerst
review_by_listing2 = df_eng.loc[df_eng.index == 517425]

exword = ['room', 'nice', 'great', 'place', 'stay', 'host', 'Berlin', 'apartment', 'really', 'good', 'everything', 'also', 'hosts', 'us']
stword = stopwords.words('english') + list(exword)

wordcloud = WordCloud(max_font_size=None, max_words=200, background_color='lightgrey',
                      width=3000, height=2000, 
                      stopwords=stword).generate(str(review_by_listing2.comments.values))
wordcloud.to_file('Top2_listing.png')
plot_wordcloud(wordcloud, 'English')


### Top4 review_number_accomodation
# Input specific listing_id to give WordClouds for host and customerst
review_by_listing4 = df_eng.loc[df_eng.index == 26970536]
exword = ['room', 'nice', 'great', 'place', 'stay', 'host', 'Berlin', 'apartment', 'really', 'good', 'everything', 'also', 'hosts', 'us']
stword = stopwords.words('english') + list(exword)

wordcloud = WordCloud(max_font_size=None, max_words=200, background_color='lightgrey',
                      width=3000, height=2000, 
                      stopwords=stword).generate(str(review_by_listing4.comments.values))
wordcloud.to_file('Top3_listing.png')
plot_wordcloud(wordcloud, 'English')


# Join dataframe
df_eng.reset_index(inplace=True)
df_map = pd.merge(df_eng, df_loc)
review_sort_listing.reset_index(drop=True, inplace=True)
review_sort_listing.rename(columns = {'id' : 'listing_id'}, inplace=True)

df_map = pd.merge(df_map, review_sort_listing)


# Number of reviews and sentimental values on map
sns.set_style('white')
cmap = sns.cubehelix_palette(rot=-0.5, as_cmap=True)
fig, ax = plt.subplots(figsize=(11, 7))

ax = sns.scatterplot(x='longitude', y='latitude', size='number_of_reviews',
                     sizes=(5, 200), hue='compound', palette=cmap, data=df_map)
ax.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0.0)
plt.title('\nAccommodations in Berlin by Number of Reviews & Sentiment\n', fontsize=12, fontweight='bold')
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

