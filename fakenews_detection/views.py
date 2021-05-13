from django.shortcuts import render
from django.http import HttpResponse;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os
from sklearn.pipeline import Pipeline
import joblib
from sklearn import linear_model
from csv import DictWriter
from csv import writer
import newspaper
from newspaper import Article
# Create your views here.

def home(request):
    return render(request, 'home.html') ;

def result(request):
    value1 = request.GET['title']
    value2 = request.GET['article']
    value3 = request.GET['url']

    # article section
    df = pd.read_csv(r'E:\COLLEGE_PROJECT\fakenews_project\static\data (1).csv')
    #replacing Body nan with Headline
    for i in range(0,df.shape[0]-1):
        if(df.Body.isnull()[i]):
            df.Body[i] = df.Headline[i]
    # ML Code
    y = df.Label
    X = df.Body

    #train_test separation
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

    #Applying tfidf to the data set
    tfidf_vect = TfidfVectorizer(stop_words = 'english')
    tfidf_train = tfidf_vect.fit_transform(X_train)
    tfidf_test = tfidf_vect.transform(X_test)
    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())

    #Applying Naive Bayes
    clf = MultinomialNB()
    clf.fit(tfidf_train, y_train)                       
    pred = clf.predict(tfidf_test)                    
    score = metrics.accuracy_score(y_test, pred)
    # print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred)
    # print(cm)
    #%%
    #Applying Passive Aggressive classifier
    linear_clf = PassiveAggressiveClassifier(max_iter=50)    
    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    # print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred)
    # print(cm)

    def getUserData(value1, value2, value3):
        event = "Submit"
        value1 = value1
        value2 = value2
        value3 = value3
        values = [value1, value2]
        # print(event, values[0], values[1],)
        pipeline = Pipeline([
            ('tfidf',TfidfVectorizer(stop_words = 'english')),
            ('linear_clf',PassiveAggressiveClassifier(max_iter=50)),
            ])
        pipeline.fit(X,y)

        if(value1 == "" and value2 == "" and value3 == ""):
            return "Enter Valid Input"
        
        # url section
        if(value1 == "" and value2 == ""):
            print("value3:",value3)
            url=value3
            # download and parse article
            article = Article(url)
            article.download()
            article.parse()
            #pipeline.predict([values[1]])
            pipeline.predict([article.text])
            filename = 'pipeline1.sav'
            joblib.dump(pipeline, filename)
            filename = './pipeline1.sav'
            loaded_model = joblib.load(filename)
            #result = loaded_model.predict([values[1]])
            result = loaded_model.predict([article.text])
            if result == 1:
                return 'Real News'
            elif result == 0:
                return 'Fake News'

        # article section
        else:
            print("value1:", value1)
            print("value2:", value2)
            pipeline.predict([values[1]])
            filename = 'pipeline1.sav'
            joblib.dump(pipeline, filename)
            filename = './pipeline1.sav'
            loaded_model = joblib.load(filename)
            result = loaded_model.predict([values[1]])

            # code for appending

            # result1 = int(result[0])
            # print(result)
            # # list of column names
            # field_names = ['URLs', 'HEADLINE', 'BODY','LABEL']
            # # Dictionary
            # dict = {'URLs': 6, 'HEADLINE': value1, 'BODY': value2,'LABEL': result1}
            # # Open your CSV file in append mode
            # # Create a file object for this file
            # with open(r'E:\COLLEGE_PROJECT\fakenews_project\static\data (1).csv', 'a') as f_object:
            #     # Pass the file object and a list
            #     # of column names to DictWriter()
            #     # You will get a object of DictWriter
            #     dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            #     # Pass the dictionary as an argument to the Writerow()
            #     dictwriter_object.writerow(dict)
            #     # Close the file object
            #     f_object.close()
            #     print(result1)

            if result == 1:
                return 'Real News'
            elif result == 0:
                return 'Fake News'

    Result = getUserData(value1, value2, value3)
    return render(request, 'result.html', {'finalResult':Result}) ;


 


 
# # print article text
# print(article.title)
# print(article.text)

