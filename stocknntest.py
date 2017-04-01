import os
import sqlite3
from mysql import connector
import sys
import tweepy
import requests
import pandas

import numpy as np
from pymongo import MongoClient
from keras.layers.core import Activation,Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
# from textblob import TextBlob
# from IPython.display import SVG



FILE_NAME = "dataset/AAPL.csv"
close_list = []

def stock_prediction():
    print "into stock prediction"
    dataset = []
    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            if n != 0:
                dataset.append(float(line.split(',')[4]))

    dataset = np.array(dataset)
    print "dataset length ::: %d"%(len(dataset))
    # print "dataset ::::" + dataset

    def create_dataset(dataset):
        dataX = [dataset[n] for n in range(len(dataset))]
        print dataX
        return np.array(dataX), dataset[:]

    trainX, trainY = create_dataset(dataset)
    # print "trainX ::: " + trainX
    print trainX
    print trainY
    # print "trainY ::: " + trainY
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation='relu'))
    model.add(Dense(8))
    # model.add(Dense(16))
    #
    # model.add(Dense(16))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY, nb_epoch=200, verbose=2)
    # model.add(LSTM(input_dim=1, output_dim=50,return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100,return_sequences=False))
    # model.add(Dense(1))
    # model.add(Activation('relu'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # model.fit(trainX, trainY, nb_epoch=200, batch_size=512, verbose=2,validation_split=0.05)
    print np.array(dataset)
    prediction = model.predict(dataset, verbose=1)
    evaluation = model.evaluate(trainX,trainY)
    print evaluation
    #print "accuracy ::: %0.6f "%(evaluation[1]*100)
    #print "length ::: " + len(prediction)
    # print "prediction ::: " + prediction
    dataset = np.append(dataset,[prediction[0][0]])
    print "Prediction ::: %.6f" % (prediction[0][0].astype(float))
    #dataset.append(prediction[0][0].astype(float))
    model.summary()
    model.save('model.h5')
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #plot(model, to_file='model.png')
    return prediction[0][0].astype(float)

def evaluate():
    print "into evaluate"
    price = stock_prediction()
    print price
    # for i in range(7):
    #
    #     close_list.append(price)
    # print close_list

if __name__=="__main__":
    evaluate()

