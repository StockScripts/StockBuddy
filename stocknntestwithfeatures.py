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



FILE_NAME = "dataset/AAPL1.csv"
close_list = []

def stock_prediction():
    print "into stock prediction"
    # dataset = []
    # with open(FILE_NAME) as f:
    #     for n, line in enumerate(f):
    #         if n != 0:
    #             dataset.append(float(line.split(',')[4]))
    #
    # dataset = np.array(dataset)
    # print len(dataset)
    # print dataset
    #
    # def create_dataset(dataset):
    #     dataX = [dataset[n + 1] for n in range(len(dataset) - 2)]
    #
    #     return np.array(dataX), dataset[2:]
    dataset = np.loadtxt(FILE_NAME,delimiter=',')

    trainX = dataset[:,0:4]
    trainY = dataset[:,3]
    print trainX
    print trainY
    model = Sequential()
    model.add(Dense(20, input_dim=4,init='uniform', activation='relu'))
    model.add(Dense(40))
    model.add(Dense(40))
    model.add(Dense(20))
    # model.add(Dense(16))
    #
    # model.add(Dense(16))

    model.add(Dense(1))
    model.save('model.hdf5')
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY, nb_epoch=200,batch_size=2048, verbose=2)
    # model.add(LSTM(input_dim=1, output_dim=50,return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100,return_sequences=False))
    # model.add(Dense(1))
    # model.add(Activation('relu'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # model.fit(trainX, trainY, nb_epoch=200, batch_size=512, verbose=2,validation_split=0.05)

    prediction = model.predict(trainX,batch_size=2048)
    evaluation = model.evaluate(trainX,trainY)
    print "Evaluation ::: "
    print evaluation
    print "Accuracy ::: "
    print evaluation*100
    print len(prediction)
    print prediction
    dataset = np.append(dataset,[prediction[0][0]])
    print dataset
    #dataset.append(prediction[0][0].astype(float))

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #plot(model, to_file='model.png')
    return prediction[0][0].astype(float)

def evaluate():
    print "into evaluate"
    print stock_prediction()
    # for i in range(7):
    #     price = stock_prediction()
    #     close_list.append(price)
    # print close_list

if __name__=="__main__":
    evaluate()

