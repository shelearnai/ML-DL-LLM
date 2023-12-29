#run with gproject env (low tensorflow version)
#need to chaneg group id name for every run

import os
from datetime import datetime
import time
import threading
import json
from kafka.errors import KafkaError
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow_io as tfio
import tensorflow as tf

def transform_data(data):
    return_arr=[]
    for x in data:
        x=str(x)[2:-5].strip().split(',')
        x_float=list(map(float,x))
        return_arr.append(x_float)
    return return_arr

def get_data(topic):
    dataset1=tfio.experimental.streaming.KafkaGroupIODataset(
    topics=[topic],
    group_id='nwe11',
    servers="localhost:9092",
    configuration=[
        "session.timeout.ms=70000",
        "max.poll.interval.ms=80000",
        "auto.offset.reset=earliest"
    ],
    )
    data=[]
    label=[]
    count=0
    for (message,key) in dataset1:
        num=message.numpy()
        data.append(num)
        label.append(float(key))
        print("in {0}".format(count))
        count+=1
    print('done')
    return data,label


def train_model(arr):
    # Set the parameters
    OPTIMIZER="adam"
    LOSS=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    METRICS=['accuracy']
    EPOCHS=10

    # design/build the model
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(18,),batch_size=None,name='input'),
        #model.add(Input(shape=input_shape, batch_size=None, name='input'))
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    tf.config.run_functions_eagerly(True)
    #model.fit(arr,label,epochs=EPOCHS)
    model.fit(arr,label,epochs=2) #to reduce running time
    return model 

def describe_model():
    print(model.summary())

def evaluate_model(model):
    global x_test
    res = model.evaluate(x_test)
    print("test loss, test acc:", res)

if __name__=="__main__":
    data,label=get_data('susy-train')
    data_arr=transform_data(data)
    model=train_model(data_arr)
    describe_model()

    #get testing data
    test_data,test_label=get_data('susy-test')
    evaluate_model(model)