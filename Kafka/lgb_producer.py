# Import packages
import pandas as pd
import json
import datetime as dt
from time import sleep
from kafka import KafkaProducer
import time
import numpy as np

def create_producer():
    # Initialize Kafka Producer Client
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    print(f'Initialized Kafka producer at {dt.datetime.utcnow()}')
    return producer

def load_data(filename):
    data=pd.read_csv(filename)
    data = data.drop(columns = ['Unnamed: 32'], axis = 1)
    data = data.drop(columns = ['id'], axis = 1)
    feature = data.drop(columns =['diagnosis'], axis = 1)
    dia = data['diagnosis']
    feature['diagnosis']=dia
    return feature

def send_data(df,producer):
    cols=df.columns
    row_key_symbol='###'
    each_row=""
    col_key_symbol="&&&"
    label_key_symbol="$$$"

    count=0
    for key,data in df.iterrows():
            count=count+1
            col_key=""
            for x in range(len(cols)):
                if x==len(cols)-2:
                    each_row=each_row+str(data[x])+label_key_symbol
                elif x==len(cols)-1:
                    each_row=each_row+str(data[x])+row_key_symbol
                else:
                    each_row=each_row+str(data[x])+col_key_symbol
            
            # Encode the dictionary into a JSON Byte Array
            data = json.dumps(each_row, default=str).encode('utf-8')
            each_row=""
            # Send the data to Kafka
            producer.send(topic="lgb-breast", value=data)

            if count>20000:
                break
            print("Sending info {0}".format(count))
    
if __name__=="__main__":
# Set a basic message counter and define the file path
    filename='data/data.csv'

    #load data
    df=load_data(filename)
    

    #create producer instance
    producer=create_producer()

    #send data
    send_data(df,producer)
    print("Finished Sending all information of hotel booking")


