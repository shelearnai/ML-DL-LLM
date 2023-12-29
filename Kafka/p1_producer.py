import pandas as pd
import json
import datetime as dt
from time import sleep
from kafka import KafkaProducer
import time

# Initialize Kafka Producer Client
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
print(f'Initialized Kafka producer at {dt.datetime.utcnow()}')

counter = 0
file = "heart.csv"

for chunk in pd.read_csv(file,encoding='unicode_escape',chunksize=10):

    if (counter<10) and len(chunk)!=0:
        # Set the counter as the message key
        key = str(counter).encode()

        # Convert the data frame chunk into a dictionary
        chunkd = chunk.to_dict()

        # Encode the dictionary into a JSON Byte Array
        data = json.dumps(chunkd, default=str).encode('utf-8')

        # Send the data to Kafka
        producer.send(topic="train-test1", key=key, value=data)

        # Sleep to simulate a real-world interval
        time.sleep(0.5)
        
        # Increment the message counter for the message key
        counter = counter + 1
        print(data)
        print(f'Sent record to topic at time {dt.datetime.utcnow()}')
    else:
        break