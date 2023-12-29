# Import packages
import pandas as pd
import json
import datetime as dt
from time import sleep
from kafka import KafkaProducer
import time

# Initialize Kafka Producer Client
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
print(f'Initialized Kafka producer at {dt.datetime.utcnow()}')

# Set a basic message counter and define the file path
counter = 0
file = "heart.csv"

for chunk in pd.read_csv(file,encoding='unicode_escape',chunksize=10):

    # Set the counter as the message key
    key = str(counter).encode()

    # Convert the data frame chunk into a dictionary
    chunkd = chunk.to_dict()

    # Encode the dictionary into a JSON Byte Array
    data = json.dumps(chunkd, default=str).encode('utf-8')

    # Send the data to Kafka
    producer.send(topic="csv-loop3", key=key, value=data)

    time.sleep(0.5)
    counter = counter + 1
    print(f'Sent record to topic at time {dt.datetime.utcnow()}')