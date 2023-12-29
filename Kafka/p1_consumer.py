from kafka import KafkaConsumer, TopicPartition
import json
import pandas as pd

# Consume all the messages from the topic but do not mark them as 'read' (enable_auto_commit=False)
# so that we can re-read them as often as we like.

# prepare consumer
tp = TopicPartition('train-test1',0)
consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                        auto_offset_reset='earliest',
                         group_id='test-consumer-group',enable_auto_commit=False)
"""consumer = KafkaConsumer('test',
                         group_id='test-consumer-group',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                         auto_offset_reset='earliest',
                         enable_auto_commit=False)"""
consumer.assign([tp])
consumer.seek_to_beginning(tp)  
# obtain the last offset value
lastOffset = consumer.end_offsets([tp])[tp]

import numpy as np
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.metrics import accuracy_score
import pickle
ch_count=0
for message in consumer:
    print(message.value)
    mframe = pd.DataFrame(message.value)
    data=mframe.drop('target',axis=1)
    label=mframe['target']
    print(data)
    lr.fit(data,label)
    ch_count+=1
    with open('checkpoint/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
        pickle.dump(lr, f)
        
    if message.offset == lastOffset - 1:
        break
print('outside of for')
pred_value=list(map(int,lr.predict(data)))
print(accuracy_score(pred_value,label))