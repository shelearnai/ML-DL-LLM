from kafka import KafkaConsumer,TopicPartition
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pickle
import os

tp = TopicPartition('csv-loop3',0)
consumer = KafkaConsumer(group_id='test-consumer-group',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                         auto_offset_reset='earliest',
                         enable_auto_commit=False)



consumer.assign([tp])
consumer.seek_to_beginning(tp)  
# obtain the last offset value
lastOffset = consumer.end_offsets([tp])[tp]
all_data=[]
all_label=[]
ch_count=0
for message in consumer:
    mframe = pd.DataFrame(message.value)
    data=mframe.drop('target',axis=1)
    label=mframe['target']
    all_data.append(data)
    all_label.append(label)
    if ch_count!=0:
        with open('checkpoint/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
            lr = pickle.load(f)

        lr.fit(data,label)
        with open('checkpoint/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(lr, f)
    else:
        print('first time training')
        lr=LinearRegression()    
        lr.fit(data,label) 
        with open('checkpoint/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(lr, f)
    ch_count+=1   
    if message.offset == lastOffset - 1:
        break

pred_value=list(map(int,lr.predict(data)))
print("The accuracy score with Linear Regression for heart dataset is {0} %".format(float(accuracy_score(pred_value,label))*100))
#have to produce onnx and do inference

# Convert into ONNX format.
from skl2onnx import to_onnx
with open('checkpoint/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
    lr = pickle.load(f)

arr=np.array(all_data[:1])
label=np.array(label[:1])
#onx = to_onnx(lr, arr[:,1:,:][0])
onx=to_onnx(lr, arr[:,1:,:][0][0].reshape(1,-1))
with open("rf_heart.onnx", "wb") as f:
    f.write(onx.SerializeToString())


# Compute the prediction with onnxruntime.
import onnxruntime as rt
newarr=np.squeeze(arr)
sess = rt.InferenceSession("rf_heart.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: newarr.reshape(newarr[:,1].size,-1).astype(np.double)})[0]
#pred_value=list(map(int,lr.predict(pred_onx)))
pred_value=list(map(int,pred_onx))
#to calculate the prediction obtained from onnx (label and pred_value)
actual_value=list(all_label[0])
print("The accuracy score with Onnx is {0} %".format(float(accuracy_score(pred_value,actual_value))*100))  