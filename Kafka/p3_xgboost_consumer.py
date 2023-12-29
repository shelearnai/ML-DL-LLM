#consuemr
import pandas as pd
import numpy as np
from kafka import KafkaConsumer,TopicPartition
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
import json
import onnxruntime as rt
import tf2onnx
import onnx
import os
import datetime
import pickle
from xgboost import XGBClassifier
import onnx
import onnxmltools
import onnxmltools.convert.common.data_types
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
import json
import onnxruntime as rt
import tf2onnx
import onnx
import os
import datetime
import pickle

def split_data(data):
    #split row
    rows=data.split('###')[0]
    value1=rows.split('&&&')
    features=value1[:len(value1)]
    temp=features[len(features)-1].split('$$$')
    features[len(features)-1]=temp[0]
    label=temp[1].split('###')[0]
    return features,label

def create_consumer():
    global tp,consumer,lastOffset

    tp = TopicPartition('diabetes',0)
    consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'],
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                            auto_offset_reset='latest',
                            enable_auto_commit=False)

    consumer.assign([tp])
    consumer.seek_to_beginning(tp)  
    # obtain the last offset value
    lastOffset = consumer.end_offsets([tp])[tp]
    
def XGB_model_prediction(ch_count,X_train,y_train):
        xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
        #chkpoint
        if ch_count>50:
            with open('chkxgb/xgb/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                gb= pickle.load(f)

        xgb.fit(X_train, y_train)
        with open('chkxgb/xgb/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(xgb, f)

def receive_data():
    global tp,consumer,lastOffset
    all_data=[]
    all_label=[]
    ch_count=0
    
    count=0
    for message in consumer:
        if len(message.value.strip())==0:
            continue
        msg=message.value
        feat,lbl=split_data(msg)   
        lbl=int(float(lbl.strip()))
        all_data.append(list(map(float,feat)))
        all_label.append(lbl)

        #this code is for KNN who cannot work only one line (one class , so wait until next 10 lines)
        if ch_count<50:
            ch_count+=1
            continue

        XGB_model_prediction(ch_count,all_data,all_label)
        ch_count=ch_count+1 
        if ch_count>70:
            break
        if message.offset == lastOffset - 1:
            break
            
    return all_data,all_label,ch_count

def model_pred(data,label,chk_count):
    #pick last checkpoint
    with open('chkxgb/xgb/lr_ckp_{}.p'.format(chk_count-1), 'rb') as f:
        lr = pickle.load(f)
    pred_value=list(map(int,lr.predict(data)))
    print("The accuracy score with xgboost  for diabetes dataset is {0} %".format(float(accuracy_score(pred_value,label))*100))

    
def Onnx_Save_and_Predict(ch_count,all_data,label):
    from sklearn.metrics import accuracy_score
    with open('chkxgb/xgb/lr_ckp_{0}.p'.format(ch_count-1), 'rb') as f:
        n_model= pickle.load(f)
    pred_value=list(map(int,n_model.predict(data)))   

    num_features = len(data[0])
    initial_type = [('feature_input', FloatTensorType([10, num_features]))]
    onx = onnxmltools.convert.convert_xgboost(n_model, initial_types=initial_type)

    # Save your model locally (or where you desire!)
    with open("Onnx_next.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    import onnxruntime as rt
    # setup runtime - load the persisted ONNX model
    sess = rt.InferenceSession("Onnx_next.onnx")

    # get model metadata to enable mapping of new input to the runtime model.
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: all_data[:10]})[0]
    pred_value=list(map(int,pred_onx))
    print("The accuracy score with Onnx in another format is {0} %".format(float(accuracy_score(pred_value,label[:10]))*100))    
    

if __name__=="__main__":
    #create consumer instance
    create_consumer()
    #receive data
    
    data,label,ch_count=receive_data()
    model_path='chkxgb/xgb/lr_ckp_{}.p'.format(ch_count-1)
    model_pred(data,label,ch_count)
    print(model_path)

    Onnx_Save_and_Predict(ch_count,data,label)
    print("All done")