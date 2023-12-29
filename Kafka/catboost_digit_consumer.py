#consuemr
import pandas as pd
import numpy as np
from kafka import KafkaConsumer,TopicPartition
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
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
from catboost import CatBoostClassifier

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
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

    tp = TopicPartition('cat-digit',0)
    consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'],
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                            auto_offset_reset='latest',
                            enable_auto_commit=False)

    consumer.assign([tp])
    consumer.seek_to_beginning(tp)  
    # obtain the last offset value
    lastOffset = consumer.end_offsets([tp])[tp]
    
def Catboost_model(ch_count,X_train,y_train):
    clf = CatBoostClassifier(
            iterations=5, 
            learning_rate=0.1, 
            #loss_function='CrossEntropy'
    )
    if ch_count>100:
        with open('catbootchk/cat_ckp_{}.p'.format(ch_count-1), 'rb') as f:
            gb= pickle.load(f)

    clf.fit(X_train, y_train, 
            verbose=False
    )
    with open('catbootchk/cat_ckp_{}.p'.format(ch_count), 'wb') as f:
        pickle.dump(clf, f)

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
        all_data.append(feat)
        all_label.append(lbl)

        #this code is for KNN who cannot work only one line (one class , so wait until next 10 lines)
        if ch_count<100:
            ch_count+=1
            continue

        Catboost_model(ch_count,all_data,all_label)
        ch_count=ch_count+1 
        if ch_count>200:
            break
        if message.offset == lastOffset - 1:
            break
            
    return all_data,all_label,ch_count

def model_pred(data,label,chk_count):
    #pick last checkpoint
    with open('catbootchk/cat_ckp_{}.p'.format(chk_count-2), 'rb') as f:
        lr = pickle.load(f)
    pred_value=lr.predict(data)
    print("The accuracy score with xgboost  for diabetes dataset is {0} %".format(float(accuracy_score(pred_value,label))*100))

def Onnx_Save_and_Predict(ch_count,all_data,label):
    from sklearn.metrics import accuracy_score
    print(ch_count)
    with open('catbootchk/cat_ckp_{}.p'.format(ch_count-1), 'rb') as f:
        n_model= pickle.load(f)
        
    n_model.save_model(
    "digit_cat.onnx",
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'test model for MultiClassification',
        'onnx_graph_name': 'CatBoostModel_for_MultiClassification'
    }
    )
    
    #predict
    import onnxruntime as rt
    import numpy as np

    sess = rt.InferenceSession('digit_cat.onnx')

    # or both
    label, probabilities = sess.run(['label', 'probabilities'],
                                    {'features': np.array(data).astype(np.float32)})
    
    pred_value=[max(x,key=x.get) for x in probabilities]
    print("The accuracy score from onnx is {0} %".format(float(accuracy_score(pred_value,label))*100))
    
    
if __name__=="__main__":
    #create consumer instance
    create_consumer()
    #receive data
    
    data,label,ch_count=receive_data()
    model_path='catbootchk/cat_ckp_{}.p'.format(ch_count-1)
    model_pred(data,label,ch_count)
    print(model_path)
    Onnx_Save_and_Predict(ch_count,data,label)
    print("All done")