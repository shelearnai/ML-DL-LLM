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
from lightgbm import LGBMClassifier
import onnx
import onnxmltools
import onnxmltools.convert.common.data_types
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn

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
    tp = TopicPartition('lgb-breast',0)
    consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'],
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                            auto_offset_reset='latest',
                            enable_auto_commit=False)

    consumer.assign([tp])
    consumer.seek_to_beginning(tp)  
    # obtain the last offset value
    lastOffset = consumer.end_offsets([tp])[tp]
    
def LGBM_model_prediction(ch_count,X_train,y_train):
        model = LGBMClassifier()
        #chkpoint
        if ch_count>50:
            with open('lightgbmchk/lgb_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                model= pickle.load(f)

        model.fit(X_train, y_train)
        with open('lightgbmchk/lgb_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(model, f)

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
        all_data.append(list(map(float,feat)))
        all_label.append(lbl)

        #this code is for KNN who cannot work only one line (one class , so wait until next 10 lines)
        if ch_count<50:
            ch_count+=1
            continue

        LGBM_model_prediction(ch_count,all_data,all_label)
        ch_count=ch_count+1 
        if message.offset == lastOffset - 1:
            break
            
    return all_data,all_label,ch_count

def model_pred(data,label,chk_count):
    #pick last checkpoint
    with open('lightgbmchk/lgb_ckp_{}.p'.format(chk_count-1), 'rb') as f:
        lr = pickle.load(f)
    pred_value=lr.predict(data)
    print("The accuracy score with xgboost  for diabetes dataset is {0} %".format(float(accuracy_score(pred_value,label))*100))

def Onnx_Save_and_Predict(ch_count,all_data,label):
    #ordinary model
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
    )  # noqa
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
        convert_lightgbm,
    )  # noqa
    import onnxmltools.convert.common.data_types
    from skl2onnx.common.data_types import FloatTensorType

    with open('lightgbmchk/lgb_ckp_{}.p'.format(ch_count-1), 'rb') as f:
        pipe = pickle.load(f)
        
    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    model_onnx = convert_sklearn(
        pipe,
        "pipeline_lightgbm",
        [("input", FloatTensorType([None, 30]))],
        target_opset={"": 12, "ai.onnx.ml": 2},
    )

    # And save.
    with open("or_lgbm.onnx", "wb") as f:
        f.write(model_onnx.SerializeToString())

    print("done save")
    #load and predict
    import onnxruntime as rt
    import numpy
    try:
        sess = rt.InferenceSession(
            "or_lgbm.onnx", providers=["CPUExecutionProvider"]
        )
    except OrtFail as e:
        print(e)
        print("The converter requires onnxmltools>=1.7.0")
        sess = None

    if sess is not None:
        pred_onx = sess.run(None, {"input": np.array(data).astype(numpy.float32)})
        print("predict", pred_onx[0])
        print("The accuracy score with light gbm  with onnx is {0} %".format(float(accuracy_score(pred_onx[0],y_train))*100))

if __name__=="__main__":
    #create consumer instance
    create_consumer()
    #receive data

    data,label,ch_count=receive_data()
    model_path='lightgbmchk/lgb_ckp_{}.p'.format(ch_count-1)
    model_pred(data,label,ch_count)
    print(model_path)
    
    Onnx_Save_and_Predict(ch_count,data,label)
    print("All done")