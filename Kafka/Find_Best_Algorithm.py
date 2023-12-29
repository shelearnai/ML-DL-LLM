
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import pickle

class FindAlgo:
    
    def __init__(self):
        self.X_train=""
        self.X_test=""
        self.y_train=""
        self.y_test=""
        self.algo_name=[]
        self.algo_acc_score=[]
        self.algo_instance={}

    def _set_dataset(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    def LR_model_prediction1(self,ch_count):
        lr = LogisticRegression()
        print('in lR')
        #chkpoint

        if ch_count>50:
            with open('checkpoint1/logreg/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                lr = pickle.load(f)

        lr.fit(self.X_train, self.y_train)
        self.algo_instance['logreg']=lr
        with open('checkpoint1/logreg/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(lr, f)
        self.algo_instance['logreg']=lr

    def KNN_model_prediction1(self,ch_count):
        knn = KNeighborsClassifier()
        print('in KNN')
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/knn/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                knn= pickle.load(f)

        knn.fit(self.X_train, self.y_train)
        with open('checkpoint1/knn/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(knn, f)
        self.algo_instance['knn']=knn

    def RFC_model_prediction1(self,ch_count):
        rd_clf = RandomForestClassifier()
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/rfc/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                rd_clf= pickle.load(f)

        rd_clf.fit(self.X_train, self.y_train)
        with open('checkpoint1/rfc/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(rd_clf, f)
        self.algo_instance['rfc']=rd_clf

    def DT_model_prediction1(self,ch_count):
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier()
        
        print('in decision tree')
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/dtc/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                dtc= pickle.load(f)

        dtc.fit(self.X_train, self.y_train)
        with open('checkpoint1/dtc/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(dtc, f)
        self.algo_instance['dtc']=dtc


    def ADB_model_prediction1(self,ch_count):
        dtc = DecisionTreeClassifier()
        import pickle
        #base_model=self.DT_model_prediction1(ch_count)
        #load last dt model checkpoint
        with open('checkpoint1/dtc/lr_ckp_{}.p'.format(ch_count), 'rb') as f:
            dtc= pickle.load(f)
        ada = AdaBoostClassifier(base_estimator =dtc )
        print('in adaboost')
        import pickle
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/adb/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                ada= pickle.load(f)

        ada.fit(self.X_train, self.y_train)
        with open('checkpoint1/adb/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(ada, f)
        self.algo_instance['adb']=ada


    def GBC_model_prediction1(self,ch_count):
        gb = GradientBoostingClassifier()
        print('in gradient boosting')
        import pickle
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/gbc/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                gb= pickle.load(f)

        gb.fit(self.X_train, self.y_train)
        with open('checkpoint1/gbc/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(gb, f)
        self.algo_instance['gbc']=gb
    
    def XGB_model_prediction1(self,ch_count):
        xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
        print('in xgb boost')
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/xgb/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                gb= pickle.load(f)

        xgb.fit(self.X_train, self.y_train)
        with open('checkpoint1/xgb/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(xgb, f)
        self.algo_instance['xgb']=xgb

    def CBC_model_prediction1(self,ch_count):
        
        cat = CatBoostClassifier(iterations=100)
        print('in cat boost')
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/cbc/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                cat= pickle.load(f)

        cat.fit(self.X_train, self.y_train)
        with open('checkpoint1/cbc/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(cat, f)
        self.algo_instance['cbc']=cat

    def ExtraTreeClassifier_model_predict1(self,ch_count):
        etc = ExtraTreesClassifier()
        print('in extra tree')
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/etc/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                etc= pickle.load(f)

        etc.fit(self.X_train, self.y_train)
        with open('checkpoint1/etc/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(etc, f)
        self.algo_instance['etc']=etc

    def LGBM_model_predict1(self,ch_count):
        lgbm = LGBMClassifier(learning_rate = 1)
        print(' in lgbm ')
        #chkpoint
        if ch_count>50:
            with open('checkpoint1/lgbm/lr_ckp_{}.p'.format(ch_count-1), 'rb') as f:
                lgbm= pickle.load(f)

        lgbm.fit(self.X_train, self.y_train)
        with open('checkpoint1/lgbm/lr_ckp_{}.p'.format(ch_count), 'wb') as f:
            pickle.dump(lgbm, f)
        self.algo_instance['lgbm']=lgbm

    def Load_Chekpoint_Check_Acc(self,ch_count):
        from sklearn.metrics import accuracy_score
        #calculate accuracy score
        acc_score={}
        for x in self.algo_instance:
            pred_value=self.algo_instance[x].predict(self.X_train)
            acc_score[x]=float(accuracy_score(pred_value,self.y_train))*100

        #find best algorithm
        return max(acc_score, key=acc_score.get)


    """def find_best_algo(self):
        scores,instance_arr=self.call_all_algorithm()
        
        inst,scr=self.Voting_model_predict()
        scores.append(scr)
        instance_arr.append(inst)
        
        models = pd.DataFrame({
        'Model' : ['Logistic Regression', 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
                 'Gradient Boosting Classifier', 'XgBoost', 'Cat Boost', 'Extra Trees Classifier', 'LGBM', 'Voting Classifier'
                  ],
        'Score' : scores, 
        'instance':instance_arr
        })

        models.sort_values(by = 'Score', ascending = False,inplace=True)

        highest_acc_model=models.iloc[0]['instance']
        return highest_acc_model,models.iloc[0]['Model']

findalgo_obj=FindAlgo(X_train,X_test,y_train,y_test)
best_model,best_model_name=findalgo_obj.find_best_algo()
print(best_model_name)
best_model.fit(X_train,y_train)
pred=best_model.predict(X_test)
accuracy_score(pred,y_test)"""