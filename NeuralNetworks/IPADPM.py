import time
import pickle
import json
import os
import argparse
import numpy as np
import scipy as sp
import pandas as pd
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.externals import joblib
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
# from sklearn.metrics import accuracy_score,classification_report,log_loss,confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
def group_rr(data,var):

    data[var]=data[var].fillna('-1')
    if('rcv_emp_id' == var):
        data[~(data.isin(['60C','1CL','7EX','6V1','6RP','0IR','7CI']))]='OTH'

    elif('daya_btn_recentcase_currentcase' == var):
        data['Bucket_data']=pd.cut(Bucket_data,1)
    return data

def process_code(data,var,threshold = 1000):
    X=data.copy()
    X=X.fillna('-1').str.get_dummies(sep=';')
    names = X.columns.values
    d={key: var+'_'+key for key in names}
    X.rename(columns=d,inplace=True)
    flag = X.sum()>=threshold
    select=[value for flag,value in zip(flag,X.columns.values) if flag]
    print('process',var,':',len(X.columns.values),'unique values',len(select),sep=' ')
    return X[select]

def process_cat(data,var,threshold = 1000):
    tmp=pd.get_dummies(data.fillna('-1'), prefix=var)
    flag = tmp.sum()>=threshold
    select=[value for flag,value in zip(flag,tmp.columns.values) if flag]
    print('process',var,':',len(tmp.columns.values),'unique values',len(select),'used',sep=' ')
    return tmp[select]

def process_cat0(data,var,threshold = 1000):
    tmp=pd.get_dummies(data.dropna(), prefix=var)
    flag = tmp.sum()>=threshold
    select=[value for flag,value in zip(flag,tmp.columns.values) if flag]
    print('process',var,':',len(tmp.columns.values),'unique values',len(select),'used',sep=' ')
    return tmp[select]

def preprocess(data_folder, file_name, label = None, result_folder = None, threshold = 1000):
    df = pd.read_csv(os.path.join(data_folder, file_name),low_memory=False) # load data
    print('input shape...', df.shape, 'threshold...', threshold)
    X=df.drop(columns=['future_unnecessary_dispatch_flag'])
    X=X.fillna('-1')

    X = df[['circuitid', 'extkey1','wrid','wlid']] #

    vars = ['prev_r14call', 'prev_r14dsp','prev_r30call','prev_r30dsp',
    'prev_30days_necessary_dispatches'
    #'Len_Report'
    ]
    X = X.join(df[vars].fillna(0))

#    vars = ['case_type_lvl2','case_type_lvl3', 'first_qc_overall_status', 'last_qc_overall_status',
#    'first_qc_home_status', 'last_qc_home_status','first_qc_line_status','last_qc_line_status',
#    'downmax','upmax','mdname','swver']
    vars = [
    'recent_first_tc_created',
    'recent_do_flag',
    'rcv_emp_id_bucketed',
    'daya_btn_recentcase_currentcase_bucketed',
    'first_tc_created','type','orig_mpc',
    #'care_center_cd',
    'ir'
       ]

    for var in vars:
        X = X.join(process_cat(df[var],var,threshold=threshold))

    # vars_0 = ['ver_cd_updated']
    #
    # for var in vars_0:
    #     X = X.join(process_cat0(df[var],var,threshold=threshold))

#    vars_bucketing=['rcv_emp_id','daya_btn_recentcase_currentcase']
#
#    for var in vars_bucketing:
#        X = X.join(group_rr(df[var],var))

    '''
    for var in vars:
        X = X.join(group_rr(df[var],var))
    #clssification ['closeout_summary','report']
    print('processed Xfeatures shape:', X.shape)'''


    if label is not None:
        X = X.join(df[[label]])
    X=X.fillna('-1')

    if result_folder is None:
        return X
    else:
        Xfile = os.path.join(result_folder, 'inputs.csv')
        X.to_csv(Xfile, index=False)

def preprocess_predict(pred_folder, pred_file, model_folder):
    X = preprocess(pred_folder, pred_file)
    X_pred_columns=list(X.columns.values)

    # match columns to training data
    Xfile = os.path.join(model_folder, 'inputs.csv')
    X_train_columns= list(pd.read_csv(Xfile, nrows = 0).columns.values)

    # drop extra columns
    new_features = set(X_pred_columns)-set(X_train_columns)
    print('observe %s new features in predicting dataset:' % len(new_features));
    X_pred = X.drop(labels=list(new_features), axis=1) # valid for pandas 0.19.2

    # add missing columns
    missing_features = set(X_train_columns)-set(X_pred_columns)
    print('observe %s missing features in predicting dataset:' % len(missing_features) );
    print(*missing_features,sep='\n')
    d = dict.fromkeys(missing_features, 0)
    tmp = X_pred.assign(**d)

    # reorder columns
    X_final = tmp[X_train_columns]

    X_final.to_csv(os.path.join(pred_folder, 'inputs.csv') , index=False)
    print('Preprocess data for predicting finished!')



def predict_proba(classifiers, model_folder, pred_folder):
    data = pd.read_csv(os.path.join(pred_folder, 'inputs.csv'),low_memory=False)
    label = data.columns[-1]
    X_train = data.drop(labels = ['circuitid', 'extkey1','wrid','wlid',label], axis=1).values
    outputs = { 'circuitid': data['circuitid'],'extkey1': data['extkey1'],
    'wrid':data['wrid'],'wlid':data['wlid']
    }

    for clf in classifiers:
        name = clf.__class__.__name__
        print('-'*40); print('Load {} model ...'.format(name))
        model = pickle.load( open( os.path.join(model_folder, name + '.sav'), "rb" ) )
        print('Making predictions...')
        outputs.update({name + '_preds': model.predict(X_train),
                        name + '_proba': model.predict_proba(X_train)[:,1]})

    pd.DataFrame(outputs).to_csv(os.path.join(pred_folder, 'outputs.csv'),index=False)
    print('Predictions saved successfully!')


def features_importance(result_folder):
    data = pd.read_csv(os.path.join(result_folder, 'inputs.csv'), nrows = 0)
    # label = data.columns[-1]
    # features = list(data.drop(labels = ['circuitid', 'extkey1','wrid','wlid',label], axis=1).columns)
    #
    # clf = pickle.load( open( os.path.join(result_folder, 'BalancedRandomForestClassifier.sav'), "rb" ) )
    # fi=clf.feature_importances_
    # rf_features = pd.Series(fi ,index=features).sort_values(ascending=False)
    # rf_features.to_csv(os.path.join(result_folder, 'imp_BalancedRandomForestClassifier.csv'))
    # print('BalancedRandomForestClassifier feature importance saved!')
    #
    # clf = pickle.load( open( os.path.join(result_folder, 'RandomForestClassifier.sav'), "rb" ) )
    # fi=clf.feature_importances_
    # rf_features = pd.Series(fi ,index=features).sort_values(ascending=False)
    # rf_features.to_csv(os.path.join(result_folder, 'imp_RandomForestClassifier.csv'))
    # print('RandomForestClassifier feature importance saved!')
    # #
    # clf = pickle.load( open( os.path.join(result_folder, 'GradientBoostingClassifier.sav'), "rb" ) )
    # fi=clf.feature_importances_
    # rf_features = pd.Series(fi ,index=features).sort_values(ascending=False)
    # rf_features.to_csv(os.path.join(result_folder, 'imp_GradientBoostingClassifier.csv'))
    # print('GradientBoostingClassifier feature importance saved!')


#================================================================================


classifiers = [
LogisticRegression(n_jobs = -1)
]


print('Training mode set True')
config={
  "data_folder": "/Users/divya.mereddy/Documents/pots/MOdel_Task/TEST1/",
  "train_file": "Train.csv",
  "result_folder": "/Users/divya.mereddy/Documents/pots/MOdel_Task/TEST1/",
  "label":"future_unnecessary_dispatch_flag"
}

data_folder = config['data_folder']
train_file = config['train_file']
result_folder = config['result_folder'] # train result folder
label = config['label']
print(data_folder, train_file, result_folder, label, sep='\n')

preprocess(data_folder, train_file, label, result_folder, threshold = 1000)
#for clf in classifiers:
#    training_model(clf, result_folder, label)
    #predict_proba(classifiers, result_folder, result_folder)
    #features_importance(result_folder)
t1 = time.time()
#name = clf.__class__.__name__
#print('-'*40); print('Training {} model on {} ...'.format(name, label))

Xfile = os.path.join(result_folder, 'inputs.csv')
data = pd.read_csv(Xfile,low_memory=False)

y_train = data[label].values
X_train = data.drop(labels = ['circuitid', 'extkey1','wrid','wlid', label], axis=1).values
data.drop(labels = ['circuitid', 'extkey1','wrid','wlid', label], axis=1).info()
#clf.fit(X_train, y_train)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

    #X_test = sc.transform(X_test)
    # sequential model to initialise our ann and dense module to build the layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
classifier = Sequential()
    # Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 141))

    # Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN | means applying SGD on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100,verbose = 0)

score, acc = classifier.evaluate(X_train, y_train,batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)

config={
  "pred_folder": "/Users/divya.mereddy/Documents/pots/MOdel_Task/TEST1/",
  "pred_file": "Test.csv",
  "model_folder": "/Users/divya.mereddy/Documents/pots/MOdel_Task/TEST1/"
}


pred_folder = config['pred_folder'] # where to store the predicted results
pred_file = config['pred_file'] # the data for predicting; should be put under pred_folder
model_folder = config['model_folder'] # the trained model for predicting
print(pred_folder, pred_file, model_folder, sep='\n')

preprocess_predict(pred_folder, pred_file, model_folder)

data = pd.read_csv(os.path.join(pred_folder, 'inputs.csv'),low_memory=False)
label = data.columns[-1]
X_train = data.drop(labels = ['circuitid', 'extkey1','wrid','wlid',label], axis=1).values
outputs = { 'circuitid': data['circuitid'],'extkey1': data['extkey1'],'wrid':data['wrid'],'wlid':data['wlid']}
y_train=data[label]

folder = pred_folder
file = pred_file
print('Merge predicted results to', file)
df = pd.read_csv(os.path.join(folder, file),low_memory=False)
outputs = pd.read_csv(os.path.join(folder, 'outputs.csv'),low_memory=False)
mdf = df.merge(outputs, how = 'left', on = ['circuitid', 'extkey1','wrid','wlid'])
mdf.to_csv(os.path.join(folder, 'outputs_' + file), index = False)


y_pred = classifier.predict(X_train)
y_pred3 = (y_pred > 0.3)
y_pred5 = (y_pred > 0.5)
y_pred8 = (y_pred > 0.8)

print('*'*20)
score, acc = classifier.evaluate(X_train, y_train,batch_size=10)
print('Test score:', score)


def report(y,y_pred,model_name='Model',data_name='Data'):
    print('-'*40)
    print(model_name,'results for',data_name,sep=' ')
    n=len(y)
    tn, fp, fn, tp=confusion_matrix(y, y_pred,labels=[0,1]).ravel()
    print('ACC:', round(accuracy_score(y, y_pred)*100,2),'%',
          'Precision',round(tp/(tp+fp)*100,2),'%',
          'Recall',round(tp/(tp+fn)*100,2),'%')
    print(' '*12,'Correct Decision')
    print(' '*6,'0 - no dsp','1 - dispatch',' Mix',sep=' | ')
    print('-'*40);
    print('Pred 0',str(round(100*tn/n,2))+'% - TN',str(round(100*fn/n,2))+'% - FN',str(round(100*(fn+tn)/n,2))+'%',sep=' | ')
    print('Pred 1',str(round(100*fp/n,2))+'% - FP',str(round(100*tp/n,2))+'% - TP',str(round(100*(fp+tp)/n,2))+'%',sep=' | ')
    print('-'*40)

def make_table(df, var):
    data = df.copy()
    data['I'] = 1

    tmp = data[['I']+var].groupby(by=var).count()
    tmp.rename(columns={'I':'data_count'},inplace=True)
    table = tmp

    tmp = data[['I']+var].groupby(by=var).count()/data.shape[0]
    tmp.rename(columns={'I':'data_pct'},inplace=True)
    table = table.join(tmp)

    tmp = data[var+['10day_dsp_flag','10day_nd_flag','10day_call_flag']].groupby(by=var).sum()
    tmp['10day_ad'] = tmp['10day_dsp_flag']-tmp['10day_nd_flag']
    table = table.join(tmp[['10day_dsp_flag','10day_ad','10day_call_flag']])

    table['10day_dsp_pct'] = table['10day_dsp_flag']/table['data_count']
    table['10day_call_pct'] = table['10day_call_flag']/table['data_count']
    table['10day_ad_pct']=table['10day_ad']/table['10day_dsp_flag']
    table['10day_ad_pct']=table['10day_ad_pct'].fillna(0)

    f = lambda x: round(x*100,2)
    for var in ['data_pct','10day_call_pct','10day_dsp_pct','10day_ad_pct']:
        table[var]=table[var].apply(f)

    return table[['data_count','data_pct','10day_call_flag','10day_call_pct','10day_dsp_flag','10day_dsp_pct','10day_ad','10day_ad_pct']]
from sklearn.metrics import accuracy_score,classification_report,log_loss,confusion_matrix
#np.array(y_pred, dtype=bool)
count=0
for i in y_train:
    if(i==0):
        count=count+1
print(count)

len(np.where( y_pred == [False]))
print(report(y_train,(y_pred > 9), model_name='future_unnecessary_dispatch_flag', data_name = label ))
