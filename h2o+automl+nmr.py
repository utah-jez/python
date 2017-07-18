
# coding: utf-8

# In[185]:

#import packages
import boto
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
from numerapi.numerapi import NumerAPI
from flatten_json import flatten
from time import sleep
from slacker import Slacker


# In[186]:

#setup slackbot
token = 'slack_token'
slack = Slacker(token)


# In[187]:

#setup numerai API
napi = NumerAPI()
napi.credentials = ('email', 'password')
username = 'username'

# In[188]:

h2o.init()
#h2o.remove_all()


# In[189]:

#download dataset
napi.download_current_dataset(dest_path='.',unzip=True)


# In[190]:

#read data into pandas
train = pd.read_csv('numerai_training_data.csv')
tournament = pd.read_csv('numerai_tournament_data.csv')
valid = tournament[tournament['data_type']=='validation']


# In[ ]:

#drop un-needed columns
valid.drop(['id','data_type','era'],axis=1,inplace=True)
train.drop(['id','data_type','era'],axis=1,inplace=True)
tournament.drop(['data_type','era'],axis=1,inplace=True)


# In[ ]:

#read data into h2o
train_h2o = h2o.H2OFrame.from_python(train,destination_frame='train')
valid_h2o = h2o.H2OFrame.from_python(valid,destination_frame='valid')
tourn_h2o = h2o.H2OFrame.from_python(tournament,destination_frame='tournament')


# In[ ]:

#set X and y variables, 
x = list(train.columns[:-1])
y = train.columns[-1]

train_h2o[y] = train_h2o[y].asfactor()
valid_h2o[y] = valid_h2o[y].asfactor()


# Run AutoML for 30 minutes
aml = H2OAutoML(max_runtime_secs = 18000, stopping_metric = 'logloss')
aml.train(x = x, y = y,
            training_frame = train_h2o,
            leaderboard_frame = valid_h2o)


# In[ ]:

#function to submit models
def submit_models(model_no):

    #convert to df
    yo = aml.leaderboard
    df = yo.as_data_frame()
    
    #choose current model
    model = h2o.get_model(df.model_id[model_no])
    
    #run current model
    preds = model.predict(tourn_h2o)
    
    #combine preds with tourney data
    final = tourn_h2o.cbind(preds)
    
    #get preds in final format
    df = final.as_data_frame()
    df['probability'] = df.p1
    df = df[['id','probability']]
    
    #save preds to csv
    df.to_csv('automl_preds.csv',index=False)
    
    #upload predictions
    napi.upload_prediction('automl_preds.csv')
    
    print("prediction from model # " +str(i) + " uploaded to Numerai, checking API for validity")


# In[ ]:

#function to check submission for passing consistency, originality, and concordance
def check_submit():
    ld = napi.get_leaderboard()
    dic_flattened = [flatten(d) for d in ld[0][0]['leaderboard']]
    nmr2 = pd.DataFrame(dic_flattened)
    my_lb = nmr2[nmr2['username'] == username]
    consis = my_lb.logloss_consistency.values[0]
    orig = my_lb.original.values[0]
    conc = my_lb.concordant.values[0]

    if (consis >= 75 and orig == True and conc == True):
        return True, slack.chat.post_message('#channel', 'AutoML Model passed! with model # '+str(i))
    else:
        return False,  slack.chat.post_message('#channel', 'RD Model# ' +str(i) + ' failed, trying again!')


# In[ ]:

#run functions
for i in range(4,29):
    submit_models(i)
    sleep(60)
    test = check_submit()
    if test[0] == True:
        break

