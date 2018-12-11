
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os 

import boto3
import io


# In[3]:
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='gdc-emr0', Key='cpv_filtered_matrix.csv')
df = pd.read_csv(io.BytesIO(obj['Body'].read()))
# df = pd.read_csv("s3://gdc-emr0/cpv_filtered_matrix.csv")


# In[5]:

obj = s3.get_object(Bucket='gdc-emr0', Key='mirna_filtered_matrix.csv')
df2 = pd.read_csv(io.BytesIO(obj['Body'].read()))
# df2 = pd.read_csv("s3://gdc-emr0/mirna_filtered_matrix.csv")


# In[43]:

obj = s3.get_object(Bucket='gdc-emr0', Key='xtrain list.txt')
train_list = pd.read_csv(io.BytesIO(obj['Body'].read()),sep='\n',header=None)
# train_list = pd.read_csv("s3://gdc-emr0/xtrain list.txt",sep='\n',header=None)


# In[178]:


df_combined = pd.merge(df2, df, on="sample_id", how='inner')
del df
del df2

# In[179]:


# sample_labels = df_combined[['sample_id','sample_type_y']]


# In[180]:


disease_type_labels = df_combined[['sample_id','disease_type_y']]


# In[181]:


# primary_diagnosis_labels = df_combined[['sample_id','primary_diagnosis_y']]


# In[182]:


# case_id_identifers = df_combined[['sample_id','case_id_y']]


# In[183]:


df_combined = df_combined.drop(['sample_type_y', 'disease_type_y', 'primary_diagnosis_y','sample_type_x', 'disease_type_x', 'primary_diagnosis_x','case_id_x','case_id_y'], axis=1)


# In[184]:


Xtrain = df_combined.loc[df_combined['sample_id'].isin(train_list[0].tolist()) ]

# In[186]:


Xtest = df_combined.loc[~df_combined['sample_id'].isin(train_list[0].tolist()) ]

del df_combined


# In[185]:


ytrain = disease_type_labels.loc[disease_type_labels['sample_id'].isin(train_list[0].tolist()) ]


# In[187]:


ytest = disease_type_labels.loc[~disease_type_labels['sample_id'].isin(train_list[0].tolist()) ]

del disease_type_labels

# In[188]:


print(len(ytrain),len(ytest),len(ytrain)+len(ytest))


# In[189]:


print(len(Xtrain),len(Xtest),len(Xtrain)+len(Xtest))


# In[190]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labeler = le.fit(ytrain['disease_type_y'])


# In[191]:


print(len(le.classes_))


# In[192]:


ytrain_idx = labeler.transform(ytrain['disease_type_y'])
ytest_idx = labeler.transform(ytest['disease_type_y'])


# In[193]:


Xtrain = Xtrain.drop('sample_id',axis=1)
Xtest = Xtest.drop('sample_id',axis=1)


# In[194]:


scaler = StandardScaler(with_mean=True, with_std=True)
scalarmodel = scaler.fit(Xtrain)
Xtrain = scalarmodel.transform(Xtrain)
Xtest = scalarmodel.transform(Xtest)


# In[ ]:


from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression

svc = SVC(kernel="linear")
# lr = LogisticRegression()

rfecv = RFECV(estimator = svc, step=0.3, cv=StratifiedKFold(5),v=1,n_jobs=-1)
rfecvModel = rfecv.fit(Xtrain, ytrain_idx,scoring='f1')


# In[ ]:


print("Optimal number of features : %d" % rfecv.n_features_)


# In[ ]:


# import matplotlib.pyplot as plt
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

