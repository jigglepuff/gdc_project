
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


samples_path = 'common_sample_ids.csv'
samples_df = pd.read_csv(samples_path)


# In[3]:


mrna_meta_path = 's3://gdc-mrna/files_meta_mRNA.csv'
mrna_meta_df = pd.read_csv(mrna_meta_path)


# In[4]:


sample_list=samples_df['sample_id'].tolist()


# In[5]:


filtered_meta_df=mrna_meta_df[mrna_meta_df['cases.0.samples.0.sample_id'].isin(sample_list)]


# In[6]:


filtered_meta_df=filtered_meta_df[filtered_meta_df['file_name'].str.endswith('.FPKM.txt.gz')]


# In[7]:


# get sample_id & file_id & other meta data correspondance
filtered_meta_df=filtered_meta_df[['file_id','cases.0.samples.0.sample_id', 'cases.0.samples.0.sample_type',
       'cases.0.project.disease_type', 'cases.0.diagnoses.0.primary_diagnosis','cases.0.case_id']]


# In[8]:


import boto3
client = boto3.client('s3')
s3 = boto3.resource('s3')
BUCKET_NAME = 'gdc-mrna'
paginator = client.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=BUCKET_NAME)


# In[14]:


output_fname = "mrna_filtered_matrix.csv"
count = 0 

for page in page_iterator:
    for item in page['Contents']:
        if(item['Key'].startswith('mrna_matrix_')):
            idname = item['Key']
            print ("processing "+idname+"...")
            s3.Bucket(BUCKET_NAME).download_file(idname, idname)
            mrna_df=pd.read_csv(idname)
            mrna_sample_filtered_enriched=pd.merge(mrna_df, filtered_meta_df, on = 'file_id', how = 'inner')
            if count == 0:
                mrna_sample_filtered_enriched.to_csv(output_fname, mode='a', header=True)
            else:
                mrna_sample_filtered_enriched.to_csv(output_fname, mode='a', header=False)
            count +=1
            os.remove(idname)
        


# In[15]:


s3.meta.client.upload_file('data/'+output_fname,BUCKET_NAME,output_fname)

