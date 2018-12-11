
# coding: utf-8

# In[49]:


import pandas as pd
import os


# In[50]:


samples_path = 'common_sample_ids.csv'
samples_df = pd.read_csv(samples_path)


# In[51]:


mrna_meta_path = 'files_meta_mRNA.csv'
mrna_meta_df = pd.read_csv(mrna_meta_path)


# In[52]:


sample_list=samples_df['sample_id'].tolist()


# In[53]:


filtered_meta_df=mrna_meta_df[mrna_meta_df['cases.0.samples.0.sample_id'].isin(sample_list)]


# In[54]:


filtered_meta_df=filtered_meta_df[filtered_meta_df['file_name'].str.endswith('.FPKM.txt.gz')]


# In[55]:


# get sample_id & file_id & other meta data correspondance
filtered_meta_df=filtered_meta_df[['file_id','cases.0.samples.0.sample_id', 'cases.0.samples.0.sample_type',
       'cases.0.project.disease_type', 'cases.0.diagnoses.0.primary_diagnosis','cases.0.case_id']]


# In[56]:


output_fname = "mrna_filtered_matrix_test.csv"
dirname = "gdc-mrna"
count = 0 
for idname in os.listdir(dirname):
    # list all the ids 
    if idname.find("mrna_matrix_0") == True:
        print ("processing "+idname+"...")
        mrna_df=pd.read_csv(dirname+'/'+idname)
        mrna_sample_filtered_enriched=pd.merge(mrna_df, filtered_meta_df, on = 'file_id', how = 'inner')
        if count == 0:
            mrna_sample_filtered_enriched.to_csv(output_fname, mode='a', header=True)
        else:
            mrna_sample_filtered_enriched.to_csv(output_fname, mode='a', header=False)
    count +=1
    # os.remove(dirname+'/'+idname)
        

