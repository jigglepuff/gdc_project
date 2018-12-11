
# coding: utf-8

# In[1]:


import pandas as pd 
import os 
import sys
import boto3
import csv


# In[3]:


client = boto3.client('s3')
s3 = boto3.resource('s3')
BUCKET_NAME = 'gdc-mrna'
paginator = client.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=BUCKET_NAME)
miRNA_data = []
count = 0
part = 0
miRNA_IDs=[]
outputfile = "mrna_matrix.csv"


for page in page_iterator:
    for item in page['Contents']:
        # e6bb1330-c761-43a5-9d17-16da727232f1.FPKM.txt.gz 
        if(item['Key'].endswith('.FPKM.txt.gz')):
            # download gzip file onto local dir
            fpath = item['Key']
            fname = fpath.split('/')[2]
            idname = fpath.split('/')[1]
            s3.Bucket(BUCKET_NAME).download_file(item['Key'], fname)
            # unzip
#             with gzip.open(fname, 'rb') as f_in:
#                 with open(fname[:-3], 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
            # read csv 
            df = pd.read_csv(fname, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False,names=['mRNA_id','val'])
#             df = pd.read_csv(fname[:-3],index_col=False,sep="\t",names=['mRNA_id','val'])
             # remove files 
            os.remove(fname)
        
            # get the miRNA_IDs 
            if count ==0:
                miRNA_IDs = df['mRNA_id'].values.tolist() # column names
                
            # per file
            id_miRNA_read_counts = [idname] + df.val.values.tolist()
            # append to all file data
            miRNA_data.append(id_miRNA_read_counts)
            
            count +=1
            
            if count%100==0 or count is 1:
                columns = ["file_id"] + miRNA_IDs
                df = pd.DataFrame(miRNA_data, columns=columns)
                
                if count==1:
                    df.to_csv(outputfile, index=False,header=True)
                else:
                    df.to_csv(outputfile, mode='a', index=False, header=False)
                miRNA_data[:] = []

                ## update s3 file
            if count % 500 == 0:
                s3_outputfile = 'mrna_matrix_'+str(part)+'.csv'
                s3.meta.client.upload_file(outputfile, BUCKET_NAME, s3_outputfile)
                os.remove(outputfile)
                count = 0
                part +=1
                

#             os.remove(fname[:-3])
                
                
columns = ["file_id"] + miRNA_IDs
df = pd.DataFrame(miRNA_data, columns=columns)

# df.to_csv(outputfile, index=False)
df.to_csv(outputfile, mode='a', index=False, header=False)
s3.meta.client.upload_file(outputfile, BUCKET_NAME, 'mrna_matrix.csv')

