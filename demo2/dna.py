import pandas as pd 
import os 
import sys
import gzip
import shutil
import boto3
client = boto3.client('s3')
s3 = boto3.resource('s3')
BUCKET_NAME = 'gdc-dna'
paginator = client.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=BUCKET_NAME)
miRNA_data = []
count = 0
part = 0 
miRNA_IDs=[]
outputfile = "dna_matrix.csv"
#data/0000c40e-9d45-4446-9dd9-a4676224d0ce
#/logs/jhu-usc.edu_GBM.HumanMethylation450.7.lvl-3.TCGA-19-5955-01A-11D-1697-05.gdc_hg38.txt.parcel

# data/0000c40e-9d45-4446-9dd9-a4676224d0ce
#/jhu-usc.edu_GBM.HumanMethylation450.7.lvl-3.TCGA-19-5955-01A-11D-1697-05.gdc_hg38.txt
for page in page_iterator:
    for item in page['Contents']:
        # e6bb1330-c761-43a5-9d17-16da727232f1.FPKM.txt.gz 
        if(item['Key'].endswith('.gdc_hg38.txt')):
            # download gzip file onto local dir
            fpath = item['Key']
            fname = fpath.split('/')[2]
            idname = fpath.split('/')[1]
            s3.Bucket(BUCKET_NAME).download_file(fpath, fname)
            # unzip
#             with gzip.open(fname, 'rb') as f_in:
#                 with open(fname[:-3], 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
            # read csv
            df = pd.read_csv(fname,index_col=False,sep="\t")
            os.remove(fname)

#             # get the miRNA_IDs 
            if count ==0:
                miRNA_IDs = df['Composite Element REF'].values.tolist() 

            id_miRNA_read_counts = [idname] + df.Beta_value.values.tolist()
            miRNA_data.append(id_miRNA_read_counts)

            count +=1

            if count%50==0 or count is 1:
                columns = ["file_id"] + miRNA_IDs
                df = pd.DataFrame(miRNA_data, columns=columns)
                
                if count==1:
                    df.to_csv(outputfile, index=False,header=True)
                else:
                    df.to_csv(outputfile, mode='a', index=False, header=False)
                miRNA_data[:] = []

            ## update s3 file
            if count % 300 == 0:
                s3_outputfile = 'dna_matrix_'+str(part)+'.csv'
                s3.meta.client.upload_file(outputfile, BUCKET_NAME, s3_outputfile)
                os.remove(outputfile)
                count = 0
                part +=1
                print(part)

                
                
columns = ["file_id"] + miRNA_IDs
df = pd.DataFrame(miRNA_data, columns=columns)

# df.to_csv(outputfile, index=False)
df.to_csv(outputfile, mode='a', index=False, header=False)
s3.meta.client.upload_file(outputfile, BUCKET_NAME, 'dna_matrix_leftover.csv')
