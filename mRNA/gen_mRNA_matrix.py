# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
# from utils import logger
import sys
import gzip
import shutil


def file_as_bytes(file):
    with file:
        return file.read()

def extractMatrix(dirname):
	'''
	return a dataframe of the miRNA matrix, each row is the miRNA counts for a file_id

	'''
	count = 0

	miRNA_data = []
	for idname in os.listdir(dirname):
		# list all the ids 
		if idname.find("-") != -1:
			idpath = dirname +"/" + idname

			# all the files in each id directory
			for filename in os.listdir(idpath):
				# check the miRNA file
				if filename.find(".FPKM.") != -1:
					filepath = idpath + "/" +filename
					with gzip.open(filepath, 'rb') as f_in:
					    with open(filepath[:-3], 'wb') as f_out:
					        shutil.copyfileobj(f_in, f_out)
					# filepath = idpath + "/" + filename[:-3]
					df = pd.read_csv(filepath[:-3],index_col=False,sep="\t",names=['mRNA_id','val'])
	# 				df.columns=[]
	# 				# columns = ["miRNA_ID", "read_count"]
					if count ==0:
						# get the miRNA_IDs 
						miRNA_IDs = df['mRNA_id'].values.tolist()

					id_miRNA_read_counts = [idname] + df.val.values.tolist()
					miRNA_data.append(id_miRNA_read_counts)
					count +=1
					# print (df)
					os.remove(filepath[:-3])
	columns = ["file_id"] + miRNA_IDs
	df = pd.DataFrame(miRNA_data, columns=columns)
	return df

def extractNorm(files):
	# df = pd.read_csv(files, sep="\t")
	df = pd.read_csv(files, sep=",")
	#
	# print (df[columns])
	df['tissue_type'] = df['cases.0.samples.0.sample_type']
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'tissue_type'] = 0
	df.loc[~df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'tissue_type'] = 1
	tumor_count = df.loc[df.tissue_type == 1].shape[0]
	normal_count = df.loc[df.tissue_type == 0].shape[0]
	print("{} Normal samples, {} Tumor samples ".format(normal_count,tumor_count))
	columns = ['file_id','tissue_type']
	return df[columns]



def extractDis(files):
	# df = pd.read_csv(files, sep="\t")
	df = pd.read_csv(files, sep=",")
	# 
	# print (df[columns])
	## TODO: make this clean and smart
	df['label'] = df['cases.0.project.disease_type']
	df.loc[df['cases.0.project.disease_type'].str.contains("Breast"), 'label'] = 1 # TCGA-BRCA
	df.loc[df['cases.0.project.disease_type'].str.contains("Uterine Corpus"), 'label'] = 2 # TCGA-UCEC
	df.loc[df['cases.0.project.disease_type'].str.contains("Head"), 'label'] = 3 # TCGA-HNSC
	df.loc[df['cases.0.project.disease_type'].str.contains("Kidney Renal Clear"), 'label'] = 4 # TCGA-KIRC
	df.loc[df['cases.0.project.disease_type'].str.contains("Lung Adenocarcinoma"), 'label'] = 5 # TCGA-LUAD
	df.loc[df['cases.0.project.disease_type'].str.contains("Brain"), 'label'] = 6 # TCGA-LGG
	df.loc[df['cases.0.project.disease_type'].str.contains("Thyroid"), 'label'] = 7 # TCGA-THCA
	df.loc[df['cases.0.project.disease_type'].str.contains("Prostate"), 'label'] = 8 # TCGA-PRAD
	df.loc[df['cases.0.project.disease_type'].str.contains("Ovarian"), 'label'] = 9 # TCGA-OV
	df.loc[df['cases.0.project.disease_type'].str.contains("Lung Squamous"), 'label'] = 10 # TCGA-LUSC
	df.loc[df['cases.0.project.disease_type'].str.contains("Skin"), 'label'] = 11 # TCGA-SKCM
	df.loc[df['cases.0.project.disease_type'].str.contains("Colon"), 'label'] = 12 # TCGA-COAD
	df.loc[df['cases.0.project.disease_type'].str.contains("Stomach"), 'label'] = 13 # TCGA-STAD
	df.loc[df['cases.0.project.disease_type'].str.contains("Bladder"), 'label'] = 14 # TCGA-BLCA
	df.loc[df['cases.0.project.disease_type'].str.contains("Liver"), 'label'] = 15 # TCGA-LIHC
	df.loc[df['cases.0.project.disease_type'].str.contains("Cervical"), 'label'] = 16 # TCGA-CESC
	df.loc[df['cases.0.project.disease_type'].str.contains("Kidney Renal Papillary"), 'label'] = 17 # TCGA-KIRP
	df.loc[df['cases.0.project.disease_type'].str.contains("Leukemia"), 'label'] = 18 # TCGA-LAML the same as TARGET-AML
	df.loc[df['cases.0.project.disease_type'].str.contains("Sarcoma"), 'label'] = 19 # TCGA-SARC
	df.loc[df['cases.0.project.disease_type'].str.contains("Esophageal"), 'label'] = 20 # TCGA-ESCA
	df.loc[df['cases.0.project.disease_type'].str.contains("Pheochromocytoma"), 'label'] = 21 # TCGA-PCPG
	df.loc[df['cases.0.project.disease_type'].str.contains("Pancreatic"), 'label'] = 22 # TCGA-PAAD
	df.loc[df['cases.0.project.disease_type'].str.contains("Rectum"), 'label'] = 23 # TCGA-READ
	df.loc[df['cases.0.project.disease_type'].str.contains("Testicular"), 'label'] = 24 # TCGA-TGCT
	df.loc[df['cases.0.project.disease_type'].str.contains("Wilms"), 'label'] = 25 # TARGET-WT
	df.loc[df['cases.0.project.disease_type'].str.contains("Thymoma"), 'label'] = 26 # TCGA-THYM
	df.loc[df['cases.0.project.disease_type'].str.contains("Mesothelioma"), 'label'] = 27 # TCGA-MESO
	df.loc[df['cases.0.project.disease_type'].str.contains("Adrenocortical"), 'label'] = 28 # TCGA-ACC
	df.loc[df['cases.0.project.disease_type'].str.contains("Uveal"), 'label'] = 29 # TCGA-UVM
	df.loc[df['cases.0.project.disease_type'].str.contains("Kidney Chromophobe"), 'label'] = 30 # TCGA-KICH
	df.loc[df['cases.0.project.disease_type'].str.contains("Uterine Carcinosarcoma"), 'label'] = 31 # TCGA-UCS
	df.loc[df['cases.0.project.disease_type'].str.contains("Lymphoid"), 'label'] = 32 # TCGA-DLBC
	df.loc[df['cases.0.project.disease_type'].str.contains("Rhabdoid"), 'label'] = 33 # TARGET-RT
	df.loc[df['cases.0.project.disease_type'].str.contains("Cholangiocarcinoma"), 'label'] = 34 # TCGA-CHOL
	df.loc[df['cases.0.project.disease_type'].str.contains("Glioblastoma"), 'label'] = 35 # TCGA-GBM
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'label'] = 0 # Normal
	# checking
	count_arr = []
	for x in range (0,36):
		count = df.loc[df.label == x].shape[0]
		count_arr.append(count)
		print("Label {} count:{}".format(x,count))
	print("Total sample count:{}".format(sum(count_arr)))
	columns = ['file_id','label']
	return df[columns]

if __name__ == '__main__':


	# data_dir ="/Users/yueshi/Downloads/project/data/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	# dirname = data_dir + "live_miRNA"
	# label_file = data_dir + "files_meta.tsv"
	
	# #output file
	# outputfile = data_dir + "miRNA_matrix.csv"

	dirname = sys.argv[1] # dir to live_miRNA
	# file_meta = sys.argv[2] # dir to files_meta.csv
	# cases_meta = sys.argv[2] # dir to files_meta.csv
	outputfile = sys.argv[2] # output filename 

	# extract data
	matrix_df = extractMatrix(dirname)
	# normal_df = extractNorm(file_meta)
	# disease_df = extractDis(file_meta)

	# #merge the two based on the file_id
	# result = pd.merge(matrix_df, disease_df, on='file_id', how="left")
	# #print(result)

	# #save data
	matrix_df.to_csv(outputfile, index=False)
	# #print (labeldf)

 




