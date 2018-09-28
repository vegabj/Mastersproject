'''
Vegard Bjørgan 2018

reader for raw data files
'''

import pandas as pd
import numpy as np
from os import getcwd
import re

'''
Hepmark
'''

def read_hepmark_microarray():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Microarray/SampleSheet-Hepmark-Microarray.txt"
	path = path + "/Data/Hepmark-Microarray/Hepmark-Microarray.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	df2 = pd.read_csv(sampleSheet, sep='\t', usecols=['Type', 'Code'])
	df = df.dropna()
	#df = df.drop(['509-1-4'])
	return df, df2.loc[:, 'Type'], df2.loc[:, 'Code']


def read_hepmark_tissue():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	path = path + "/Data/Hepmark-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	# Compose sampleSheet metadata
	# TODO: make code readable
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t", usecols=['ID', 'Normal', 'Tumor'], index_col=['ID'])
	sampleSheetDf = sampleSheetDf.dropna()
	index = np.concatenate([sampleSheetDf.loc[:,'Normal'], sampleSheetDf.loc[:,'Tumor']], axis = 0)
	type = ['Tumor' if idx in sampleSheetDf.loc[:, 'Tumor'].values else 'Normal' if idx in sampleSheetDf.loc[:, 'Normal'].values else 'Undefined' for idx in index]
	df2 = pd.DataFrame({'type': type}, index=index)
	df2['group'] = [sampleSheetDf.index[sampleSheetDf['Normal'] == id][0] if not sampleSheetDf.index[sampleSheetDf['Normal'] == id].empty else sampleSheetDf.index[sampleSheetDf['Tumor'] == id][0] for id in df2.axes[0]]
	# TODO: Samplesheet and MatureMatrix missmatch
	df2 = df2.drop(['XXXX', 'ta-164'])
	df = df.drop(['na144_2', 'na-164'])

	return df, df2.loc[:, 'type'], df2.loc[:, 'group']


def read_hepmark_tissue_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	path = path + "/Data/Hepmark-Tissue/MatureMatrixFormatted.csv"
	df = pd.read_csv(path, index_col=0).transpose()
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t", usecols=['ID', 'Normal', 'Tumor'], index_col=['ID'])
	sampleSheetDf = sampleSheetDf.dropna()
	index = np.concatenate([sampleSheetDf.loc[:,'Normal'], sampleSheetDf.loc[:,'Tumor']], axis = 0)
	type = ['Tumor' if idx in sampleSheetDf.loc[:, 'Tumor'].values else 'Normal' if idx in sampleSheetDf.loc[:, 'Normal'].values else 'Undefined' for idx in index]
	df2 = pd.DataFrame({'type': type}, index=index)
	df2['group'] = [sampleSheetDf.index[sampleSheetDf['Normal'] == id][0] if not sampleSheetDf.index[sampleSheetDf['Normal'] == id].empty else sampleSheetDf.index[sampleSheetDf['Tumor'] == id][0] for id in df2.axes[0]]
	df2 = df2.drop(['XXXX', 'ta-164', 'ta157', 'tb140']) # 2 missmatch and 2 extrimities
	return df, df2.loc[:, 'type'], df2.loc[:, 'group']


def read_hepmark_paired_tissue():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Paired-Tissue/SampleSheetPairedSamples-8Mar2017.txt"
	path = path + "/Data/Hepmark-Paired-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t", usecols=['ID', 'Normal', 'Tumor', 'Code'], index_col='ID')
	# Setup types
	normals = sampleSheetDf['Normal'][~pd.isnull(sampleSheetDf['Normal'])]
	tumors = sampleSheetDf['Tumor'][~pd.isnull(sampleSheetDf['Tumor'])]
	sampleSheetDf = sampleSheetDf.dropna(axis=1)
	sampleSheetDf['Type'] = ['Normal' if ax in normals else 'Tumor' if ax in tumors else np.nan for ax in sampleSheetDf.index]
	sampleSheetDf = sampleSheetDf.dropna(axis=0)
	return df, sampleSheetDf.loc[:,'Type'], sampleSheetDf.loc[:, 'Code']


'''
ColonCancer
'''
def read_coloncancer_GCF_2014_295():
	print("SampleSheet missing")

def read_guihuaSun_PMID_26646696_colon():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['Diease', 'File', 'ID', 'Tissue'], index_col="File")
	sampleSheet['group'] = sampleSheet.apply(lambda row: row['ID'].split('-')[0], axis=1)
	# Drop rectal columns
	df['tissue'] = sampleSheet.loc[:, 'Tissue']
	df = df[df.tissue != 'Rectal']
	df = df.drop(['tissue'], axis=1)

	df.dropna()
	return df, sampleSheet.loc[:, 'Diease'], sampleSheet.loc[:, 'group']

def read_guihuaSun_PMID_26646696_rectal():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['Diease', 'File', 'ID', 'Tissue'], index_col="File")
	sampleSheet['group'] = sampleSheet.apply(lambda row: row['ID'].split('-')[0], axis=1)
	# Drop colon columns
	df['tissue'] = sampleSheet.loc[:, 'Tissue']
	df = df[df.tissue != 'Colon']
	df = df.drop(['tissue'], axis=1)

	df.dropna()
	return df, sampleSheet.loc[:, 'Diease'], sampleSheet.loc[:, 'group']

def read_publicCRC_GSE46622_colon():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_GSE46622/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SraRunTable.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['disease_state_s', 'Run_s', 'subject_s', 'tissue_s'], index_col='Run_s')
	# Drop metastasis
	df['disease_state'] = sampleSheet.loc[:, 'disease_state_s']
	sub = df[df.disease_state == 'metastasis']
	df = df.drop(sub.index)
	df = df.drop(['disease_state'], axis=1)
	# Drop rectum columns
	df['tissue'] = sampleSheet.loc[:, 'tissue_s']
	df = df[df.tissue != 'colorectal biopsy, rectum/sigma']
	df = df.drop(['tissue'], axis=1)

	df.dropna()
	return df, sampleSheet.loc[:, 'disease_state_s'], sampleSheet.loc[:, 'subject_s']

def read_publicCRC_GSE46622_rectum():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_GSE46622/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SraRunTable.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['disease_state_s', 'Run_s', 'subject_s', 'tissue_s'], index_col='Run_s')
	# Drop metastasis
	df['disease_state'] = sampleSheet.loc[:, 'disease_state_s']
	sub = df[df.disease_state == 'metastasis']
	df = df.drop(sub.index)
	df = df.drop(['disease_state'], axis=1)
	# Drop colon columns
	df['tissue'] = sampleSheet.loc[:, 'tissue_s']
	df = df[df.tissue == 'colorectal biopsy, rectum/sigma']
	df = df.drop(['tissue'], axis=1)
	df.dropna()
	return df, sampleSheet.loc[:, 'disease_state_s'], sampleSheet.loc[:, 'subject_s']

def read_publicCRC_PMID_23824282():
	'''
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_PMID_23824282/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t",
		usecols=['Characteristics[disease]', 'Run_s', 'subject_s', 'Characteristics[organism part]'],
		index_col='Run_s', names=['disease', 'id'])
	# TODO: Split in colon / rectal tissue and /or Age, Gender and Race

	df.dropna()
	'''
	print("Missing target variable (whole set of same type with no normal tissue to compare)")

def read_publicCRC_PMID_26436952():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_PMID_26436952/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['anonymized_name', 'tumor_type', 'subject_alias', 'disease_site'], index_col='anonymized_name')
	print(sampleSheet)
	sub = sampleSheet[sampleSheet.tumor_type == 'Metastasis']
	sampleSheet = sampleSheet.drop(sub.index)
	print(sampleSheet)

	# TODO: disease_site split into types
	# Liver
	# Colon
	# Lung
	# Rectum

	df.dropna()
	return df, sampleSheet.loc[:, 'tumor_type'], sampleSheet.loc[:, 'subject_alias']
