'''
Vegard Bjørgan 2018

reader for data files
'''

import pandas as pd
import numpy as np
from os import getcwd
import re


def get_sets():
	return ["Hepmark_Microarray", "Hepmark_Tissue", "Hepmark_Paired_Tissue",
			"Coloncancer_GCF_2014_295", "GuihuaSun_PMID_26646696_colon",
			"GuihuaSun_PMID_26646696_rectal", "PublicCRC_GSE46622_colon",
			"PublicCRC_GSE46622_rectal", "PublicCRC_PMID_23824282_colon",
			"PublicCRC_PMID_23824282_rectal", "PublicCRC_PMID_26436952_colon",
			"PublicCRC_PMID_26436952_rectal"]

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
	df = df.drop(['509-1-4'])
	df2 = df2.drop(['509-1-4'])
	return df, df2.loc[:, 'Type'], df2.loc[:, 'Code']


def read_hepmark_tissue():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	path = path + "/Data/Hepmark-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	# Compose sampleSheet metadata
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t",
					usecols=['ID', 'Normal', 'Tumor'], index_col=['ID'])
	sampleSheetDf = sampleSheetDf.dropna()
	index = np.concatenate([sampleSheetDf.loc[:,'Normal'], sampleSheetDf.loc[:,'Tumor']], axis = 0)
	type = ['Tumor' if idx in sampleSheetDf.loc[:, 'Tumor'].values
			else 'Normal' if idx in sampleSheetDf.loc[:, 'Normal'].values
			else 'Undefined' for idx in index]
	df2 = pd.DataFrame({'type': type}, index=index)
	df2['group'] = [sampleSheetDf.index[sampleSheetDf['Normal'] == id][0]
					if not sampleSheetDf.index[sampleSheetDf['Normal'] == id].empty
					else sampleSheetDf.index[sampleSheetDf['Tumor'] == id][0]
					for id in df2.axes[0]]
	# NB: Samplesheet and MatureMatrix missmatch
	df2 = df2.drop(['XXXX', 'ta-164'])
	df = df.drop(['na144_2', 'na-164'])

	return df, df2.loc[:, 'type'], df2.loc[:, 'group']


def read_hepmark_tissue_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	path = path + "/Data/Hepmark-Tissue/MatureMatrixFormatted.csv"
	df = pd.read_csv(path, index_col=0).transpose()
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t",
						usecols=['ID', 'Normal', 'Tumor'], index_col=['ID'])
	sampleSheetDf = sampleSheetDf.dropna()
	index = np.concatenate([sampleSheetDf.loc[:,'Normal'], sampleSheetDf.loc[:,'Tumor']], axis = 0)
	type = ['Tumor' if idx in sampleSheetDf.loc[:, 'Tumor'].values
			else 'Normal' if idx in sampleSheetDf.loc[:, 'Normal'].values
			else 'Undefined' for idx in index]
	df2 = pd.DataFrame({'type': type}, index=index)
	df2['group'] = [sampleSheetDf.index[sampleSheetDf['Normal'] == id][0]
					if not sampleSheetDf.index[sampleSheetDf['Normal'] == id].empty
					else sampleSheetDf.index[sampleSheetDf['Tumor'] == id][0]
					for id in df2.axes[0]]
	df2 = df2.drop(['XXXX', 'ta-164', 'ta157', 'tb140']) # 2 missmatch and 2 extremities
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
	sampleSheetDf['Type'] = ['Normal' if ax in normals else 'Tumor'
							if ax in tumors else np.nan for ax in sampleSheetDf.index]
	sampleSheetDf = sampleSheetDf.dropna(axis=0)
	return df, sampleSheetDf.loc[:,'Type'], sampleSheetDf.loc[:, 'Code']


def read_hepmark_paired_tissue_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Paired-Tissue/SampleSheetPairedSamples-8Mar2017.txt"
	path = path + "/Data/Hepmark-Paired-Tissue/MatureMatrixFormatted.csv"
	df = pd.read_csv(path, index_col = 0).transpose()
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t", usecols=['ID', 'Normal', 'Tumor', 'Code'], index_col='ID')
	# Setup types
	normals = sampleSheetDf['Normal'][~pd.isnull(sampleSheetDf['Normal'])]
	tumors = sampleSheetDf['Tumor'][~pd.isnull(sampleSheetDf['Tumor'])]
	sampleSheetDf = sampleSheetDf.dropna(axis=1)
	sampleSheetDf['Type'] = ['Normal' if ax in normals else 'Tumor'
				if ax in tumors else np.nan for ax in sampleSheetDf.index]
	sampleSheetDf = sampleSheetDf.dropna(axis=0)
	df['Type'] = sampleSheetDf.loc[:, 'Type']
	df = df.dropna()
	df = df.drop('Type', axis = 1)
	return df, sampleSheetDf.loc[:,'Type'], sampleSheetDf.loc[:, 'Code']

'''
ColonCancer
'''

def read_coloncancer_GCF_2014_295():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/ColonCancer_GCF-2014-295/"
	analyses = path + "analyses/MatureMatrix.csv"
	df = pd.read_csv(analyses, sep="\t").transpose()
	df = df.drop(['Sample_240R', 'Sample_240G', 'Sample_335R', 'Sample_335G'])
	target = ['Normal' if ax[-1] == 'R' else 'Tumor' if ax[-1] else 'Undefined' for ax in df.axes[0]]
	group = [ax[:-1] for ax in df.axes[0]]
	# TODO: Determine if R is regular og G is regular
	return df, target, group


def read_coloncancer_GCF_2014_295_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/ColonCancer_GCF-2014-295/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	df = pd.read_csv(analyses, index_col = 0).transpose()
	target = ['Normal' if ax[-1] == 'R' else 'Tumor' if ax[-1] else 'Undefined' for ax in df.axes[0]]
	group = [ax[:-1] for ax in df.axes[0]]
	# TODO: Determine if R is regular og G is regular
	return df, target, group


def read_guihuaSun_PMID_26646696_colon():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, index_col = 0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['Diease', 'File', 'ID', 'Tissue'], index_col="File")
	sampleSheet['group'] = sampleSheet.apply(lambda row: row['ID'].split('-')[0], axis=1)
	# Drop rectal columns
	df['tissue'] = sampleSheet.loc[:, 'Tissue']
	df = df[df.tissue != 'Rectal']
	df = df.drop(['tissue'], axis=1)
	sampleSheet = sampleSheet[sampleSheet.Tissue != 'Rectal']
	sampleSheet = sampleSheet.drop(['Tissue'], axis=1)

	df.dropna()
	return df, sampleSheet.loc[:, 'Diease'], sampleSheet.loc[:, 'group']


def read_guihuaSun_PMID_26646696_rectal():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, index_col = 0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['Diease', 'File', 'ID', 'Tissue'], index_col="File")
	sampleSheet['group'] = sampleSheet.apply(lambda row: row['ID'].split('-')[0], axis=1)
	# Drop colon columns
	df['tissue'] = sampleSheet.loc[:, 'Tissue']
	df = df[df.tissue != 'Colon']
	df = df.drop(['tissue'], axis=1)
	sampleSheet = sampleSheet[sampleSheet.Tissue != 'Colon']
	sampleSheet = sampleSheet.drop(['Tissue'], axis=1)

	df.dropna()
	return df, sampleSheet.loc[:, 'Diease'], sampleSheet.loc[:, 'group']


def read_publicCRC_GSE46622_colon():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_GSE46622/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SraRunTable.txt"
	df = pd.read_csv(analyses, index_col=0).transpose()
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
	sampleSheet = sampleSheet[sampleSheet.tissue_s != 'colorectal biopsy, rectum/sigma']
	sampleSheet = sampleSheet.drop(['tissue_s'], axis=1)
	sampleSheet['disease_state_s'] = ['Normal' if s == 'benign' else 'Tumor' if s == 'tumor' else 'error' for s in sampleSheet.loc[:, 'disease_state_s']]
	sampleSheet = sampleSheet.ix[df.index]

	df.dropna()
	return df, sampleSheet.loc[:, 'disease_state_s'], sampleSheet.loc[:, 'subject_s']


def read_publicCRC_GSE46622_rectal():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_GSE46622/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SraRunTable.txt"
	df = pd.read_csv(analyses, index_col = 0).transpose()
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
	sampleSheet['disease_state_s'] = ['Normal' if s == 'benign' else 'Tumor' if s == 'tumor' else 'error' for s in sampleSheet.loc[:, 'disease_state_s']]
	sampleSheet = sampleSheet.ix[df.index]
	#NB: only 2 samples. Does not work with 4 principal components
	# TODO: Look into this

	return df, sampleSheet.loc[:, 'disease_state_s'], sampleSheet.loc[:, 'subject_s']


def read_publicCRC_PMID_23824282_colon():
	path = r'%s' % getcwd().replace('\\','/')
	analyses = path + "/Data/ColonCancer/PublicCRC_PMID_23824282/analyses/MatureMatrixFormatted.csv"
	df = pd.read_csv(analyses, index_col=0).transpose()
	sspath = path+"/Data/ColonCancer/PublicCRC_PMID_23824282/raw/SampleSheet.txt"
	sampleSheet = pd.read_csv(sspath, sep="\t", index_col='Run_s')
	sampleSheet = sampleSheet[sampleSheet['Characteristics[organism part]'] == 'colon']
	sampleSheet = sampleSheet.drop(['SRR5914652', 'SRR5914656','SRR5914655', ])
	df = df.ix[sampleSheet.index]
	return df, ['Tumor' for i in range(len(df))], [i for i in range(len(df))]


def read_publicCRC_PMID_23824282_rectal():
	path = r'%s' % getcwd().replace('\\','/')
	analyses = path + "/Data/ColonCancer/PublicCRC_PMID_23824282/analyses/MatureMatrixFormatted.csv"
	df = pd.read_csv(analyses, index_col=0).transpose()
	sspath = path+"/Data/ColonCancer/PublicCRC_PMID_23824282/raw/SampleSheet.txt"
	sampleSheet = pd.read_csv(sspath, sep="\t", index_col='Run_s')
	sampleSheet = sampleSheet[sampleSheet['Characteristics[organism part]'] == 'rectum']
	df = df.ix[sampleSheet.index]
	df = df.dropna()

	return df, ['Tumor' for i in range(len(df))], [i for i in range(len(df))]


def read_publicCRC_PMID_26436952_colon():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_PMID_26436952/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, index_col = 0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['anonymized_name', 'tumor_type', 'subject_alias', 'disease_site'], index_col='anonymized_name')
	sub = sampleSheet[sampleSheet.tumor_type == 'Metastasis']
	sampleSheet, df = sampleSheet.drop(sub.index), df.drop(sub.index)
	sub = sampleSheet[sampleSheet.tumor_type == 'Local Recurrence']
	sampleSheet, df = sampleSheet.drop(sub.index), df.drop(sub.index)
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Rectum']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Liver']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Stomach']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Lung']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Ovarian']
	df = df.ix[sampleSheet.index]
	types = ['Normal' if type == 'Normal' else 'Tumor' if type == 'Primary Tumor' else type for type in sampleSheet.loc[:, 'tumor_type']]

	return df, types, sampleSheet.loc[:, 'subject_alias']


def read_publicCRC_PMID_26436952_rectal():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_PMID_26436952/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, index_col = 0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['anonymized_name', 'tumor_type', 'subject_alias', 'disease_site'], index_col='anonymized_name')
	sub = sampleSheet[sampleSheet.tumor_type == 'Metastasis']
	sampleSheet, df = sampleSheet.drop(sub.index), df.drop(sub.index)
	sub = sampleSheet[sampleSheet.tumor_type == 'Local Recurrence']
	sampleSheet, df = sampleSheet.drop(sub.index), df.drop(sub.index)
	sampleSheet = sampleSheet[sampleSheet.disease_site == 'Rectum']
	df = df.ix[sampleSheet.index]
	types = ['Normal' if type == 'Normal' else 'Tumor' if type == 'Primary Tumor' else type for type in sampleSheet.loc[:, 'tumor_type']]

	return df, types, sampleSheet.loc[:, 'subject_alias']


def read_number(i):
	if i == 0:
		return read_hepmark_microarray()
	elif i == 1:
		return read_hepmark_tissue_formatted()
	elif i == 2:
		return read_hepmark_paired_tissue_formatted()
	elif i == 3:
		return read_coloncancer_GCF_2014_295_formatted()
	elif i == 4:
		return read_guihuaSun_PMID_26646696_colon()
	elif i == 5:
		return read_guihuaSun_PMID_26646696_rectal()
	elif i == 6:
		return read_publicCRC_GSE46622_colon()
	elif i == 7:
		return read_publicCRC_GSE46622_rectal()
	elif i == 8:
		return read_publicCRC_PMID_23824282_colon()
	elif i == 9:
		return read_publicCRC_PMID_23824282_rectal()
	elif i == 10:
		return read_publicCRC_PMID_26436952_colon()
	elif i == 11:
		return read_publicCRC_PMID_26436952_rectal()
