"""
Vegard Bjørgan 2019

Readers for data sets.

read_main() is used in most cases as it includes all functionality to extract raw data,
log transformed data and enrichment scores for each data set through a user interface.
"""

import pandas as pd
import numpy as np
from os import getcwd
import df_utils
import warnings

def get_sets():
	return ["Hepmark_Microarray", "Hepmark_Tissue", "Hepmark_Paired_Tissue",
			"Coloncancer_GCF_2014_295", "GuihuaSun_PMID_26646696",
			"PublicCRC_GSE46622", "PublicCRC_PMID_23824282","PublicCRC_PMID_26436952"]

'''
Hepmark
'''

def read_hepmark_microarray():
	path = r'%s' % getcwd().replace('\\','/')
	samplesheet_path = path + "/Data/Hepmark-Microarray/SampleSheet-Hepmark-Microarray.txt"
	path = path + "/Data/Hepmark-Microarray/Hepmark-Microarray.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	samplesheet = pd.read_csv(samplesheet_path, sep='\t', usecols=['Type', 'Code'])
	df = df.dropna()
	df = df.drop(['509-1-4'])
	samplesheet = samplesheet.drop(['509-1-4'])
	df.index = ["X"+ix for ix in df.index]
	samplesheet.index = ["X"+ix for ix in samplesheet.index]
	return df, samplesheet.loc[:, 'Type'], samplesheet.loc[:, 'Code']


def read_hepmark_tissue():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	path = path + "/Data/Hepmark-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	# Compose sampleSheet metadata
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t",
					usecols=['Code', 'Normal', 'Tumor'], index_col=['Code'])
	sampleSheetDf = sampleSheetDf.dropna()
	index = np.concatenate([sampleSheetDf.loc[:,'Normal'], sampleSheetDf.loc[:,'Tumor']], axis = 0)
	type = ['Tumor' if idx in sampleSheetDf.loc[:, 'Tumor'].values
			else 'Normal' if idx in sampleSheetDf.loc[:, 'Normal'].values
			else 'Undefined' for idx in index]
	samplesheet = pd.DataFrame({'type': type}, index=index)
	samplesheet['group'] = [sampleSheetDf.index[sampleSheetDf['Normal'] == id][0]
					if not sampleSheetDf.index[sampleSheetDf['Normal'] == id].empty
					else sampleSheetDf.index[sampleSheetDf['Tumor'] == id][0]
					for id in samplesheet.axes[0]]
	# NB: Samplesheet and MatureMatrix missmatch
	samplesheet = samplesheet.drop(['XXXX', 'ta-164'])
	df = df.drop(['na144_2', 'na-164'])

	return df, samplesheet.loc[:, 'type'], samplesheet.loc[:, 'group']

def read_hepmark_tissue_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	sampleSheet = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	path = path + "/Data/Hepmark-Tissue/MatureMatrixFormatted.csv"
	df = pd.read_csv(path, index_col=0).transpose()
	sampleSheetDf = pd.read_csv(sampleSheet, sep="\t",
						usecols=['Code', 'Normal', 'Tumor'], index_col=['Code'])
	sampleSheetDf = sampleSheetDf.dropna()
	index = np.concatenate([sampleSheetDf.loc[:,'Normal'], sampleSheetDf.loc[:,'Tumor']], axis = 0)
	type = ['Tumor' if idx in sampleSheetDf.loc[:, 'Tumor'].values
			else 'Normal' if idx in sampleSheetDf.loc[:, 'Normal'].values
			else 'Undefined' for idx in index]
	samplesheet = pd.DataFrame({'type': type}, index=index)
	samplesheet['group'] = [sampleSheetDf.index[sampleSheetDf['Normal'] == id][0]
					if not sampleSheetDf.index[sampleSheetDf['Normal'] == id].empty
					else sampleSheetDf.index[sampleSheetDf['Tumor'] == id][0]
					for id in samplesheet.axes[0]]
	samplesheet = samplesheet.drop(['XXXX', 'ta-164', 'ta157', 'tb140', 'nc014']) # 2 missmatch and 3 bad samples
	df = df.drop(['nc014']) #  1 bad samples
	return df, samplesheet.loc[:, 'type'], samplesheet.loc[:, 'group']


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
	df['Type'] = sampleSheetDf.loc[:, 'Type']
	df = df.dropna()
	df = df.drop('Type', axis = 1)
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


def read_guihuaSun_PMID_26646696():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep='\t').transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['Diease', 'File', 'ID', 'Tissue'], index_col="File")
	sampleSheet['group'] = sampleSheet.apply(lambda row: row['ID'].split('-')[0], axis=1)

	return df, sampleSheet.loc[:, 'Diease'], sampleSheet.loc[:, 'group']

def read_guihuaSun_PMID_26646696_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrixFormatted.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, index_col = 0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['Diease', 'File', 'ID', 'Tissue'], index_col="File")
	sampleSheet['group'] = sampleSheet.apply(lambda row: row['ID'].split('-')[0], axis=1)
	df.index = ["X"+ix for ix in df.index]
	sampleSheet.index = ["X"+ix for ix in sampleSheet.index]

	return df, sampleSheet.loc[:, 'Diease'], sampleSheet.loc[:, 'group']


def read_publicCRC_GSE46622():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_GSE46622/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SraRunTable.txt"
	df = pd.read_csv(analyses, sep="\t", index_col=0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['disease_state_s', 'Run_s', 'subject_s'], index_col='Run_s')
	# Drop metastasis
	df['disease_state'] = sampleSheet.loc[:, 'disease_state_s']
	sub = df[df.disease_state == 'metastasis']
	df = df.drop(sub.index)
	df = df.drop(['disease_state'], axis=1)
	sampleSheet['disease_state_s'] = ['Normal' if s == 'benign' else 'Tumor' if s == 'tumor' else 'error' for s in sampleSheet.loc[:, 'disease_state_s']]
	sampleSheet = sampleSheet.ix[df.index]
	return df, sampleSheet.loc[:, 'disease_state_s'], sampleSheet.loc[:, 'subject_s']

def read_publicCRC_GSE46622_formatted():
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
	sampleSheet['disease_state_s'] = ['Normal' if s == 'benign' else 'Tumor' if s == 'tumor' else 'error' for s in sampleSheet.loc[:, 'disease_state_s']]
	sampleSheet = sampleSheet.ix[df.index]
	return df, sampleSheet.loc[:, 'disease_state_s'], sampleSheet.loc[:, 'subject_s']


def read_publicCRC_PMID_23824282():
	path = r'%s' % getcwd().replace('\\','/')
	analyses = path + "/Data/ColonCancer/PublicCRC_PMID_23824282/analyses/MatureMatrix.csv"
	df = pd.read_csv(analyses, sep='\t', index_col=0).transpose()
	sspath = path+"/Data/ColonCancer/PublicCRC_PMID_23824282/raw/SampleSheet.txt"
	sampleSheet = pd.read_csv(sspath, sep="\t", index_col='Run_s')
	sampleSheet = sampleSheet.drop(['SRR5914652', 'SRR5914656','SRR5914655', 'SRR5914651'])
	df = df.loc[sampleSheet.index]
	return df, ['Tumor' for i in range(len(df))], [i for i in range(len(df))]

def read_publicCRC_PMID_23824282_formatted():
	path = r'%s' % getcwd().replace('\\','/')
	analyses = path + "/Data/ColonCancer/PublicCRC_PMID_23824282/analyses/MatureMatrixFormatted.csv"
	df = pd.read_csv(analyses, index_col=0).transpose()
	sspath = path+"/Data/ColonCancer/PublicCRC_PMID_23824282/raw/SampleSheet.txt"
	sampleSheet = pd.read_csv(sspath, sep="\t", index_col='Run_s')
	sampleSheet = sampleSheet.drop(['SRR5914652', 'SRR5914656','SRR5914655', 'SRR5914651'])
	df = df.loc[sampleSheet.index]
	return df, ['Tumor' for i in range(len(df))], [i for i in range(len(df))]


def read_publicCRC_PMID_26436952():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/PublicCRC_PMID_26436952/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep='\t', index_col = 0).transpose()
	sampleSheet = pd.read_csv(raw, sep="\t", usecols=['anonymized_name', 'tumor_type', 'subject_alias', 'disease_site'], index_col='anonymized_name')
	sub = sampleSheet[sampleSheet.tumor_type == 'Metastasis']
	sampleSheet, df = sampleSheet.drop(sub.index), df.drop(sub.index)
	sub = sampleSheet[sampleSheet.tumor_type == 'Local Recurrence']
	sampleSheet, df = sampleSheet.drop(sub.index), df.drop(sub.index)
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Liver']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Stomach']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Lung']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Ovarian']
	df = df.ix[sampleSheet.index]
	sampleSheet['tumor_type'] = sampleSheet['tumor_type'].map({'Primary Tumor' : 'Tumor', 'Normal' : 'Normal'})
	# Note: Had to make "blocks / subject" non integer as it does not work with R modules.
	sampleSheet['subject_alias'] = sampleSheet['subject_alias'].map('S{}'.format)

	return df, sampleSheet.loc[:, 'tumor_type'], sampleSheet.loc[:, 'subject_alias']

def read_publicCRC_PMID_26436952_formatted():
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
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Liver']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Stomach']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Lung']
	sampleSheet = sampleSheet[sampleSheet.disease_site != 'Ovarian']
	df = df.ix[sampleSheet.index]
	# TODO: sampleSheet['tumor_type'] = sampleSheet['tumor_type'].map({'Primary Tumor' : 'Tumor', 'Normal' : 'Normal'})
	types = ['Normal' if type == 'Normal' else 'Tumor' if type == 'Primary Tumor' else type for type in sampleSheet.loc[:, 'tumor_type']]

	return df, types, sampleSheet.loc[:, 'subject_alias']


"""
Enrichement_scores
"""

def read_enrichment_hepmark_microarray():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_hepmark_microarray.csv"
	df = pd.read_csv(path, index_col = 0)
	return df

def read_enrichment_hepmark_tissue():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_hepmark_tissue.csv"
	df = pd.read_csv(path, index_col = 0)
	df = df.drop(['ta157', 'tb140'])
	return df

def read_enrichment_hepmark_paired_tissue():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_hepmark_paired_tissue.csv"
	df = pd.read_csv(path, index_col = 0)
	return df

def read_enrichment_colon():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_colon.csv"
	df = pd.read_csv(path, index_col = 0)
	return df

def read_enrichment_guihuasun():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_guihuasun.csv"
	df = pd.read_csv(path, index_col = 0)
	df.index = ["X"+ix for ix in df.index]
	return df

def read_enrichment_gse46622():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_gse46622.csv"
	df = pd.read_csv(path, index_col = 0)
	return df

def read_enrichment_PMID_23824282():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_PMID_23824282.csv"
	df = pd.read_csv(path, index_col = 0)
	return df

def read_enrichment_PMID_26436952():
	path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/es_PMID_26436952.csv"
	df = pd.read_csv(path, index_col = 0)
	return df


"""
Main methods
"""

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
		return read_guihuaSun_PMID_26646696_formatted()
	elif i == 5:
		return read_publicCRC_GSE46622_formatted()
	elif i == 6:
		return read_publicCRC_PMID_23824282_formatted()
	elif i == 7:
		return read_publicCRC_PMID_26436952_formatted()


def read_number_raw(i):
	if i == 0:
		return read_hepmark_microarray()
	elif i == 1:
		return read_hepmark_tissue()
	elif i == 2:
		return read_hepmark_paired_tissue()
	elif i == 3:
		return read_coloncancer_GCF_2014_295()
	elif i == 4:
		return read_guihuaSun_PMID_26646696()
	elif i == 5:
		return read_publicCRC_GSE46622()
	elif i == 6:
		return read_publicCRC_PMID_23824282()
	elif i == 7:
		return read_publicCRC_PMID_26436952()

def read_es(i):
	if i == 0:
		return read_enrichment_hepmark_microarray()
	elif i == 1:
		return read_enrichment_hepmark_tissue()
	elif i == 2:
		return read_enrichment_hepmark_paired_tissue()
	elif i == 3:
		return read_enrichment_colon()
	elif i == 4:
		return read_enrichment_guihuasun()
	elif i == 5:
		return read_enrichment_gse46622()
	elif i == 6:
		return read_enrichment_PMID_23824282()
	elif i == 7:
		return read_enrichment_PMID_26436952()

def read_main(raw=False):
	names = get_sets()
	print("Available data sets are:")
	for i,e in enumerate(names):
	    print(str(i)+":", e)
	selected = input("Select data set (multiselect separate with ' '): ")
	selected = selected.split(' ')

	multi_select = False if len(selected) == 1 else True
	if multi_select:
		dfs, target, group, es = [], [], [], []
		for select in selected:
			try:
				df, tar, grp = read_number_raw(int(select)) if raw else read_number(int(select))
			except OSError as e:
				warnings.warn("Dataset "+names[int(select)]+" was not found")
				continue
			try:
				es_df = read_es(int(select))
			except OSError as e:
				warnings.warn("Enrichment score for dataset "+names[int(select)]+" was not found")
				es_df = pd.DataFrame({'A': [0]})
			es.append(es_df)
			dfs.append(df)
			target.extend(tar)
			group.extend(grp)
		df = df_utils.merge_frames(dfs)
		es = pd.concat(es, axis=0)
		lengths = [d.values.shape[0] for d in dfs]
	else:
		df, target, group = read_number_raw(int(selected[0])) if raw else read_number(int(selected[0]))
		es = read_es(int(selected[0]))
		lengths = [df.values.shape[0]]

	return df, target, group, lengths, es
