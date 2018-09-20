'''
Vegard Bjørgan 2018

reader for raw data files
'''

import pandas as pd
import numpy as np
from os import getcwd
import re

def read_hepmark_microarray():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Microarray/Hepmark-Microarray.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	df2 = read_hepmark_microarray_sampleSheet()
	df = df.dropna()
	return df, df2.loc[:, 'Type'], df2.loc[:, 'Code']


def read_hepmark_microarray_sampleSheet():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Microarray/SampleSheet-Hepmark-Microarray.txt"
	df = pd.read_csv(path, sep="\t")
	return df.loc[:, ['Type', 'Code']]


def read_hepmark_tissue():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	# Setup tumor/ normal df
	# TODO: Samples not used
	#samples = read_hepmark_tissue_samplenames()
	d = {'Type': []}
	for value in df.axes[0].values:
		if "n" in value: # TODO: Bad practice
			d['Type'] += ['Normal']
		else:
			d['Type'] += ['Tumor']
	targets = pd.DataFrame(data=d, index=df.axes[0].values)
	return df, targets


def read_hepmark_tissue_samplenames():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	df = pd.read_csv(path, sep="\t", usecols=['Normal', 'Tumor'])
	df = df.dropna()
	return df


def read_hepmark_paired_tissue():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Paired-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	d = {'Type': []}
	# TODO update
	for value in df.axes[0].values:
		if "N" in value:
			d['Type'] += ['Normal']
		elif "T" in value:
			d['Type'] += ['Tumor']
		else:
			d['Type'] += ['Undefined']
	targets = pd.DataFrame(data=d, index=df.axes[0].values)

	return df, targets

def read_hepmark_paired_tissue_samplesheet():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Paired-Tissue/SampleSheetPairedSamples-8Mar2017.txt"
	df = pd.read_csv(path, sep="\t", usecols=['ID', 'Normal', 'Tumor'])

	return df

'''
ColonCancer
'''

def read_guihuaSun_PMID_26646696():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
	analyses = path + "analyses/MatureMatrix.csv"
	raw = path + "raw/SampleSheet.txt"
	df = pd.read_csv(analyses, sep="\t").transpose()
	sampleSheet = pd.read_csv(raw, sep="\t")
	print(sampleSheet)
	print(df)
	# TODO

	df.dropna()

#read_guihuaSun_PMID_26646696()
#df = read_hepmark_microarray_sampleSheet()
#print(df)
