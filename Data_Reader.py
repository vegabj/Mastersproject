import pandas as pd
import numpy as np
from os import getcwd
import re

def Read_Hepmark_Microarray():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Microarray/Hepmark-Microarray.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	df2 = Read_Hepmark_Microarray_SampleSheet()
	df = df.dropna()
	return df, df2.loc[:, 'Type']


def Read_Hepmark_Microarray_SampleSheet():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Microarray/SampleSheet-Hepmark-Microarray.txt"
	df = pd.read_csv(path, sep="\t")
	return df.loc[:, ['Type']]


def Read_Hepmark_Tissue():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	# Setup tumor/ normal df
	# TODO: Samples not used
	#samples = Read_Hepmark_Tissue_SampleNames()
	d = {'Type': []}
	for value in df.axes[0].values:
		if "n" in value: # TODO: Bad practice
			d['Type'] += ['Normal']
		else:
			d['Type'] += ['Tumor']
	targets = pd.DataFrame(data=d, index=df.axes[0].values)
	return df, targets


def Read_Hepmark_Tissue_SampleNames():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Tissue/SampleNamesHEP-28Mar2017.txt"
	df = pd.read_csv(path, sep="\t", usecols=['Normal', 'Tumor'])
	df = df.dropna()
	return df


def Read_Hepmark_Hepmark_Paired_Tissue():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Paired-Tissue/MatureMatrix.csv"
	df = pd.read_csv(path, sep="\t").transpose()
	d = {'Type': []}
	for value in df.axes[0].values:
		if "N" in value:
			d['Type'] += ['Normal']
		elif "T" in value:
			d['Type'] += ['Tumor']
		else:
			d['Type'] += ['Undefined']
	targets = pd.DataFrame(data=d, index=df.axes[0].values)

	return df, targets

def Read_Hepmark_Hepmark_Paired_Tissue_SampleSheet():
	path = r'%s' % getcwd().replace('\\','/')
	path = path + "/Data/Hepmark-Paired-Tissue/SampleSheetPairedSamples-8Mar2017.txt"
	df = pd.read_csv(path, sep="\t", usecols=['ID', 'Normal', 'Tumor'])

	return df
