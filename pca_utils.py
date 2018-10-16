'''
Vegard Bjørgan 2018

pca_utils.py contains utilities for pca.py
'''
import data_reader
import numpy as np
import pandas as pd
from os import getcwd

def transform_sequence_to_microarray(df):
    df = df.transpose()
    # Solve reads per million for each sample
    for ax in df.axes[1]:
        df[ax] = df[ax] + 0.5  # Avoid -inf for log2(0)
        col_sum = sum(df[ax])
        df[ax] = (df[ax] * 1000000) / col_sum
        df[ax] = np.log2(df[ax])
    # Removes microRNAs where mean is not significant
    keep_columns = [ax for ax in df.axes[0] if df.loc[ax].mean() > 1.0]
    df = df.loc[keep_columns]

    return df

# Tests

def transform_sequence_to_microarray_test_hepmark_paired_tissue():
    df, _, _ = data_reader.read_hepmark_paired_tissue()
    df = transform_sequence_to_microarray(df)
    path = r'%s' % getcwd().replace('\\','/') + "/Data/Hepmark-Paired-Tissue/MatureMatrixFormatted.csv"
    df.to_csv(path)

def transform_sequence_to_microarray_test_hepmark_tissue():
    df, _, _ = data_reader.read_hepmark_tissue()
    # Remove extremes for Tissue
    df = df.drop(['ta157', 'tb140'])
    df = transform_sequence_to_microarray(df)
    path = r'%s' % getcwd().replace('\\','/') + "/Data/Hepmark-Tissue/MatureMatrixFormatted.csv"
    df.to_csv(path)

def transform_sequence_to_microarray_test_coloncancer_GCF_2014_295():
    df, _, _ = data_reader.read_coloncancer_GCF_2014_295()
    df = transform_sequence_to_microarray(df)
    path = r'%s' % getcwd().replace('\\','/') + "/Data/ColonCancer/ColonCancer_GCF-2014-295/analyses/MatureMatrixFormatted.csv"
    df.to_csv(path)

def transform_sequence_to_microarray_test():
    # Run 4 datasets through
    path = r'%s' % getcwd().replace('\\','/')
    path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
    analyses = path + "analyses/MatureMatrix.csv"
    df = pd.read_csv(analyses, sep="\t").transpose()
    df = transform_sequence_to_microarray(df)
    to = path + "analyses/MatureMatrixFormatted.csv"
    df.to_csv(to)

#transform_sequence_to_microarray_test_hepmark_paired_tissue()
#transform_sequence_to_microarray_test_hepmark_tissue()
#transform_sequence_to_microarray_test_coloncancer_GCF_2014_295()
transform_sequence_to_microarray_test()
