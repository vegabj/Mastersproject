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
    # Solve reads per million
    for ax in df.axes[1]:
        df[ax] = df[ax] + 0.5  # Avoid -inf for log2(0)
        col_sum = sum(df[ax])
        df[ax] = (df[ax] * 1000000) / col_sum
        df[ax] = np.log2(df[ax])

    return df

def transform_sequence_to_microarray_test_hepmark_paired_tissue():
    df, _, _ = data_reader.read_hepmark_paired_tissue()
    # Remove not significant columns
    keep_columns = [ax for ax in df.axes[1] if any(df[ax] > 100)]
    df = df.loc[:, keep_columns]
    df = transform_sequence_to_microarray(df)
    path = r'%s' % getcwd().replace('\\','/') + "/Data/Hepmark-Paired-Tissue/MatureMatrixFormatted.csv"
    df.to_csv(path)

def transform_sequence_to_microarray_test_hepmark_tissue():
    df, _, _ = data_reader.read_hepmark_tissue()
    # Remove not significant columns
    keep_columns = [ax for ax in df.axes[1] if any(df[ax] > 100)]
    df = df.loc[:, keep_columns]
    # Remove extremes for Tissue
    df = df.drop(['ta157', 'tb140'])
    df = transform_sequence_to_microarray(df)
    path = r'%s' % getcwd().replace('\\','/') + "/Data/Hepmark-Tissue/MatureMatrixFormatted.csv"
    df.to_csv(path)


#transform_sequence_to_microarray_test_hepmark_tissue()
