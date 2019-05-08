"""
Vegard Bjørgan 2019

df_utils.py contains utilities for data frames
"""

import pandas as pd
import numpy as np
import data_reader
from os import getcwd

def merge_frames(dfs, drop=True):
    df = pd.concat(dfs, axis = 0, sort=False)
    if drop:
        df = df.dropna(axis=1)
    else:
        df = df.fillna(-1)
    return df


# Tester for merge_frames
def test_merge_frames():
    df1, _, _ = data_reader.read_hepmark_microarray()
    df2, _, _ = data_reader.read_hepmark_tissue_formatted()
    df3, _, _ = data_reader.read_hepmark_paired_tissue_formatted()
    df = merge_frames([df1,df2])


def fetch_df_samples(df_val, pos, neg):
    if pos == 'all':
        df_val_pos = df_val.loc[df_val["target"] == 1]
    else:
        if df_val.loc[df_val["target"] == 1].empty:
            df_val_pos = df_val.loc[df_val["target"] == 1]
        else:
            df_val_pos = df_val.loc[df_val["target"] == 1].sample(n=pos)
    if neg == 'all':
        df_val_neg = df_val.loc[df_val["target"] == 0]
    else:
        if df_val.loc[df_val["target"] == 0].empty:
            df_val_neg = df_val.loc[df_val["target"] == 0]
        else:
            df_val_neg = df_val.loc[df_val["target"] == 0].sample(n=neg)

    # Allign features
    df_val = pd.concat([df_val_pos, df_val_neg])
    tar_val = df_val.loc[:, "target"]
    df_val = df_val.drop("target", axis = 1)
    return df_val, tar_val

def check_df_samples(df_val, pos, neg):
    df_val_neg = df_val.loc[df_val["target"] == 0]
    df_val_pos = df_val.loc[df_val["target"] == 1]
    pos_length = len(df_val_pos)
    neg_length = len(df_val_neg)
    if pos == 'all':
        pos = pos_length
        if pos_length == 0:
            pos = 1
    if neg == 'all':
        neg = neg_length
        if neg_length == 0:
            neg = 1

    return pos <= pos_length and neg <= neg_length

def transform_sequence_to_microarray(df, all=False):
    df = df.transpose()
    # Solve reads per million for each sample
    for ax in df.axes[1]:
        df[ax] = df[ax] + 0.5  # Avoid -inf for log2(0)
        col_sum = sum(df[ax])
        df[ax] = (df[ax] * 1000000) / col_sum
        df[ax] = np.log2(df[ax])
    # Removes microRNAs where mean is not significant
    if not all:
        keep_columns = [ax for ax in df.axes[0] if df.loc[ax].mean() > 1.0]
        df = df.loc[keep_columns]
    return df

# Test for transforming sequencing data to microarray
def transform_sequence_to_microarray_test():
    path = r'%s' % getcwd().replace('\\','/')
    path = path + "/Data/ColonCancer/GuihuaSun-PMID_26646696/"
    analyses = path + "analyses/MatureMatrix.csv"
    df = pd.read_csv(analyses, sep="\t").transpose()
    df = transform_sequence_to_microarray(df)
    to = path + "analyses/MatureMatrixFormatted.csv"
    df.to_csv(to)

#transform_sequence_to_microarray_test()
