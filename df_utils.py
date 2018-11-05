import pandas as pd
import numpy as np
import data_reader

def merge_frames(dfs):
    #print(df1)
    df = pd.concat(dfs, axis = 0, sort=False)
    #df = df.dropna(axis='columns')
    df = df.fillna(-1)
    #print(df)
    #import density_plot
    #density_plot.make_full_density_plot(df.transpose())
    #if True: # remove low expressed essential miRNAs
    #   keep_columns = [ax for ax in df.axes[1] if not df[ax].mean() > 50]
    #   df = df.loc[:, keep_columns]
    return df


# Tester
def test_merge_frames():
    df1, _, _ = data_reader.read_hepmark_microarray()
    df2, _, _ = data_reader.read_hepmark_tissue_formatted()
    df3, _, _ = data_reader.read_hepmark_paired_tissue_formatted()
    df = merge_frames([df1,df2])

### TODO: Clean up

def fetch_df_samples(df_val, pos, neg):
    df_val_pos = df_val.loc[df_val["target"] == "Normal"].sample(n=neg)
    df_val_neg = df_val.loc[df_val["target"] == "Tumor"].sample(n=pos)

    # Allign features
    df_val = pd.concat([df_val_pos, df_val_neg])
    tar_val = df_val.loc[:, "target"]
    df_val = df_val.drop("target", axis = 1)
    return df_val, tar_val

def fetch_df_samples2(df_val, pos, neg):
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
