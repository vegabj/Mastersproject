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

def fetch_df_samples(df_val, pos, neg):
    df_val_pos = df_val.loc[df_val["target"] == "Normal"].sample(n=pos)
    df_val_neg = df_val.loc[df_val["target"] == "Tumor"].sample(n=neg)

    # Allign features
    df_val = pd.concat([df_val_pos, df_val_neg])
    tar_val = df_val.loc[:, "target"]
    df_val = df_val.drop("target", axis = 1)
    return df_val, tar_val
