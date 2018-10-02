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
    return df


def test_merge_frames():
    df1, _, _ = data_reader.read_hepmark_microarray()
    df2, _, _ = data_reader.read_hepmark_tissue_formatted()
    merge_frames(df1,df2)

#test_merge_frames()
