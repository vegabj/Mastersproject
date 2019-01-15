'''
Vegard Bjørgan 2019

gseapy tester for miRNAs
'''

import pandas as pd
import data_reader
import gseapy

def main():
    # Import data
    df, tar, grp = data_reader.read_number(0)

    sample = df.head(10).T
    #print(sample)
    #print(sample.index)
    print(tar[:10])

    ss = gseapy.ssgsea(data=sample
                    , gene_sets='new_gsea_test/new_gmt.gmt'
                    , outdir='new_gsea_test')
    print(ss)


if __name__ == "__main__":
    main()
