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

    sample = df.head().T
    #print(sample)
    #print(sample.index)

    ss = gseapy.ssgsea(data=sample
                    , gene_sets='gsea_test/my_gmt.gmt'
                    , outdir='gsea_test')


if __name__ == "__main__":
    main()
