'''
Vegard Bjørgan 2019

gseapy tester for miRNAs
'''

import pandas as pd
from os import getcwd
import data_reader
import gseapy
import df_utils

def main():
    # Import data
    df, tar, grp, lengths, _ = data_reader.read_main(raw=True)

    # Log transform keeping all columns as they will be used in the gsea.
    df = df_utils.transform_sequence_to_microarray(df.T, all=True)

    # Edge case microarray set 0 - not needed?
    """
    df1, _, _ = data_reader.read_number(1)
    df2, _, _ = data_reader.read_number(2)
    df_len = len(df)
    df = df_utils.merge_frames([df,df1,df2])
    df = df.head(df_len)
    """


    sample = df.T

    ss = gseapy.ssgsea(data=sample
                    , gene_sets='Out/new_gmt.gmt'
                    , no_plot=True
                    , outdir='Out/gsea_hepmark'
                    , min_size=10)
    # "When you run the gene set enrichment analysis, the GSEA software automatically normalizes
    # the enrichment scores for variation in gene set size, as described in GSEA Statistics.
    # Nevertheless, the normalization is not very accurate for extremely small or extremely
    # large gene sets. For example, for gene sets with fewer than 10 genes, just 2 or 3 genes
    # can generate significant results. Therefore, by default, GSEA ignores gene sets that
    # contain fewer than 25 genes or more than 500 genes; defaults that are appropriate for
    # datasets with 10,000 to 20,000 features. To change these default values, use the Max Size
    # and Min Size parameters on the Run GSEA Page; however, keep in mind the possibility of
    # inflated scorings for very small gene sets and inaccurate normalization for large ones."

    # PRED
    targets = [0 if t == "Normal" else 1 for t in tar]
    pred = []

    # Setup df file
    rows = []
    for s in ss.resultsOnSamples:
        row = [s]
        for val in ss.resultsOnSamples[s]:
            row.append(val)
        rows.append(row)

        # Add prediction
        if row[1] > row[2]:
            pred.append(0)
        else:
            pred.append(1)

    columns = ['index']
    columns.extend([x for x in ss.resultsOnSamples[s].axes[0]])

    # PRED
    points = 0
    type_1 = 0
    type_2 = 0

    for t, p in zip (targets, pred):
        if t == p:
            points+=1
        elif t == 1:
            type_1 += 1
        elif t == 0:
            type_2 += 1
    print("Score:", points/len(targets))
    print("Type 1 (Tumor but pred Normal):", type_1)
    print("Type 2 (Normal but pred Tumor):", type_2)

    # NB :  Do not use the res2d as this is the normalized score

    # Create es file
    df_out = pd.DataFrame(rows, columns = columns)
    df_out.set_index('index', inplace=True)
    path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/"
    df_out.to_csv(path+"es_test.csv")


if __name__ == "__main__":
    main()
