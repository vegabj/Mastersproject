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
    df, tar, grp, lengths = data_reader.read_main(raw=True)

    # Log transform keeping all columns as they will be used in the gsea.
    df = df.T
    df = df_utils.transform_sequence_to_microarray(df, all=True)

    sample = df.T

    ss = gseapy.ssgsea(data=sample
                    , gene_sets='Out/gmt_colon.gmt'
                    , no_plot=True
                    , outdir='Out/gsea_colon')
    # "When you run the gene set enrichment analysis, the GSEA software automatically normalizes
    # the enrichment scores for variation in gene set size, as described in GSEA Statistics.
    # Nevertheless, the normalization is not very accurate for extremely small or extremely
    # large gene sets. For example, for gene sets with fewer than 10 genes, just 2 or 3 genes
    # can generate significant results. Therefore, by default, GSEA ignores gene sets that
    # contain fewer than 25 genes or more than 500 genes; defaults that are appropriate for
    # datasets with 10,000 to 20,000 features. To change these default values, use the Max Size
    # and Min Size parameters on the Run GSEA Page; however, keep in mind the possibility of
    # inflated scorings for very small gene sets and inaccurate normalization for large ones."

    # Setup df file
    rows = []
    for s in ss.resultsOnSamples:
        row = [s]
        for val in ss.resultsOnSamples[s]:
            row.append(val)
        rows.append(row)
    columns = ['index']
    columns.extend([x for x in ss.resultsOnSamples[s].axes[0]])

    # Create es file
    df_out = pd.DataFrame(rows, columns = columns)
    df_out.set_index('index', inplace=True)
    path = r'%s' % getcwd().replace('\\','/') + "/Out/enrichment_scores/"
    df_out.to_csv(path+"es_PMID_26436952.csv")


if __name__ == "__main__":
    main()
