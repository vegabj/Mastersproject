# Vegard B 2019

# A gmt (Gene Matrix Transposed) is a file used by gsea to evaluate sequences and score
# Column 1 Name of Test
# Column 2 Test Description
# Column 3-n miRNAs to check

import pandas as pd
from os import getcwd
import data_reader
import numpy as np

def create_gmt_file(lst, path, new=False):
    outframe = pd.DataFrame(lst)
    if new:
        outframe.to_csv(path + "new_gmt.gmt", sep='\t', header=False, index=False)
    else:
        with open(path + "new_gmt.gmt", 'a') as f:
            outframe.to_csv(f, sep='\t', header=False, index=False)

def extract_mirnas_r(df, ss, gs_name=""):
    # Import necessary modules
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri
    limma = importr('limma')
    edger = importr('edgeR')
    # Activates automatic conversion between pandas dataframes and R data frames
    pandas2ri.activate()

    # Setup R function
    r('''
        # create a function `f`
        f <- function(matrix, sampleSheet) {
            groups <- factor(sampleSheet$groups)
            block <- na.omit(factor(sampleSheet$block))
            design <- model.matrix(~ groups)
            #print(block)
            #print(design)

            count.mat <- DGEList(matrix)
            # table of counts (rows=features, columns=samples), group indicator for each column
            count.voom <- voom(count.mat)
            # Transform RNA-Seq Data Ready For Linear Modelling

            # Handling for microarray data
            #count.voom = matrix


            dup.cor <- duplicateCorrelation(count.voom, design=design, block=block)
            # Estimate the correlation between duplicate spots (regularly spaced replicate
            # spots on the same array) or between technical replicates from a series of arrays.
            #print(dup.cor)
            fit <- lmFit(count.voom, design=design, block=block, correlation=dup.cor$consensus.correlation)
            # Fit linear model for each gene given a series of arrays
            fit <- eBayes(fit)
            # Given a microarray linear model fit, compute moderated t-statistics,
            # moderated F-statistic, and log-odds of differential expression by
            # empirical Bayes moderation of the standard errors towards a common value.
            topTab <- topTable(fit, coef=2, p.value=0.05, number="inf", sort.by="p")
            # Extract a table of the top-ranked genes from a linear model fit.

            #print(topTab)
            return(topTab)
        }
        ''')

    # Setup function binding
    r_f = r['f']
    # Run R function
    res = r_f(df, ss)
    # Note: Convertion loses miRNAs as index, manually set.
    pd_res = pandas2ri.ri2py(res)
    index = [i for i in res.rownames]
    pd_res['index'] = index
    pd_res.set_index('index', inplace=True)

    # Extract abs(logFC) > 1, negative logFC as Normal and positive as Tumor
    n_res = pd_res.loc[pd_res['logFC'] < -1.0] # -0.5 for ds 0,4,5
    t_res = pd_res.loc[pd_res['logFC'] > 1.0] # 0.5 for ds 0,4,5
    lst_t, lst_n = ["Tumor"+gs_name, ""], ["Normal"+gs_name, ""]
    # Code for ds 4
    #n_res = n_res.head(30)
    #t_res = t_res.head(30)
    lst_t.extend(t_res.index.values)
    lst_n.extend(n_res.index.values)
    return [lst_t, lst_n]


def main():
    # Running extract_mirnas_r
    df, tar, grp, _ = data_reader.read_main(raw=True)
    ss = pd.DataFrame([tar, grp])
    ss = ss.rename({ss.axes[0][0]: 'groups', ss.axes[0][1]: 'block'}, axis='index').transpose()
    df = df.transpose()
    lst = extract_mirnas_r(df, ss, gs_name="_7")

    # Creating gmt file
    path = r'%s' % getcwd().replace('\\','/') + "/Out/"
    create_gmt_file(lst, path, new=True)

if __name__ == "__main__":
    main()
