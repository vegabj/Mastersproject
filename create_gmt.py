# Vegard B 2019

# A gmt (Gene Matrix Transposed) is a file used by gsea to evaluate sequences and score
# Column 1 Name of Test
# Column 2 Test Description
# Column 3-n miRNAs to check

import pandas as pd
from os import getcwd
import data_reader
import numpy as np

def create_gmt_file(lst, path):
    outframe = pd.DataFrame(lst)
    outframe.to_csv(path + "new_gmt.gmt", sep='\t', header=False, index=False)

def extract_mirnas(df):
    lst = []
    tumor_df = df.loc[df.target == 'Tumor']
    tumor_df = tumor_df.drop('target', axis=1)
    tumor_formatted_df = pd.DataFrame([tumor_df.mean().sort_values(ascending=False),
                        tumor_df.std().sort_values(ascending=False)])
    tumor_formatted_df = tumor_formatted_df.transpose()
    tumor_formatted_df['sum'] = tumor_formatted_df[0] - tumor_formatted_df[1]
    tumor_res = tumor_formatted_df['sum'].sort_values(ascending=False)[:30]
    tumor = ["Tumor", ""]
    tumor.extend(tumor_res.index.values)
    lst.append(tumor)
    normal_df = df.loc[df.target == 'Normal']
    normal_df = normal_df.drop('target', axis=1)
    normal_formatted_df = pd.DataFrame([normal_df.mean().sort_values(ascending=False),
                        normal_df.std().sort_values(ascending=False)])
    normal_formatted_df = normal_formatted_df.transpose()
    normal_formatted_df['sum'] = normal_formatted_df[0] - normal_formatted_df[1]
    normal_res = normal_formatted_df['sum'].sort_values(ascending=False)[:30]
    normal = ["Normal", ""]
    normal.extend(normal_res.index.values)
    lst.append(normal)
    return lst

def extract_mirnas_r(df, ss):
    # Import necessary modules
    from rpy2.robjects.packages import importr
    limma = importr('limma')
    edger = importr('edgeR')
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate()
    # Convert pandas df to r data frame
    r_matrix = pandas2ri.py2ri(df)
    r_samplesheet = pandas2ri.py2ri(ss)

    # Setup R function
    r('''
        # create a function `f`
        f <- function(matrix, sampleSheet) {
            groups <- factor(sampleSheet$groups)
            block <- na.omit(factor(sampleSheet$block))
            design <- model.matrix(~ groups)

            count.mat <- DGEList(matrix)
            count.voom <- voom(count.mat)

            dup.cor <- duplicateCorrelation(count.voom, design=design, block=block)
            fit <- lmFit(count.voom, design=design, block=block, correlation=dup.cor$consensus.correlation)

            fit <- eBayes(fit)
            topTab <- topTable(fit, coef=2, p.value=1, number=Inf, sort.by="logFC")
            #return(topTab)
            print(topTab)
        }
        ''')

    r_f = r['f']
    # Run R function
    r_f(r_matrix, r_samplesheet)
    print("Success")

"""
print("Testing extract")
df, tar, grp = data_reader.read_number(0)
df['target'] = tar
lst = extract_mirnas(df)
path = r'%s' % getcwd().replace('\\','/') + "/Out/"
create_gmt_file(lst, path)
"""

print("Testing extract_mirnas_r")
df, tar, grp = data_reader.raw()
ss = pd.DataFrame([tar, grp])
ss = ss.rename({ss.axes[0][0]: 'groups', ss.axes[0][1]: 'block'}, axis='index').transpose()
df = df.transpose()
lst = extract_mirnas_r(df, ss)
path = r'%s' % getcwd().replace('\\','/') + "/Out/"
create_gmt_file(lst, path)
