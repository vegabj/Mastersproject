"""
Vegard Bjørgan 2019

Create a gmt file or gene set to an existing gmt file

# A gmt (Gene Matrix Transposed) is a file used by gsea to evaluate sequences and score
# Column 1 Name of Test
# Column 2 Test Description
# Column 3-n miRNAs in the gene set
"""

import pandas as pd
from os import getcwd
import data_reader
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
# Import necessary modules in R
limma = importr('limma')
edger = importr('edgeR')
# Activates automatic conversion between python dataframes and R dataframes
pandas2ri.activate()

# Creates the gmt file
def create_gmt_file(lst, path, new=False):
    outframe = pd.DataFrame(lst)
    if new:
        outframe.to_csv(path + "new_gmt.gmt", sep='\t', header=False, index=False)
    else:
        with open(path + "new_gmt.gmt", 'a') as f:
            outframe.to_csv(f, sep='\t', header=False, index=False)

# Extracts the miRNAs from a dataframe that are higher expressed in both tumor and normal samples
def extract_mirnas_r(df, ss, gs_name=""):
    # Setup R function
    r('''
        # create a function `f`
        f <- function(matrix, sampleSheet) {
            groups <- factor(sampleSheet$groups)
            block <- na.omit(factor(sampleSheet$block))
            design <- model.matrix(~ groups)

            count.mat <- DGEList(matrix)
            # table of counts (rows=features, columns=samples), group indicator for each column
            count.voom <- voom(count.mat)
            # Transform RNA-Seq Data Ready For Linear Modelling

            # Handling for microarray data - Uncomment when using microarray data
            #count.voom = matrix


            dup.cor <- duplicateCorrelation(count.voom, design=design, block=block)
            # Estimate the correlation between duplicate spots (regularly spaced replicate
            # spots on the same array) or between technical replicates from a series of arrays.
            fit <- lmFit(count.voom, design=design, block=block, correlation=dup.cor$consensus.correlation)
            # Fit linear model for each gene given a series of arrays
            fit <- eBayes(fit)
            # Given a microarray linear model fit, compute moderated t-statistics,
            # moderated F-statistic, and log-odds of differential expression by
            # empirical Bayes moderation of the standard errors towards a common value.
            topTab <- topTable(fit, coef=2, p.value=0.05, number="inf", sort.by="p")
            # Extract a table of the top-ranked genes from a linear model fit.

            return(topTab)
        }
        ''')

    # Setup function binding
    r_f = r['f']
    # Run R function
    res = r_f(df, ss)
    # Note: Conversion loses miRNAs as index, so this is manually set.
    pd_res = pandas2ri.ri2py(res)
    index = [i for i in res.rownames]
    pd_res['index'] = index
    pd_res.set_index('index', inplace=True)

    # Extract abs(logFC) > 1, negative logFC as Normal and positive as Tumor
    n_res = pd_res.loc[pd_res['logFC'] < -1.0]
    t_res = pd_res.loc[pd_res['logFC'] > 1.0]
    lst_t, lst_n = ["Tumor"+gs_name, ""], ["Normal"+gs_name, ""]
    lst_t.extend(t_res.index.values)
    lst_n.extend(n_res.index.values)
    return [lst_t, lst_n]


# Same as extract_mirnas_r but made specifically for data set GuihuaSun as it needs extra
# information by using paired samples to find distinct separations between normal and tumor
def extract_mirnas_r_guihuasun(df, ss, gs_name=""):
    # Setup R functions
    r('''
        CreateDesign <- function(expressionMatrix, sampleSheet) {
            sampleSheet[sampleSheet == ""] <- NA
            sampleType <- sampleSheet$Diease
            groups <- factor(sampleType)
            site <- factor(sampleSheet$Tissue)
            groupSite <- factor(paste0(groups, site))
            stage <- factor(sampleSheet$Stage)
            age <- sampleSheet$Age
            gender <- factor(sampleSheet$Gender)
            race <- factor(sampleSheet$Race)
            block <- factor(sub("-.", "", sampleSheet$ID))
            design <- model.matrix(~ 0 + groupSite + stage + age + gender)
            colnames(design) <- sub("^site", "", sub("^groupSite", "", colnames(design)))
            colnames(design) <- sub("^site", "", sub("^groups", "", colnames(design)))
            return(list(design=design, block=block[as.numeric(rownames(design))],
                contrasts=makeContrasts(TvN=0.5*(TumorColon+TumorRectal)-0.5*(NormalColon+NormalRectal), ColVsRect_Norm=NormalColon-NormalRectal, ColVsRect_Tum=TumorColon-TumorRectal,
                ColIntRect=(TumorColon-TumorRectal)-(NormalColon-NormalRectal), Stage3_4vsRest=0.5*(stage4+stage3)-0.5*(stage2+stage1), MvsF=genderMale, Stage3vs1=stage3-stage1, levels=design), groups=groups[as.numeric(rownames(design))]))
            return(list(design=design, block=block, contrasts=makeContrasts(TvN=Tumor-Normal, ColVsRect=Rectal, levels=design), groups=groups))
            }''')
    r_create_design = r['CreateDesign']
    r('''
        CreateSampleSheet <- function(expressionMatrix) {
            sampleSheet <- read.table("Data/ColonCancer/GuihuaSun-PMID_26646696/raw/SampleSheet.txt", header=TRUE, sep="\t")
            ret <- sampleSheet[,c("ID", "Diease", "Tissue", "Stage", "Age", "Gender", "Race", "File")]
            rownames(ret) <- ret$File
            return(ret[sub("^X", "", colnames(expressionMatrix)),])
            }
    ''')
    r_create_samplesheet = r['CreateSampleSheet']
    r('''
        # create a function `f`
        f <- function(matrix, sampleSheet, designInfo) {
            groups <- factor(sampleSheet$groups)
            block <- na.omit(factor(sampleSheet$block))
            design <- designInfo$design

            count.mat <- DGEList(matrix)
            count.voom <- voom(count.mat)

            dup.cor <- duplicateCorrelation(count.voom, design=design, block=block)
            fit <- lmFit(count.voom, design=design, block=block, correlation=dup.cor$consensus.correlation)
            fit <- contrasts.fit(fit, designInfo$contrasts)
            fit <- eBayes(fit)
            topTab <- topTable(fit, coef="TvN", p.value=0.05, number="inf", sort.by="p")
            return(topTab)
        }
        ''')
    # Make function bindings between R and python
    r_f = r['f']

    # run R functions
    sampleSheet = r_create_samplesheet(df)
    designInfo = r_create_design(df, sampleSheet)
    res = r_f(df, ss, designInfo)

    pd_res = pandas2ri.ri2py(res)
    index = [i for i in res.rownames]
    pd_res['index'] = index
    pd_res.set_index('index', inplace=True)

    # Extract abs(logFC) > 1, negative logFC as Normal and positive as Tumor
    n_res = pd_res.loc[pd_res['logFC'] < -1.0]
    t_res = pd_res.loc[pd_res['logFC'] > 1.0]
    lst_t, lst_n = ["Tumor"+gs_name, ""], ["Normal"+gs_name, ""]
    lst_t.extend(t_res.index.values)
    lst_n.extend(n_res.index.values)
    return [lst_t, lst_n]


def main():
    # Reads in a data set
    df, tar, grp, _, _ = data_reader.read_main(raw=True)

    # Code for DS 4 (GuihuaSun)
    """
    missing = ['110608_TGCTCG_s_5', '110608_TGCTCG_s_1', '110608_TCGTCG_s_6', '110608_TCGTCG_s_5', '110602_TGCTCG_s_4', '110602_TGCTCG_s_3', '110602_TCGTCG_s_4', '110602_TCGTCG_s_3']
    df = df.drop(missing)
    tar = tar.drop(missing)
    grp = grp.drop(missing)
    #lst = extract_mirnas_r_guihuasun(df, ss, gs_name="_4")
    """

    # Running extract_mirnas_r
    ss = pd.DataFrame([tar, grp])
    ss = ss.rename({ss.axes[0][0]: 'groups', ss.axes[0][1]: 'block'}, axis='index').transpose()
    df = df.transpose()
    lst = extract_mirnas_r(df, ss, gs_name="_3")

    # Creating gmt file
    path = r'%s' % getcwd().replace('\\','/') + "/Out/"
    create_gmt_file(lst, path, new=True)

if __name__ == "__main__":
    main()
