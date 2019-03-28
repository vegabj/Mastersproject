'''
Vegard Bjørgan 2018

density_plot.py makes simple density plot of row data
'''

import seaborn as sns
from matplotlib import pyplot as plt

# Make default density plot
def make_density_plot(df):
    for i in range((len(df.axes[1]) // 10) + 1):
        samples = df.axes[1][i*10:(i*10)+10]

        for sample in samples:
            subset = df.loc[:, [sample]]
            sns.distplot(subset, hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 3},
                label = sample)

        plt.legend(prop={'size': 16}, title = 'Samples')
        plt.title('Density Plot with Multiple Samples')
        plt.xlabel('Expression Microarray')
        plt.ylabel('Density')
        plt.show()

def make_full_density_plot(df):

    for sample in df.axes[1]:
        subset = df.loc[:, [sample]]
        sns.distplot(subset, hist = False, kde = True,
            kde_kws = {'shade': False, 'linewidth': 1})

    #plt.legend(prop={'size': 16}, title = 'Samples')
    #plt.tick_params(axis='both', which='major', labelsize=25)

    # control x and y limits
    plt.ylim(0, 0.23)
    plt.xlim(-5, 20)

    plt.title('Density Plot of Hepmark Microarray')
    plt.xlabel('Expression Microarray')
    plt.ylabel('Density')
    plt.savefig("testDens.pdf", bbox_inches='tight')
    #plt.show()



def test_make_density_plot():
    import data_reader
    df1, target, group = data_reader.read_hepmark_microarray()
    #df2, _, _ = data_reader.read_hepmark_tissue_formatted()
    #df3, _, _ = data_reader.read_hepmark_paired_tissue_formatted()
    #df1, _, _ = data_reader.read_coloncancer_GCF_2014_295_formatted()
    #df2, _, _ = data_reader.read_guihuaSun_PMID_26646696_colon()
    #df3, _, _ = data_reader.read_publicCRC_GSE46622_colon()
    #df4, _, _ = data_reader.read_publicCRC_PMID_23824282_colon()
    #df5, _, _ = data_reader.read_publicCRC_PMID_26436952_colon()
    #dfs = [df1,df2,df3] #, df4, df5]
    #import pca_utils
    #for df in [df1, df2, df3]:
    #    keep_columns = [ax for ax in df.axes[1] if not df[ax].mean() > 50]
    #    df = df.loc[:, keep_columns]
    #    df = pca_utils.transform_sequence_to_microarray(df)
    #    dfs.append(df.transpose())
    import df_utils
    #df1 = df_utils.transform_sequence_to_microarray(df1, all=True)
    #df = df_utils.merge_frames(dfs)
    df = df1
    df = df.transpose()
    #make_density_plot(df)
    from utils import latexify
    latexify(columns=1)
    make_full_density_plot(df)

test_make_density_plot()
