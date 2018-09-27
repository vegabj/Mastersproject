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
            kde_kws = {'shade': False, 'linewidth': 3},
            label = sample)

    plt.legend(prop={'size': 16}, title = 'Samples')
    plt.title('Density Plot with Multiple Samples')
    plt.xlabel('Expression Microarray')
    plt.ylabel('Density')
    plt.show()



def test_make_density_plot():
    import data_reader
    #df, target, group = data_reader.read_hepmark_microarray()
    df, _, _ = data_reader.read_hepmark_tissue_formatted()
    #df, target, group = data_reader.read_guihuaSun_PMID_26646696()
    df = df.transpose()
    #make_density_plot(df)
    make_full_density_plot(df)

test_make_density_plot()
