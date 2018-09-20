'''
Vegard Bjørgan 2018

density_plot.py makes simple density plot of row data
'''

import seaborn as sns
from matplotlib import pyplot as plt

# Make default density plot
def make_density_plot(df):
    for i in range(15):
        samples = df.axes[1][i*10:(i*10)+10]

        for sample in samples:
            subset = df.loc[:, [sample]]
            sns.distplot(subset, hist = False, kde = True,
            kde_kws = {'linewidth': 3},
            label = sample)

        plt.legend(prop={'size': 16}, title = 'Samples')
        plt.title('Density Plot with Multiple Samples')
        plt.xlabel('Expression Microarray')
        plt.ylabel('Density')
        print(i, i+10)
        plt.show()



def test_make_density_plot():
    import data_reader
    df, target = data_reader.read_hepmark_microarray()
    df = df.transpose()
    make_density_plot(df)

test_make_density_plot()
