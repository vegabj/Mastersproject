"""
Vegard Bjørgan 2019

density_plot.py makes simple density plot of row data
"""

import seaborn as sns
from matplotlib import pyplot as plt
import data_reader
import scaler as MiRNAScaler
from utils import latexify
import pandas as pd

# Make density plot with 10 samples at a time
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

# Make a density plot of all samples
def make_full_density_plot(df, title):

    for sample in df.axes[1]:
        subset = df.loc[:, [sample]]
        sns.distplot(subset, hist = False, kde = True,
            kde_kws = {'shade': False, 'linewidth': 1})

    #plt.legend(prop={'size': 16}, title = 'Samples')
    #plt.tick_params(axis='both', which='major', labelsize=25)

    # control x and y limits
    #plt.ylim(0, 0.23)
    #plt.xlim(-5, 20)

    plt.title(title)
    #plt.xlabel('Microarray Expression')
    plt.xlabel('Normalized Expression')
    plt.ylabel('Density')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()



def test_make_density_plot():
    df, _, _, length, _ = data_reader.read_main(raw=False)
    features = df.axes[1]
    samples = df.axes[0]
    df = MiRNAScaler.choose_scaling(df, length)
    df = pd.DataFrame(df, index=samples, columns=features)
    df = df.transpose()
    #make_density_plot(df)
    latexify(columns=2)
    make_full_density_plot(df, 'Density Plot of Hepmark Tissue')

test_make_density_plot()
