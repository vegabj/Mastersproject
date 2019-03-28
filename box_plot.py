import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_reader
import scaler as MiRNAScaler

# Import data
df, tar, grp, _, _ = data_reader.read_main()


# Scale data
features = df.axes[1].values
#df[features] = MiRNAScaler.standard_scaler(df)
#df[features] = MiRNAScaler.robust_scaler(df)
#df[features] = MiRNAScaler.miRNA_scaler(df)
#df[features] = MiRNAScaler.quantile_scaler(df)
df[features] = MiRNAScaler.individual_scaler(df.values)

'''
data = []
miRNAs = []
for miRNA in df.axes[1]:
    data.append(df[miRNA].values)
    miRNAs.append(miRNA)

last = 0
for i in range(1, len(data) // 10):
    d = data[last*10:i*10]
    legend = miRNAs[last*10:i*10]
    last = i
    fig, ax = plt.subplots()
    ax.set_title('MiRNA boxplots')
    ax.boxplot(d, labels=legend)
    #ax.legend(legend)

    plt.show()
    break
'''

data = []
row = []
for sample in df.axes[0]:
    data.append(df.loc[sample])
    row.append(sample)

last = 0
for i in range(1, len(data) // 10):
    d = data[last*10:i*10]
    legend = row[last*10:i*10]
    last = i
    fig, ax = plt.subplots()
    ax.set_title('MiRNA boxplots')
    ax.boxplot(d, labels=legend)
    #ax.legend(legend)

    plt.show()
    break
