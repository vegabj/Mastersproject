import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_reader
import scaler

# Import data
df, tar, grp = data_reader.read_hepmark_microarray()


# Scale data
features = df.axes[1].values
#df[features] = scaler.MiRNAScaler.standard_scaler(df)
#df[features] = scaler.MiRNAScaler.robust_scaler(df)
df[features] = scaler.MiRNAScaler.miRNA_scaler(df)
#df[features] = scaler.MiRNAScaler.quantile_scaler(df)


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
