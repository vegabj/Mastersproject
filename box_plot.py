"""
Vegard Bjørgan 2019

Analyze one or more data sets and the effects of scaling miRNAs in a box plot
"""

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
df[features] = MiRNAScaler.minmax_scaler(df)
#df[features] = MiRNAScaler.quantile_scaler(df)
#df[features] = MiRNAScaler.individual_scaler(df.values)


data = []
row = []
for sample in df.axes[0]:
    data.append(df.loc[sample])
    row.append(sample)

# Extract 10 miRNA og show them in a box plot
last = 0
for i in range(1, len(data) // 10):
    d = data[last*10:i*10]
    legend = row[last*10:i*10]
    last = i
    fig, ax = plt.subplots()
    ax.set_title('MiRNA boxplots')
    ax.boxplot(d, labels=legend)

    plt.show()
    break
