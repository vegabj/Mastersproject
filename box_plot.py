import numpy as np
import matplotlib.pyplot as plt

#import data
import data_reader
df, tar, grp = data_reader.read_hepmark_microarray()

#import scaler
#miRNA = df.axes[1]
#X = scaler.MiRNAScaler.standard_scaler(df)
# TODO

data = []
miRNAs = []
for miRNA in df.axes[1]:
    data.append(df[miRNA].values)
    miRNAs.append(miRNA)

last = 0
for i in range(1, len(data) // 5):
    d = data[last*5:i*5]
    legend = miRNAs[last*5:i*5]
    last = i
    fig, ax = plt.subplots()
    ax.set_title('MiRNA boxplots')
    ax.boxplot(d)
    ax.legend(legend)

    plt.show()
