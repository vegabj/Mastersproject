from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import numpy as np


class MiRNAScaler():

    def __init__():
        pass

    def miRNA_scaler(x):
        return MinMaxScaler().fit_transform(x)

    def standard_scaler(x):
        return StandardScaler().fit_transform(x)

    def group_scaler(df, features):
        xs = []
        grouped = df.groupby('group')

        for groupname, group in grouped:
            xs.append(
                StandardScaler().fit_transform(
                    group.loc[:, features]))

        return np.concatenate((xs), axis=0)

    def set_scaler(df, lengths):
        dfs = []
        dfs.append(df.head(lengths[0]))
        current = lengths[0]
        for i in range(1, len(lengths)):
            dfs.append(df.tail(len(df)-current).head(lengths[i]))
            current += lengths[i]
        dfs = [StandardScaler().fit_transform(d) for d in dfs]
        return np.concatenate((dfs), axis=0)

    def robust_scaler(x):
        return RobustScaler().fit_transform(x)

    def quantile_scaler(x, n=1000):
        return QuantileTransformer(n_quantiles=n).fit_transform(x)
