from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import numpy as np

# TODO: a class is not needed
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
        dfs = [StandardScaler().fit_transform(d.values) for d in dfs]
        return np.concatenate((dfs), axis=0)

    def set_scales(df, lengths):
        dfs = []
        dfs.append(df.head(lengths[0]))
        current = lengths[0]
        for i in range(1, len(lengths)):
            dfs.append(df.tail(len(df)-current).head(lengths[i]))
            current += lengths[i]
        # Gather means and std for each df
        scales = [StandardScaler().fit(d.values) for d in dfs]
        values = [[df.mean(axis=0), df.std(axis=0)] for df in dfs]
        return scales, values

    def robust_scaler(x):
        return RobustScaler().fit_transform(x)

    def quantile_scaler(x, n=1000):
        return QuantileTransformer(n_quantiles=n).fit_transform(x)

    def generate_scale(df, lengths, biases):
        pass

    def individual_scaler(X, mean=1.0):
        #print(X[0], sum(X[0]))
        X = X.T
        X_scaled = StandardScaler().fit_transform(X)
        #X_scaled = X_scaled * mean
        #print(X_scaled.T[0], sum(X_scaled.T[0]))
        return X_scaled.T
