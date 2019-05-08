"""
Vegard Bjørgan 2019

utils for scaling data sets
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import numpy as np

def choose_scaling(df, lengths):
    scalers = [MinMaxScaler(feature_range=(-1,1)), RobustScaler(), StandardScaler()]
    scalers_names = ["Unscaled", "MinMax", "Robust", "Standard"]
    print("Select a scaler:")
    for i,s in enumerate(scalers_names):
        print(i,s)
    selected_scaler = int(input("Select: "))
    if selected_scaler:
        return set_scaler(df, lengths, scaler=scalers[selected_scaler-1])
    else:
        return df.values

def set_scaler(df, lengths, scaler=MinMaxScaler(feature_range=(-1,1))):
    dfs = []
    dfs.append(df.head(lengths[0]))
    current = lengths[0]
    for i in range(1, len(lengths)):
        dfs.append(df.tail(len(df)-current).head(lengths[i]))
        current += lengths[i]
    dfs = [scaler.fit_transform(d.values) for d in dfs]
    return np.concatenate((dfs), axis=0)

def set_scales(df, lengths, scaler=MinMaxScaler(feature_range=(-1,1))):
    dfs = []
    dfs.append(df.head(lengths[0]))
    current = lengths[0]
    for i in range(1, len(lengths)):
        dfs.append(df.tail(len(df)-current).head(lengths[i]))
        current += lengths[i]
    # Gather means and std for each df
    scales = [scaler.fit(d.values) for d in dfs]
    values = [[df.mean(axis=0), df.std(axis=0)] for df in dfs]
    return scales, values

def minmax_scaler(x):
    return MinMaxScaler((-1,1)).fit_transform(x)

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

def robust_scaler(x):
    return RobustScaler().fit_transform(x)

def quantile_scaler(x, n=1000):
    return QuantileTransformer(n_quantiles=n).fit_transform(x)

def individual_scaler(X):
    X = X.T
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = np.array(X_scaled)
    return X_scaled.T
