import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def split(n_data, ratio=0.8, seed=0):
    """
    Split a dataset into training and test sets.
    """
    n_trn_data = int(n_data * ratio)
    index = np.arange(n_data)
    np.random.RandomState(seed).shuffle(index)
    trn_idx = index[:n_trn_data]
    test_idx = index[n_trn_data:]
    return trn_idx, test_idx


def normalize(data, trn_idx):
    """
    Normalize each dataset based on training index.
    """
    avg = data[trn_idx].mean(axis=0)
    std = data[trn_idx].std(axis=0)
    return (data - avg) / std


def to_numerical_features(df, trn_idx, columns=None):
    """
    Preprocess numerical features in a dataset.
    """
    columns = df.columns if columns is None else columns
    for col in columns:
        series = df[col]
        if series.dtype == 'float64':
            trn_series = series[trn_idx]
            avg = trn_series[trn_series.notna()].mean()
            df.loc[df[col].isna(), col] = avg
        elif series.dtype == 'object':
            trn_series = series[trn_idx]
            avg = trn_series[trn_series != '?'].astype(np.float32).mean()
            df.loc[series == '?', col] = avg
            df[col] = df[col].astype(np.float32)
    return normalize(df[columns].values.astype(np.float32), trn_idx)


def to_categorical_features(df, columns=None):
    """
    Preprocess categorical features as one-hot vectors.
    """
    def to_onehot(series):
        values = pd.unique(series)
        values_map = {v: i for i, v in enumerate(values)}
        n_data = series.shape[0]
        array = np.zeros((n_data, len(values)), dtype=np.float32)
        array[np.arange(n_data), series.map(values_map).values] = 1
        return array

    columns = df.columns if columns is None else columns
    return np.concatenate([to_onehot(df[c]) for c in columns], axis=1)


def to_labels(series):
    """
    Convert string labels into integer labels.
    """
    return series.map({v: i for i, v in enumerate(pd.unique(series))}).values


def preprocess(data, data_in='../data/raw', data_out='../data/preprocessed'):
    """
    Preprocess a dataset based on its name.
    """
    if data == 'breast-cancer':
        df = pd.read_csv('{}/{}/breast-cancer.data'.format(data_in, data), header=None)
        trn_idx, test_idx = split(df.shape[0])
        features = to_categorical_features(df.iloc[:, 1:])
        labels = to_labels(df[0])
    elif data == 'breast-cancer-wisconsin':
        df = pd.read_csv('{}/{}/breast-cancer-wisconsin.data'.format(data_in, data), header=None)
        trn_idx, test_idx = split(df.shape[0])
        features = to_numerical_features(df, trn_idx, list(range(1, 10)))
        labels = to_labels(df[10])
    elif data == 'heart-disease':
        df = pd.read_csv('{}/{}/processed.cleveland.data'.format(data_in, data), header=None)
        trn_idx, test_idx = split(df.shape[0])
        x1 = to_numerical_features(df, trn_idx, [0, 3, 4, 7, 9])
        x2 = to_categorical_features(df, [1, 2, 5, 6, 8, 10, 12])
        features = np.concatenate((x1, x2), axis=1)
        labels = to_labels((df[13] > 0).astype(np.int64))
    elif data == 'hepatitis':
        df = pd.read_csv('{}/{}/hepatitis.data'.format(data_in, data), header=None)
        trn_idx, test_idx = split(df.shape[0])
        x1 = to_numerical_features(df, trn_idx, [1, 14, 15, 16, 17, 18])
        x2 = to_categorical_features(df, [2, 4, 5])
        features = np.concatenate((x1, x2), axis=1)
        labels = to_labels(df[19])
    elif data == 'brain-tumor':
        df = pd.read_csv('{}/{}/Dataset.csv'.format(data_in, data))
        df = df[df['Area'] != 0].reset_index()
        trn_idx, test_idx = split(df.shape[0])
        cols = ['Area', 'Perimeter', 'Convex Area', 'Solidity', 'Equivalent Diameter', 'Major Axis', 'Minor Axis']
        features = to_numerical_features(df, trn_idx, cols)
        labels = to_labels(df['Class'])
    elif data == 'diabetes':
        df = pd.read_csv('{}/{}/pima-indians-diabetes.csv'.format(data_in, data), header=None)
        trn_idx, test_idx = split(df.shape[0])
        df.iloc[:, 1:-1] = df.iloc[:, 1:-1].replace(0, np.nan)
        features = to_numerical_features(df.iloc[:, 1:-1], trn_idx)
        labels = to_labels(df.iloc[:, -1])
    elif data == 'synthetic':
        df = pd.read_csv('{}/breast-cancer-wisconsin/breast-cancer-wisconsin.data'.format(data_in), header=None)
        trn_idx, test_idx = split(df.shape[0])
        features = to_numerical_features(df, trn_idx, list(range(1, 10)))
        features = TSNE(random_state=0).fit_transform(features)
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        labels = to_labels(df[10])
    else:
        raise ValueError(data)

    print('{}\t{}\t{}\t{}'.format(data, features.shape[0], features.shape[1], labels.max() + 1))

    trn_x = features[trn_idx]
    trn_y = labels[trn_idx]
    test_x = features[test_idx]
    test_y = labels[test_idx]

    os.makedirs('{}/{}'.format(data_out, data), exist_ok=True)
    np.save('{}/{}/trn_x'.format(data_out, data), trn_x)
    np.save('{}/{}/trn_y'.format(data_out, data), trn_y)
    np.save('{}/{}/test_x'.format(data_out, data), test_x)
    np.save('{}/{}/test_y'.format(data_out, data), test_y)


def read_dataset(dataset):
    """
    Read a preprocessed dataset for training or test.
    """
    trn_x = np.load('../data/preprocessed/{}/trn_x.npy'.format(dataset))
    trn_y = np.load('../data/preprocessed/{}/trn_y.npy'.format(dataset))
    test_x = np.load('../data/preprocessed/{}/test_x.npy'.format(dataset))
    test_y = np.load('../data/preprocessed/{}/test_y.npy'.format(dataset))
    return trn_x, trn_y, test_x, test_y


def main():
    """
    Main function for preprocessing all datasets.
    """
    print('Preprocessing data...')
    print('data\texamples\tfeatures\tlabels')
    preprocess('brain-tumor')
    preprocess('breast-cancer')
    preprocess('breast-cancer-wisconsin')
    preprocess('diabetes')
    preprocess('heart-disease')
    preprocess('hepatitis')
    preprocess('synthetic')
    print()


if __name__ == '__main__':
    main()
