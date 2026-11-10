"""The eleven tabular datasets of the paper.

Raw files expected directly under ``data_dir``:

    credit_card_fraud    creditcard.csv
    aps_failure          aps_failure_training_set.csv, aps_failure_test_set.csv
    bank_marketing       bank-full.csv
    telco_churn          WA_Fn-UseC_-Telco-Customer-Churn.csv
    german_credit        german.data
    secom                secom.data, secom_labels.data
    credit_card_default  default of credit card clients.xls   (reading .xls needs xlrd)
    glass                glass.data
    steel_faults         Faults.NNA
    yeast                yeast.data
    page_blocks          page-blocks.data

aps_failure uses its official train/test files (missing values filled with the training
means). Every other dataset is split 70/30 (stratified, random_state=42) and standardised
with a StandardScaler fit on the training part. Some encodings and imputations below are
fit on the whole dataset before the split; they are kept as they were for reproducibility.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASETS = ['credit_card_fraud', 'aps_failure', 'bank_marketing', 'telco_churn',
            'german_credit', 'secom', 'credit_card_default',
            'glass', 'steel_faults', 'yeast', 'page_blocks']


def load_dataset(dataset_name, data_dir, test_size=0.3, random_state=42):
    """Read, encode, split and scale one dataset.

    Returns:
        (X_train, X_test, y_train, y_test); X as a DataFrame or ndarray, y as a Series
        or ndarray, depending on the dataset.
    """
    if dataset_name == 'aps_failure':
        train_df = pd.read_csv(os.path.join(data_dir, 'aps_failure_training_set.csv'),
                               skiprows=20, na_values='na')
        test_df = pd.read_csv(os.path.join(data_dir, 'aps_failure_test_set.csv'),
                              skiprows=20, na_values='na')
        train_df['class'] = train_df['class'].map({'pos': 1, 'neg': 0})
        test_df['class'] = test_df['class'].map({'pos': 1, 'neg': 0})

        X_train = train_df.drop('class', axis=1)
        y_train = train_df['class']
        X_test = test_df.drop('class', axis=1)
        y_test = test_df['class']

        mean_values = X_train.mean()
        X_train.fillna(mean_values, inplace=True)
        X_test.fillna(mean_values, inplace=True)   # training means, also for the test set

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    elif dataset_name == 'credit_card_fraud':
        df = pd.read_csv(os.path.join(data_dir, 'creditcard.csv'))
        df = df.drop('Time', axis=1)
        # Amount is scaled on the whole dataset before the split (kept for reproducibility).
        scaler = StandardScaler()
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        X = df.drop('Class', axis=1)
        y = df['Class']

    elif dataset_name == 'credit_card_default':
        df = pd.read_excel(os.path.join(data_dir, 'default of credit card clients.xls'), header=1)
        df = df.drop(columns=['ID'])
        df.rename(columns={'default payment next month': 'target'}, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)
        X = df.drop('target', axis=1)
        y = df['target']
        categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
        preprocessor = ColumnTransformer(
            [('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
              categorical_features)],
            remainder='passthrough')
        X = preprocessor.fit_transform(X)   # fit before the split (kept for reproducibility)

    elif dataset_name == 'telco_churn':
        df = pd.read_csv(os.path.join(data_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
        df = df.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # The original `df['TotalCharges'].fillna(0, inplace=True)` (a chained assignment that
        # pandas 3 no longer applies) had exactly this effect under pandas 2.
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        binary_cols = [col for col in X.columns
                       if X[col].nunique() == 2 and X[col].dtype == 'object']
        multi_cols = [col for col in X.columns
                      if X[col].nunique() > 2 and X[col].dtype == 'object']
        for col in binary_cols:
            X[col], _ = pd.factorize(X[col])
        X = pd.get_dummies(X, columns=multi_cols, drop_first=True)

    elif dataset_name == 'german_credit':
        column_names = [f'attr_{i}' for i in range(20)] + ['target']
        df = pd.read_csv(os.path.join(data_dir, 'german.data'), header=None, sep=r'\s+',
                         names=column_names)
        df['target'] = df['target'].map({1: 0, 2: 1})
        X = df.drop('target', axis=1)
        y = df['target']
        categorical_features = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    elif dataset_name == 'secom':
        X = pd.read_csv(os.path.join(data_dir, 'secom.data'), header=None, sep=' ')
        y_df = pd.read_csv(os.path.join(data_dir, 'secom_labels.data'), header=None, sep=' ')
        y = y_df[0].map({-1: 0, 1: 1})
        X.fillna(X.mean(), inplace=True)   # whole-dataset means (kept for reproducibility)

    elif dataset_name == 'bank_marketing':
        df = pd.read_csv(os.path.join(data_dir, 'bank-full.csv'), sep=';')
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        X = df.drop('y', axis=1)
        y = df['y']
        categorical_features = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    elif dataset_name == 'glass':
        df = pd.read_csv(os.path.join(data_dir, 'glass.data'), header=None)
        X = df.drop(columns=[0, 10])
        y = df[10] - 1   # labels 1-7 without 4: the gap is remapped in load_tabular_dataset

    elif dataset_name == 'steel_faults':
        df = pd.read_csv(os.path.join(data_dir, 'Faults.NNA'), header=None, sep=r'\s+')
        X = df.iloc[:, :-7]
        y = pd.Series(np.argmax(df.iloc[:, -7:].values, axis=1))   # one-hot label columns

    elif dataset_name == 'yeast':
        df = pd.read_csv(os.path.join(data_dir, 'yeast.data'), header=None, sep=r'\s+')
        X = df.drop(columns=[0, 9])
        y, _ = pd.factorize(df[9])
        y = pd.Series(y)

    elif dataset_name == 'page_blocks':
        df = pd.read_csv(os.path.join(data_dir, 'page-blocks.data'), header=None, sep=r'\s+')
        X = df.drop(columns=[10])
        y = df[10] - 1

    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    if isinstance(X_train, pd.DataFrame):
        cols, train_idx, test_idx = X_train.columns, X_train.index, X_test.index
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=train_idx)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols, index=test_idx)
    else:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_tabular_dataset(name, data_dir):
    """Load one dataset as arrays ready for training, labels remapped to 0..C-1.

    Returns:
        (X_tr, X_te, y_tr, y_te, class_counts, num_classes) with float32 features,
        int64 labels and the training count of each class.
    """
    X_tr, X_te, y_tr, y_te = load_dataset(name, data_dir, test_size=0.3, random_state=42)

    X_tr = (X_tr.values if hasattr(X_tr, 'values') else X_tr).astype(np.float32)
    X_te = (X_te.values if hasattr(X_te, 'values') else X_te).astype(np.float32)
    y_tr = np.array(y_tr, dtype=np.int64)
    y_te = np.array(y_te, dtype=np.int64)

    unique = np.unique(y_tr)
    if not np.array_equal(unique, np.arange(len(unique))):   # e.g. glass {0,1,2,4,5,6}
        remap = {v: i for i, v in enumerate(unique)}
        y_tr = np.array([remap[v] for v in y_tr], dtype=np.int64)
        y_te = np.array([remap[v] for v in y_te], dtype=np.int64)

    num_classes = len(np.unique(y_tr))
    class_counts = [int((y_tr == c).sum()) for c in range(num_classes)]
    ir = max(class_counts) / max(min(class_counts), 1)
    print(f'{name:25s}  feat={X_tr.shape[1]:4d}  C={num_classes:2d}  '
          f'train={len(y_tr):6d}  IR={ir:8.1f}')
    return X_tr, X_te, y_tr, y_te, class_counts, num_classes
