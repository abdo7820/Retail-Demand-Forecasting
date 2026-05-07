import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df


def get_data_info(df):
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'numeric_stats': df.describe().round(2).to_dict(),
        'categorical_cols': list(df.select_dtypes(include=['object']).columns),
        'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
        'sample_data': df.head(10).to_dict('records'),
        'duplicates': int(df.duplicated().sum()),
    }
    return info


def handle_missing_values(df, strategy='mean'):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
    elif strategy == 'drop':
        df = df.dropna()
    elif strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')

    return df


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def encode_categorical(df, columns=None, method='label'):
    df = df.copy()
    encoders = {}

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    if method == 'label':
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=True)

    return df, encoders


def scale_features(df, columns=None, method='standard'):
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return df, None

    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def feature_engineering(df):
    df = df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['day_of_month'] = df['date'].dt.day

    if 'base_price' in df.columns and 'final_price' in df.columns:
        df['price_diff'] = df['base_price'] - df['final_price']
        df['discount_amount'] = df['base_price'] * df['discount_pct']

    if 'final_price' in df.columns and 'competitor_price' in df.columns:
        df['price_ratio'] = df['final_price'] / df['competitor_price'].replace(0, np.nan)
        df['price_ratio'] = df['price_ratio'].fillna(1.0)
        df['competitor_diff'] = df['competitor_price'] - df['final_price']
        df['is_cheaper'] = (df['final_price'] < df['competitor_price']).astype(int)

    if 'stock_level' in df.columns and 'demand' in df.columns:
        df['stock_demand_ratio'] = df['stock_level'] / df['demand'].replace(0, np.nan)
        df['stock_demand_ratio'] = df['stock_demand_ratio'].fillna(0.0)

    if 'demand' in df.columns:
        q33 = df['demand'].quantile(0.33)
        q66 = df['demand'].quantile(0.66)
        df['demand_level'] = pd.cut(
            df['demand'],
            bins=[-np.inf, q33, q66, np.inf],
            labels=['Low', 'Medium', 'High']
        )

    return df


def prepare_regression_data(df, target='demand'):
    df = df.copy()
    drop_cols = ['date', 'demand_level']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df, _ = encode_categorical(df, columns=cat_cols, method='label')

    X = df.drop(columns=[target], errors='ignore')
    y = df[target] if target in df.columns else None

    return X, y


def prepare_classification_data(df, target='demand_level'):
    df = df.copy()
    drop_cols = ['date', 'demand']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    if target in df.columns:
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
    else:
        le = None

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df, _ = encode_categorical(df, columns=cat_cols, method='label')

    X = df.drop(columns=[target], errors='ignore')
    y = df[target] if target in df.columns else None

    return X, y, le


def prepare_timeseries_data(df, target='demand', lag_features=7):
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    ts_df = df.groupby('date')[target].mean().reset_index()
    ts_df.columns = ['date', target]

    for lag in range(1, lag_features + 1):
        ts_df[f'{target}_lag_{lag}'] = ts_df[target].shift(lag)

    ts_df[f'{target}_rolling_mean_7'] = ts_df[target].rolling(window=7, min_periods=1).mean()
    ts_df[f'{target}_rolling_std_7'] = ts_df[target].rolling(window=7, min_periods=1).std().fillna(0)
    ts_df[f'{target}_rolling_mean_14'] = ts_df[target].rolling(window=14, min_periods=1).mean()
    ts_df[f'{target}_ewm_7'] = ts_df[target].ewm(span=7, min_periods=1).mean()

    ts_df['day_of_week'] = ts_df['date'].dt.dayofweek
    ts_df['month'] = ts_df['date'].dt.month
    ts_df['quarter'] = ts_df['date'].dt.quarter
    ts_df['week_of_year'] = ts_df['date'].dt.isocalendar().week.astype(int)

    ts_df = ts_df.dropna().reset_index(drop=True)

    X = ts_df.drop(columns=['date', target])
    y = ts_df[target]
    dates = ts_df['date']

    return X, y, dates
