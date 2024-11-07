# src/data/data_processor.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, List, Dict, Optional, Union


def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola statistiche mensili per ogni anno dai dati meteo.

    Parameters
    ----------
    weather_df : pd.DataFrame
        DataFrame contenente i dati meteorologici

    Returns
    -------
    pd.DataFrame
        DataFrame con statistiche mensili
    """
    # Calcola statistiche mensili per ogni anno
    monthly_weather = weather_df.groupby(['year', 'month']).agg({
        'temp': ['mean', 'min', 'max'],
        'humidity': 'mean',
        'precip': 'sum',
        'windspeed': 'mean',
        'cloudcover': 'mean',
        'solarradiation': 'sum',
        'solarenergy': 'sum',
        'uvindex': 'max'
    }).reset_index()

    # Rinomina le colonne
    monthly_weather.columns = ['year', 'month'] + [
        f'{col[0]}_{col[1]}' for col in monthly_weather.columns[2:]
    ]

    return monthly_weather


def create_sequences(timesteps: int, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[
    np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Crea sequenze temporali dai dati.

    Parameters
    ----------
    timesteps : int
        Numero di timestep per ogni sequenza
    X : array-like
        Dati di input
    y : array-like, optional
        Target values

    Returns
    -------
    tuple o array
        Se y è fornito: (X_sequences, y_sequences)
        Se y è None: X_sequences
    """
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i + timesteps])

    if y is not None:
        ys = []
        for i in range(len(X) - timesteps):
            ys.append(y[i + timesteps])
        return np.array(Xs), np.array(ys)

    return np.array(Xs)


def prepare_solar_data(weather_data: pd.DataFrame, features: List[str]) -> Tuple:
    """
    Prepara i dati per i modelli solari.

    Parameters
    ----------
    weather_data : pd.DataFrame
        DataFrame contenente i dati meteorologici
    features : list
        Lista delle feature da utilizzare

    Returns
    -------
    tuple
        (X_scaled, scaler_X, y_scaled, scaler_y, data_after_2010)
    """
    # Aggiunge le caratteristiche temporali
    weather_data = add_advanced_features(weather_data)
    weather_data = pd.get_dummies(weather_data, columns=['season', 'time_period'], drop_first=True)

    # Filtra dati dopo 2010
    data_after_2010 = weather_data[weather_data['year'] >= 2010].copy()
    data_after_2010 = data_after_2010.sort_values('datetime')
    data_after_2010.set_index('datetime', inplace=True)

    # Interpola valori mancanti
    target_variables = ['solarradiation', 'solarenergy', 'uvindex']
    for column in target_variables:
        data_after_2010[column] = data_after_2010[column].interpolate(method='time')

    # Rimuovi righe con valori mancanti
    data_after_2010.dropna(subset=features + target_variables, inplace=True)

    # Prepara X e y
    X = data_after_2010[features].values
    y = data_after_2010[target_variables].values

    # Normalizza features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, scaler_X, y_scaled, scaler_y, data_after_2010


def prepare_transformer_data(df: pd.DataFrame, olive_varieties_df: pd.DataFrame) -> Tuple:
    """
    Prepara i dati per il modello transformer.
    """
    # Copia del DataFrame
    df = df.copy()

    # Ordina per zona e anno
    df = df.sort_values(['zone', 'year'])

    # Feature definition
    temporal_features = ['temp_mean', 'precip_sum', 'solar_energy_sum']
    static_features = ['ha']
    target_features = ['olive_prod', 'min_oil_prod', 'max_oil_prod', 'avg_oil_prod', 'total_water_need']

    # Get clean varieties
    all_varieties = olive_varieties_df['Varietà di Olive'].unique()
    varieties = [clean_column_name(variety) for variety in all_varieties]

    # Variety features structure
    variety_features = [
        'tech', 'pct', 'prod_t_ha', 'oil_prod_t_ha', 'oil_prod_l_ha',
        'min_yield_pct', 'max_yield_pct', 'min_oil_prod_l_ha', 'max_oil_prod_l_ha',
        'avg_oil_prod_l_ha', 'l_per_t', 'min_l_per_t', 'max_l_per_t', 'avg_l_per_t'
    ]

    # Prepare columns
    new_columns = {}

    # Prepare features for each variety
    for variety in varieties:
        for feature in variety_features:
            col_name = f"{variety}_{feature}"
            if col_name in df.columns:
                if feature != 'tech':
                    static_features.append(col_name)

        # Binary features for cultivation techniques
        for technique in ['tradizionale', 'intensiva', 'superintensiva']:
            col_name = f"{variety}_{technique}"
            new_columns[col_name] = df[f"{variety}_tech"].notna() & (
                    df[f"{variety}_tech"].str.lower() == technique
            ).fillna(False)
            static_features.append(col_name)

    # Add all new columns at once
    new_df = pd.concat([df] + [pd.Series(v, name=k) for k, v in new_columns.items()], axis=1)

    # Sort by zone and year
    df_sorted = new_df.sort_values(['zone', 'year'])

    # Window size definition
    window_size = 41

    # Prepare lists for data collection
    temporal_sequences = []
    static_features_list = []
    targets_list = []

    # Process data by zone
    for zone in df_sorted['zone'].unique():
        zone_data = df_sorted[df_sorted['zone'] == zone].reset_index(drop=True)

        if len(zone_data) >= window_size:
            for i in range(len(zone_data) - window_size + 1):
                temporal_window = zone_data.iloc[i:i + window_size][temporal_features].values
                if not np.isnan(temporal_window).any():
                    temporal_sequences.append(temporal_window)
                    static_features_list.append(zone_data.iloc[i + window_size - 1][static_features].values)
                    targets_list.append(zone_data.iloc[i + window_size - 1][target_features].values)

    # Convert to numpy arrays
    X_temporal = np.array(temporal_sequences)
    X_static = np.array(static_features_list)
    y = np.array(targets_list)

    # Split data
    indices = np.random.permutation(len(X_temporal))
    train_idx = int(len(indices) * 0.65)
    val_idx = int(len(indices) * 0.85)

    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]

    # Split datasets
    X_temporal_train = X_temporal[train_indices]
    X_temporal_val = X_temporal[val_indices]
    X_temporal_test = X_temporal[test_indices]

    X_static_train = X_static[train_indices]
    X_static_val = X_static[val_indices]
    X_static_test = X_static[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    # Standardization
    scaler_temporal = StandardScaler()
    scaler_static = StandardScaler()
    scaler_y = StandardScaler()

    # Apply standardization
    X_temporal_train = scaler_temporal.fit_transform(X_temporal_train.reshape(-1, len(temporal_features))).reshape(
        X_temporal_train.shape)
    X_temporal_val = scaler_temporal.transform(X_temporal_val.reshape(-1, len(temporal_features))).reshape(
        X_temporal_val.shape)
    X_temporal_test = scaler_temporal.transform(X_temporal_test.reshape(-1, len(temporal_features))).reshape(
        X_temporal_test.shape)

    X_static_train = scaler_static.fit_transform(X_static_train)
    X_static_val = scaler_static.transform(X_static_val)
    X_static_test = scaler_static.transform(X_static_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # Prepare input dictionaries
    train_data = {'temporal': X_temporal_train, 'static': X_static_train}
    val_data = {'temporal': X_temporal_val, 'static': X_static_val}
    test_data = {'temporal': X_temporal_test, 'static': X_static_test}

    # Save scalers
    base_path = './kaggle/working/models/oil_transformer/'
    os.makedirs(base_path, exist_ok=True)
    joblib.dump(scaler_temporal, os.path.join(base_path, 'scaler_temporal.joblib'))
    joblib.dump(scaler_static, os.path.join(base_path, 'scaler_static.joblib'))
    joblib.dump(scaler_y, os.path.join(base_path, 'scaler_y.joblib'))

    return (train_data, y_train), (val_data, y_val), (test_data, y_test), (scaler_temporal, scaler_static, scaler_y)


def encode_techniques(df: pd.DataFrame,
                      mapping_path: str = './kaggle/working/models/technique_mapping.joblib') -> pd.DataFrame:
    """
    Codifica le tecniche di coltivazione usando un mapping salvato.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenente le colonne delle tecniche
    mapping_path : str
        Percorso al file di mapping

    Returns
    -------
    pd.DataFrame
        DataFrame con le tecniche codificate
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping not found at {mapping_path}. Run create_technique_mapping first.")

    technique_mapping = joblib.load(mapping_path)

    # Trova tutte le colonne delle tecniche
    tech_columns = [col for col in df.columns if col.endswith('_tech')]

    # Applica il mapping a tutte le colonne delle tecniche
    for col in tech_columns:
        df[col] = df[col].str.lower().map(technique_mapping).fillna(0).astype(int)

    return df


def decode_techniques(df: pd.DataFrame,
                      mapping_path: str = './kaggle/working/models/technique_mapping.joblib') -> pd.DataFrame:
    """
    Decodifica le tecniche di coltivazione usando un mapping salvato.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenente le colonne delle tecniche codificate
    mapping_path : str
        Percorso al file di mapping

    Returns
    -------
    pd.DataFrame
        DataFrame con le tecniche decodificate
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping not found at {mapping_path}")

    technique_mapping = joblib.load(mapping_path)
    reverse_mapping = {v: k for k, v in technique_mapping.items()}
    reverse_mapping[0] = ''  # Mapping per 0 a stringa vuota

    # Trova tutte le colonne delle tecniche
    tech_columns = [col for col in df.columns if col.endswith('_tech')]

    # Applica il reverse mapping
    for col in tech_columns:
        df[col] = df[col].map(reverse_mapping)

    return df