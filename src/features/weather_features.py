import pandas as pd
import numpy as np
from typing import Union

def calculate_vpd(temp: Union[float, np.ndarray], humidity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcola il Deficit di Pressione di Vapore (VPD).
    VPD è una misura della domanda evaporativa dell'aria.

    Parameters
    ----------
    temp : float or np.ndarray
        Temperatura in Celsius
    humidity : float or np.ndarray
        Umidità relativa (0-100)

    Returns
    -------
    float or np.ndarray
        VPD in kPa
    """
    # Pressione di vapore saturo (kPa)
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))

    # Pressione di vapore attuale (kPa)
    ea = es * (humidity / 100.0)

    # VPD (kPa)
    vpd = es - ea

    return np.maximum(vpd, 0)  # VPD non può essere negativo


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature relative alla radiazione solare.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame di input

    Returns
    -------
    pd.DataFrame
        DataFrame con feature solari aggiunte
    """
    # Calcola angolo solare
    df['solar_angle'] = np.sin(df['day_of_year'] * (2 * np.pi / 365.25)) * \
                        np.sin(df['hour'] * (2 * np.pi / 24))

    # Interazioni tra feature rilevanti
    df['cloud_temp_interaction'] = df['cloudcover'] * df['temp']
    df['visibility_cloud_interaction'] = df['visibility'] * (100 - df['cloudcover'])

    # Feature derivate
    df['clear_sky_index'] = (100 - df['cloudcover']) / 100
    df['temp_gradient'] = df['temp'] - df['tempmin']

    # Feature di efficienza solare
    df['solar_efficiency'] = df['solarenergy'] / (df['solarradiation'] + 1e-6)  # evita divisione per zero
    df['solar_temp_ratio'] = df['solarradiation'] / (df['temp'] + 273.15)  # temperatura in Kelvin

    return df


def add_solar_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature specifiche per l'analisi solare.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame di input

    Returns
    -------
    pd.DataFrame
        DataFrame con feature solari specifiche aggiunte
    """
    # Angolo solare e durata del giorno
    df['day_length'] = 12 + 3 * np.sin(2 * np.pi * (df['day_of_year'] - 81) / 365.25)
    df['solar_noon'] = 12 - df['hour']
    df['solar_elevation'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25) * \
                            np.cos(2 * np.pi * df['solar_noon'] / 24)

    # Interazioni
    df['cloud_elevation'] = df['cloudcover'] * df['solar_elevation']
    df['visibility_elevation'] = df['visibility'] * df['solar_elevation']

    # Rolling features
    df['cloud_rolling_12h'] = df['cloudcover'].rolling(window=12, min_periods=1).mean()
    df['temp_rolling_12h'] = df['temp'].rolling(window=12, min_periods=1).mean()

    # Feature di efficienza energetica
    df['solar_energy_density'] = df['solarenergy'] / df['day_length']
    df['cloud_impact'] = df['solarradiation'] * (1 - df['cloudcover'] / 100)

    return df


def add_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature ambientali derivate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame di input

    Returns
    -------
    pd.DataFrame
        DataFrame con feature ambientali aggiunte
    """
    # Calcola VPD
    df['vpd'] = calculate_vpd(df['temp'], df['humidity'])

    # Feature di stress idrico
    df['water_stress_index'] = df['vpd'] * (1 - df['humidity'] / 100)
    df['dryness_index'] = (df['temp'] - df['dew']) * (100 - df['humidity']) / 100

    # Indici di comfort
    df['heat_index'] = np.where(
        df['temp'] >= 27,
        -42.379 + 2.04901523 * df['temp'] + 10.14333127 * df['humidity'] -
        0.22475541 * df['temp'] * df['humidity'] - 0.00683783 * df['temp'] ** 2 -
        0.05481717 * df['humidity'] ** 2 + 0.00122874 * df['temp'] ** 2 * df['humidity'] +
        0.00085282 * df['temp'] * df['humidity'] ** 2 -
        0.00000199 * df['temp'] ** 2 * df['humidity'] ** 2,
        df['temp']
    )

    # Rolling means per trend
    windows = [3, 6, 12, 24]  # ore
    for window in windows:
        df[f'temp_rolling_mean_{window}h'] = df['temp'].rolling(window=window, min_periods=1).mean()
        df[f'humid_rolling_mean_{window}h'] = df['humidity'].rolling(window=window, min_periods=1).mean()
        df[f'precip_rolling_sum_{window}h'] = df['precip'].rolling(window=window, min_periods=1).sum()

    return df


def add_weather_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge indicatori meteorologici complessi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame di input

    Returns
    -------
    pd.DataFrame
        DataFrame con indicatori meteorologici aggiunti
    """
    # Indicatori di stabilità atmosferica
    df['temp_stability'] = df['temp_rolling_mean_12h'].std()
    df['pressure_tendency'] = df['pressure'].diff()

    # Indicatori di precipitazioni
    df['rain_intensity'] = np.where(
        df['precip'] > 0,
        df['precip'] / (df['precip_rolling_sum_24h'] + 1e-6),
        0
    )
    df['dry_spell'] = (df['precip'] == 0).astype(int).groupby(
        (df['precip'] != 0).cumsum()
    ).cumsum()

    # Indicatori di comfort termico
    df['apparent_temp'] = df['temp'] + 0.33 * df['vpd'] - 0.7 * df['windspeed'] - 4.0
    df['frost_risk'] = (df['temp'] < 2).astype(int)
    df['heat_stress'] = (df['temp'] > 30).astype(int) * (df['humidity'] > 70).astype(int)

    # Indicatori di qualità dell'aria
    df['stagnation_index'] = (df['windspeed'] < 5).astype(int) * (df['cloudcover'] > 80).astype(int)
    df['visibility_index'] = df['visibility'] * (1 - df['cloudcover'] / 100)

    # Indicatori agrometeorologici
    df['growing_degree_days'] = np.maximum(0, df['temp'] - 10)  # base 10°C
    df['chill_hours'] = (df['temp'] < 7).astype(int)
    df['evapotranspiration_proxy'] = df['vpd'] * df['solarradiation'] * (1 + 0.536 * df['windspeed'])

    return df