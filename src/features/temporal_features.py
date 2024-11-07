import pandas as pd
import numpy as np
from typing import Union, Optional
from datetime import datetime


def get_season(date: datetime) -> str:
    """
    Determina la stagione in base alla data.

    Parameters
    ----------
    date : datetime
        Data per cui determinare la stagione

    Returns
    -------
    str
        Nome della stagione ('Winter', 'Spring', 'Summer', 'Autumn')
    """
    month = date.month
    day = date.day

    if (month == 12 and day >= 21) or (month <= 3 and day < 20):
        return 'Winter'
    elif (month == 3 and day >= 20) or (month <= 6 and day < 21):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month <= 9 and day < 23):
        return 'Summer'
    elif (month == 9 and day >= 23) or (month <= 12 and day < 21):
        return 'Autumn'
    else:
        return 'Unknown'


def get_time_period(hour: int) -> str:
    """
    Determina il periodo del giorno in base all'ora.

    Parameters
    ----------
    hour : int
        Ora del giorno (0-23)

    Returns
    -------
    str
        Periodo del giorno ('Morning', 'Afternoon', 'Evening', 'Night')
    """
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature temporali al DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenente una colonna 'datetime'

    Returns
    -------
    pd.DataFrame
        DataFrame con feature temporali aggiuntive
    """
    # Assicurati che datetime sia nel formato corretto
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Feature temporali di base
    df['timestamp'] = df['datetime'].astype(np.int64) // 10 ** 9
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    # Feature cicliche
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))

    # Feature calendario
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['datetime'].dt.quarter

    # Feature cicliche giorno dell'anno
    df['day_of_year_sin'] = np.sin(df['day_of_year'] * (2 * np.pi / 365.25))
    df['day_of_year_cos'] = np.cos(df['day_of_year'] * (2 * np.pi / 365.25))

    # Flag speciali
    df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = df['datetime'].dt.is_quarter_end.astype(int)
    df['is_year_end'] = df['datetime'].dt.is_year_end.astype(int)

    # Periodi del giorno e stagioni
    df['season'] = df['datetime'].apply(get_season)
    df['time_period'] = df['hour'].apply(get_time_period)

    return df


def create_time_based_features(
        df: pd.DataFrame,
        datetime_col: str = 'datetime',
        add_cyclical: bool = True,
        add_time_periods: bool = True,
        add_seasons: bool = True,
        custom_features: Optional[list] = None
) -> pd.DataFrame:
    """
    Crea feature temporali personalizzate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame di input
    datetime_col : str
        Nome della colonna datetime
    add_cyclical : bool
        Se True, aggiunge feature cicliche
    add_time_periods : bool
        Se True, aggiunge periodi del giorno
    add_seasons : bool
        Se True, aggiunge stagioni
    custom_features : list, optional
        Lista di feature temporali personalizzate da aggiungere

    Returns
    -------
    pd.DataFrame
        DataFrame con le nuove feature temporali
    """
    # Crea una copia del DataFrame
    result = df.copy()

    # Converti la colonna datetime se necessario
    if not pd.api.types.is_datetime64_any_dtype(result[datetime_col]):
        result[datetime_col] = pd.to_datetime(result[datetime_col])

    # Feature temporali di base
    result['year'] = result[datetime_col].dt.year
    result['month'] = result[datetime_col].dt.month
    result['day'] = result[datetime_col].dt.day
    result['hour'] = result[datetime_col].dt.hour
    result['day_of_week'] = result[datetime_col].dt.dayofweek
    result['day_of_year'] = result[datetime_col].dt.dayofyear

    # Feature cicliche
    if add_cyclical:
        # Ora
        result['hour_sin'] = np.sin(result['hour'] * (2 * np.pi / 24))
        result['hour_cos'] = np.cos(result['hour'] * (2 * np.pi / 24))

        # Mese
        result['month_sin'] = np.sin((result['month'] - 1) * (2 * np.pi / 12))
        result['month_cos'] = np.cos((result['month'] - 1) * (2 * np.pi / 12))

        # Giorno dell'anno
        result['day_of_year_sin'] = np.sin((result['day_of_year'] - 1) * (2 * np.pi / 365.25))
        result['day_of_year_cos'] = np.cos((result['day_of_year'] - 1) * (2 * np.pi / 365.25))

        # Giorno della settimana
        result['day_of_week_sin'] = np.sin(result['day_of_week'] * (2 * np.pi / 7))
        result['day_of_week_cos'] = np.cos(result['day_of_week'] * (2 * np.pi / 7))

    # Periodi del giorno
    if add_time_periods:
        result['time_period'] = result['hour'].apply(get_time_period)
        # One-hot encoding del periodo del giorno
        time_period_dummies = pd.get_dummies(result['time_period'], prefix='time_period')
        result = pd.concat([result, time_period_dummies], axis=1)

    # Stagioni
    if add_seasons:
        result['season'] = result[datetime_col].apply(get_season)
        # One-hot encoding delle stagioni
        season_dummies = pd.get_dummies(result['season'], prefix='season')
        result = pd.concat([result, season_dummies], axis=1)

    # Feature personalizzate
    if custom_features:
        for feature in custom_features:
            if feature == 'is_weekend':
                result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
            elif feature == 'is_business_hour':
                result['is_business_hour'] = ((result['hour'] >= 9) &
                                              (result['hour'] < 18) &
                                              ~result['is_weekend']).astype(int)
            elif feature == 'season_progress':
                result['season_progress'] = result.apply(
                    lambda x: (x['day_of_year'] % 91) / 91.0, axis=1
                )

    return result