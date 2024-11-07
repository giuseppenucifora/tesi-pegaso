import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict

def create_technique_mapping(olive_varieties: pd.DataFrame,
                             mapping_path: str = './kaggle/working/models/technique_mapping.joblib') -> Dict[str, int]:
    """
    Crea un mapping numerico per le tecniche di coltivazione.

    Parameters
    ----------
    olive_varieties : pd.DataFrame
        DataFrame contenente le varietà di olive e le tecniche
    mapping_path : str
        Percorso dove salvare il mapping

    Returns
    -------
    Dict[str, int]
        Dizionario di mapping tecnica -> codice numerico
    """
    # Estrai tecniche uniche e convertile in lowercase
    all_techniques = olive_varieties['Tecnica di Coltivazione'].str.lower().unique()

    # Crea il mapping partendo da 1 (0 è riservato per valori mancanti)
    technique_mapping = {tech: i + 1 for i, tech in enumerate(sorted(all_techniques))}

    # Salva il mapping
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    joblib.dump(technique_mapping, mapping_path)

    return technique_mapping


def calculate_stress_index(weather_data: pd.DataFrame,
                           olive_info: pd.Series,
                           vpd_threshold: float = 2.0) -> float:
    """
    Calcola l'indice di stress per le olive basato su condizioni ambientali.

    Parameters
    ----------
    weather_data : pd.DataFrame
        Dati meteorologici
    olive_info : pd.Series
        Informazioni sulla varietà di oliva
    vpd_threshold : float
        Soglia VPD per lo stress

    Returns
    -------
    float
        Indice di stress calcolato
    """
    # Calcola componenti di stress
    temp_stress = np.where(
        weather_data['temp'] > olive_info['Temperatura Ottimale'],
        (weather_data['temp'] - olive_info['Temperatura Ottimale']) / 10,
        0
    )

    water_stress = np.where(
        weather_data['vpd'] > vpd_threshold,
        (weather_data['vpd'] - vpd_threshold) / 2,
        0
    )

    # Considera la resistenza alla siccità
    resistance_factor = 1.0
    if olive_info['Resistenza alla Siccità'] == 'Alta':
        resistance_factor = 0.7
    elif olive_info['Resistenza alla Siccità'] == 'Media':
        resistance_factor = 0.85

    # Calcola stress complessivo
    total_stress = (temp_stress + water_stress * resistance_factor)

    return total_stress.mean()


def calculate_quality_indicators(olive_data: pd.DataFrame,
                                 weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola indicatori di qualità per le olive.

    Parameters
    ----------
    olive_data : pd.DataFrame
        Dati sulle olive
    weather_data : pd.DataFrame
        Dati meteorologici

    Returns
    -------
    pd.DataFrame
        DataFrame con indicatori di qualità aggiunti
    """
    result = olive_data.copy()

    # Calcola indicatori base
    result['oil_content_index'] = result['Max % Resa'] * (1 - result['stress_index'] * 0.1)

    result['fruit_size_index'] = np.clip(
        result['Produzione (tonnellate/ettaro)'] * (1 - result['water_stress'] * 0.15),0, None
    )

    # Calcola indice di maturazione ottimale
    optimal_harvest_conditions = (
            (weather_data['temp'].between(15, 25)) &
            (weather_data['humidity'].between(50, 70)) &
            (weather_data['cloudcover'] < 60)
    )

    result['maturity_index'] = optimal_harvest_conditions.mean()

    # Calcola indice di qualità complessivo
    result['quality_index'] = (
            result['oil_content_index'] * 0.4 +
            result['fruit_size_index'] * 0.3 +
            result['maturity_index'] * 0.3
    )

    return result


def add_olive_features(df: pd.DataFrame,
                       weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature specifiche per le olive.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame delle varietà di olive
    weather_data : pd.DataFrame
        Dati meteorologici

    Returns
    -------
    pd.DataFrame
        DataFrame con feature aggiuntive
    """
    result = df.copy()

    # Calcola stress index per ogni varietà
    result['stress_index'] = result.apply(
        lambda row: calculate_stress_index(weather_data, row),
        axis=1
    )

    # Aggiungi indicatori di qualità
    result = calculate_quality_indicators(result, weather_data)

    # Calcola efficienza produttiva
    result['production_efficiency'] = result['Produzione (tonnellate/ettaro)'] / \
                                      result['Fabbisogno Idrico Annuale (m³/ettaro)']

    # Calcola indice di adattamento climatico
    result['climate_adaptation'] = np.where(
        result['Resistenza alla Siccità'] == 'Alta',
        0.9,
        np.where(result['Resistenza alla Siccità'] == 'Media', 0.7, 0.5)
    )

    # Aggiungi feature di produzione
    result = add_production_features(result, weather_data)

    return result


def add_production_features(df: pd.DataFrame,
                            weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature relative alla produzione di olive.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame delle varietà di olive
    weather_data : pd.DataFrame
        Dati meteorologici

    Returns
    -------
    pd.DataFrame
        DataFrame con feature di produzione
    """
    result = df.copy()

    # Calcola i rapporti di produzione
    result['oil_yield_ratio'] = result['Produzione Olio (tonnellate/ettaro)'] / result['Produzione (tonnellate/ettaro)']

    result['water_efficiency'] = result['Produzione (tonnellate/ettaro)'] / result['Fabbisogno Idrico Annuale (m³/ettaro)']

    # Calcola indici di produttività
    result['productivity_index'] = (
            result['oil_yield_ratio'] * 0.4 +
            result['water_efficiency'] * 0.3 +
            result['climate_adaptation'] * 0.3
    )

    # Aggiungi indicatori di rendimento
    result['yield_stability'] = 1 - (
            (result['Max % Resa'] - result['Min % Resa']) / result['Max % Resa']
    )

    result['oil_quality_potential'] = (
            result['Max Litri per Tonnellata'] / 1000 * result['yield_stability'] * (1 - result['stress_index'] * 0.1)
    )

    # Calcola intervalli di produzione ottimale
    result['optimal_production_lower'] = result['Produzione (tonnellate/ettaro)'] * 0.8
    result['optimal_production_upper'] = result['Produzione (tonnellate/ettaro)'] * 1.2

    # Aggiungi indici economici
    result['economic_efficiency'] = (result['Produzione Olio (litri/ettaro)'] / result['Fabbisogno Idrico Annuale (m³/ettaro)']) * result['productivity_index']

    return result