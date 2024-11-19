import psutil
import multiprocessing
import re
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import joblib


def get_optimal_workers() -> int:
    """
    Calcola il numero ottimale di workers basandosi sulle risorse del sistema.

    Returns
    -------
    int
        Numero ottimale di workers
    """
    # Ottiene il numero di CPU logiche (inclusi i thread virtuali)
    cpu_count = multiprocessing.cpu_count()

    # Ottiene la memoria totale e disponibile in GB
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024 ** 3)
    available_memory_gb = memory.available / (1024 ** 3)

    # Stima della memoria necessaria per worker (esempio: 2GB per worker)
    memory_per_worker_gb = 2

    # Calcola il numero massimo di workers basato sulla memoria disponibile
    max_workers_by_memory = int(available_memory_gb / memory_per_worker_gb)

    # Usa il minimo tra:
    # - numero di CPU disponibili - 1 (lascia una CPU libera per il sistema)
    # - numero massimo di workers basato sulla memoria
    # - un limite massimo arbitrario (es. 32) per evitare troppo overhead
    optimal_workers = min(
        cpu_count - 1,
        max_workers_by_memory,
        32  # limite massimo arbitrario
    )

    # Assicura almeno 1 worker
    return max(1, optimal_workers)


def clean_column_name(name: str) -> str:
    """
    Rimuove caratteri speciali e spazi, converte in snake_case e abbrevia.

    Parameters
    ----------
    name : str
        Nome della colonna da pulire

    Returns
    -------
    str
        Nome della colonna pulito
    """
    # Rimuove caratteri speciali
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    # Converte in snake_case
    name = name.lower().replace(' ', '_')

    # Abbreviazioni comuni
    abbreviations = {
        'production': 'prod',
        'percentage': 'pct',
        'hectare': 'ha',
        'tonnes': 't',
        'litres': 'l',
        'minimum': 'min',
        'maximum': 'max',
        'average': 'avg'
    }

    for full, abbr in abbreviations.items():
        name = name.replace(full, abbr)

    return name


def clean_column_names(df: pd.DataFrame) -> List[str]:
    """
    Pulisce tutti i nomi delle colonne in un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con le colonne da pulire

    Returns
    -------
    list
        Lista dei nuovi nomi delle colonne puliti
    """
    new_columns = []

    for col in df.columns:
        # Usa regex per separare le varietà
        varieties = re.findall(r'([a-z]+)_([a-z_]+)', col)
        if varieties:
            new_columns.append(f"{varieties[0][0]}_{varieties[0][1]}")
        else:
            new_columns.append(col)

    return new_columns


def to_camel_case(text: str) -> str:
    """
    Converte una stringa in camelCase.
    Gestisce stringhe con spazi, trattini o underscore.
    Se è una sola parola, la restituisce in minuscolo.

    Parameters
    ----------
    text : str
        Testo da convertire

    Returns
    -------
    str
        Testo convertito in camelCase
    """
    # Rimuove eventuali spazi iniziali e finali
    text = text.strip()

    # Se la stringa è vuota, ritorna stringa vuota
    if not text:
        return ""

    # Sostituisce trattini e underscore con spazi
    text = text.replace('-', ' ').replace('_', ' ')

    # Divide la stringa in parole
    words = text.split()

    # Se non ci sono parole dopo lo split, ritorna stringa vuota
    if not words:
        return ""

    # Se c'è una sola parola, ritorna in minuscolo
    if len(words) == 1:
        return words[0].lower()

    # Altrimenti procedi con il camelCase
    result = words[0].lower()
    for word in words[1:]:
        result += word.capitalize()

    return result


def get_full_data(simulated_data: pd.DataFrame,
                  olive_varieties: pd.DataFrame) -> pd.DataFrame:
    """
    Ottiene il dataset completo combinando dati simulati e varietà di olive.

    Parameters
    ----------
    simulated_data : pd.DataFrame
        DataFrame con i dati simulati
    olive_varieties : pd.DataFrame
        DataFrame con le informazioni sulle varietà

    Returns
    -------
    pd.DataFrame
        DataFrame completo con tutte le informazioni
    """
    # Colonne base rilevanti
    relevant_columns = [
        'year', 'temp_mean', 'precip_sum', 'solar_energy_sum',
        'ha', 'zone', 'olive_prod'
    ]

    # Aggiungi colonne specifiche per varietà
    all_varieties = olive_varieties['Varietà di Olive'].unique()
    varieties = [clean_column_name(variety) for variety in all_varieties]

    for variety in varieties:
        relevant_columns.extend([
            f'{variety}_olive_prod',
            f'{variety}_tech'
        ])

    # Seleziona solo le colonne rilevanti
    full_data = simulated_data[relevant_columns].copy()

    # Aggiungi feature calcolate
    for variety in varieties:
        # Calcola efficienza produttiva
        if f'{variety}_olive_prod' in full_data.columns:
            full_data[f'{variety}_efficiency'] = (
                    full_data[f'{variety}_olive_prod'] / full_data['ha']
            )

        # Aggiungi indicatori tecnici
        if f'{variety}_tech' in full_data.columns:
            technique_dummies = pd.get_dummies(
                full_data[f'{variety}_tech'],
                prefix=f'{variety}_technique'
            )
            full_data = pd.concat([full_data, technique_dummies], axis=1)

    # Aggiungi feature temporali
    full_data['month'] = 1  # Assumiamo dati annuali
    full_data['day'] = 1  # Assumiamo dati annuali

    # Calcola medie mobili
    for col in ['temp_mean', 'precip_sum', 'solar_energy_sum']:
        full_data[f'{col}_ma3'] = full_data[col].rolling(window=3, min_periods=1).mean()
        full_data[f'{col}_ma5'] = full_data[col].rolling(window=5, min_periods=1).mean()

    return full_data

def prepare_static_features_multiple(varieties_info: List[Dict],
                                     percentages: List[float],
                                     hectares: float,
                                     all_varieties: List[str]) -> np.ndarray:
    """
    Prepara le feature statiche per multiple varietà.

    Parameters
    ----------
    varieties_info : List[Dict]
        Lista di dizionari contenenti le informazioni sulle varietà selezionate
    percentages : List[float]
        Lista delle percentuali corrispondenti a ciascuna varietà selezionata
    hectares : float
        Numero di ettari totali
    all_varieties : List[str]
        Lista di tutte le possibili varietà nel dataset originale

    Returns
    -------
    np.ndarray
        Array numpy contenente tutte le feature statiche
    """
    # Inizializza un dizionario per tutte le varietà possibili
    variety_data = {variety.lower(): {
        'pct': 0,
        'prod_t_ha': 0,
        'tech': '',
        'oil_prod_t_ha': 0,
        'oil_prod_l_ha': 0,
        'min_yield_pct': 0,
        'max_yield_pct': 0,
        'min_oil_prod_l_ha': 0,
        'max_oil_prod_l_ha': 0,
        'avg_oil_prod_l_ha': 0,
        'l_per_t': 0,
        'min_l_per_t': 0,
        'max_l_per_t': 0,
        'avg_l_per_t': 0,
        'water_need_spring': 0,
        'water_need_summer': 0,
        'water_need_autumn': 0,
        'water_need_winter': 0,
        'annual_water_need': 0,
        'optimal_temp': 0,
        'drought_resistance': 0
    } for variety in all_varieties}

    # Aggiorna i dati per le varietà selezionate
    for variety_info, percentage in zip(varieties_info, percentages):
        variety_name = clean_column_name(variety_info['variet_di_olive']).lower()
        technique = clean_column_name(variety_info['tecnica_di_coltivazione']).lower()

        if variety_name not in variety_data:
            print(f"Attenzione: La varietà '{variety_name}' non è presente nella lista delle varietà conosciute.")
            continue

        variety_data[variety_name].update({
            'pct': percentage / 100,
            'prod_t_ha': variety_info['produzione_tonnellateettaro'],
            'tech': technique,
            'oil_prod_t_ha': variety_info['produzione_olio_tonnellateettaro'],
            'oil_prod_l_ha': variety_info['produzione_olio_litriettaro'],
            'min_yield_pct': variety_info['min__resa'],
            'max_yield_pct': variety_info['max__resa'],
            'min_oil_prod_l_ha': variety_info['min_produzione_olio_litriettaro'],
            'max_oil_prod_l_ha': variety_info['max_produzione_olio_litriettaro'],
            'avg_oil_prod_l_ha': variety_info['media_produzione_olio_litriettaro'],
            'l_per_t': variety_info['litri_per_tonnellata'],
            'min_l_per_t': variety_info['min_litri_per_tonnellata'],
            'max_l_per_t': variety_info['max_litri_per_tonnellata'],
            'avg_l_per_t': variety_info['media_litri_per_tonnellata'],
            'water_need_spring': variety_info['fabbisogno_acqua_primavera_mettaro'],
            'water_need_summer': variety_info['fabbisogno_acqua_estate_mettaro'],
            'water_need_autumn': variety_info['fabbisogno_acqua_autunno_mettaro'],
            'water_need_winter': variety_info['fabbisogno_acqua_inverno_mettaro'],
            'annual_water_need': variety_info['fabbisogno_idrico_annuale_mettaro'],
            'optimal_temp': variety_info['temperatura_ottimale'],
            'drought_resistance': variety_info['resistenza_alla_siccit']
        })

    # Crea il vettore delle feature
    static_features = [hectares]

    # Lista delle feature per ogni varietà
    variety_features = ['pct', 'prod_t_ha', 'oil_prod_t_ha', 'oil_prod_l_ha',
                        'min_yield_pct', 'max_yield_pct', 'min_oil_prod_l_ha',
                        'max_oil_prod_l_ha', 'avg_oil_prod_l_ha', 'l_per_t',
                        'min_l_per_t', 'max_l_per_t', 'avg_l_per_t',
                        'water_need_spring', 'water_need_summer', 'water_need_autumn',
                        'water_need_winter', 'annual_water_need', 'optimal_temp',
                        'drought_resistance']

    # Appiattisci i dati delle varietà
    for variety in all_varieties:
        variety_lower = variety.lower()
        # Feature esistenti
        for feature in variety_features:
            static_features.append(variety_data[variety_lower][feature])

        # Feature binarie per le tecniche
        for technique in ['tradizionale', 'intensiva', 'superintensiva']:
            static_features.append(1 if variety_data[variety_lower]['tech'] == technique else 0)

    return np.array(static_features).reshape(1, -1)


def get_feature_names(all_varieties: List[str]) -> List[str]:
    """
    Genera i nomi delle feature nell'ordine corretto.

    Parameters
    ----------
    all_varieties : List[str]
        Lista di tutte le varietà possibili

    Returns
    -------
    List[str]
        Lista dei nomi delle feature
    """
    feature_names = ['hectares']

    variety_features = ['pct', 'prod_t_ha', 'oil_prod_t_ha', 'oil_prod_l_ha',
                        'min_yield_pct', 'max_yield_pct', 'min_oil_prod_l_ha',
                        'max_oil_prod_l_ha', 'avg_oil_prod_l_ha', 'l_per_t',
                        'min_l_per_t', 'max_l_per_t', 'avg_l_per_t']

    techniques = ['tradizionale', 'intensiva', 'superintensiva']

    for variety in all_varieties:
        for feature in variety_features:
            feature_names.append(f"{variety}_{feature}")
        for technique in techniques:
            feature_names.append(f"{variety}_tech_{technique}")

    return feature_names

def add_controlled_variation(base_value: float, max_variation_pct: float = 0.20) -> float:
    """
    Aggiunge una variazione controllata a un valore base.

    Parameters
    ----------
    base_value : float
        Valore base da modificare
    max_variation_pct : float
        Percentuale massima di variazione (default 20%)

    Returns
    -------
    float
        Valore con variazione applicata
    """
    variation = np.random.uniform(-max_variation_pct, max_variation_pct)
    return base_value * (1 + variation)

def get_growth_phase(month):
    if month in [12, 1, 2]:
        return 'dormancy'
    elif month in [3, 4, 5]:
        return 'flowering'
    elif month in [6, 7, 8]:
        return 'fruit_set'
    else:
        return 'ripening'

def calculate_weather_effect(row, optimal_temp):
    # Effetti base
    temp_effect = -0.1 * (row['temp_mean'] - optimal_temp) ** 2
    rain_effect = -0.05 * (row['precip_sum'] - 600) ** 2 / 10000
    sun_effect = 0.1 * row['solarenergy_sum'] / 1000

    # Fattori di scala basati sulla fase di crescita
    if row['growth_phase'] == 'dormancy':
        temp_scale = 0.5
        rain_scale = 0.2
        sun_scale = 0.1
    elif row['growth_phase'] == 'flowering':
        temp_scale = 2.0
        rain_scale = 1.5
        sun_scale = 1.0
    elif row['growth_phase'] == 'fruit_set':
        temp_scale = 1.5
        rain_scale = 1.0
        sun_scale = 0.8
    else:  # ripening
        temp_scale = 1.0
        rain_scale = 0.5
        sun_scale = 1.2

    # Calcolo dell'effetto combinato
    combined_effect = (
            temp_scale * temp_effect +
            rain_scale * rain_effect +
            sun_scale * sun_effect
    )

    # Aggiustamenti specifici per fase
    if row['growth_phase'] == 'flowering':
        combined_effect -= 0.5 * max(0, row['precip_sum'] - 50)  # Penalità per pioggia eccessiva durante la fioritura
    elif row['growth_phase'] == 'fruit_set':
        combined_effect += 0.3 * max(0, row['temp_mean'] - (optimal_temp + 5))  # Bonus per temperature più alte durante la formazione dei frutti

    return combined_effect

def calculate_water_need(weather_data, base_need, optimal_temp):
    # Calcola il fabbisogno idrico basato su temperatura e precipitazioni
    temp_factor = 1 + 0.05 * (weather_data['temp_mean'] - optimal_temp)  # Aumenta del 5% per ogni grado sopra l'ottimale
    rain_factor = 1 - 0.001 * weather_data['precip_sum']  # Diminuisce leggermente con l'aumentare delle precipitazioni
    return base_need * temp_factor * rain_factor

def create_technique_mapping(olive_varieties, mapping_path='./kaggle/working/models/technique_mapping.joblib'):
    # Estrai tutte le tecniche uniche dal dataset e convertile in lowercase
    all_techniques = olive_varieties['Tecnica di Coltivazione'].str.lower().unique()

    # Crea il mapping partendo da 1
    technique_mapping = {tech: i + 1 for i, tech in enumerate(sorted(all_techniques))}

    # Salva il mapping
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    joblib.dump(technique_mapping, mapping_path)

    return technique_mapping


def encode_techniques(df, mapping_path='./kaggle/working/models/technique_mapping.joblib'):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping not found at {mapping_path}. Run create_technique_mapping first.")

    technique_mapping = joblib.load(mapping_path)

    # Trova tutte le colonne delle tecniche
    tech_columns = [col for col in df.columns if col.endswith('_tech')]

    # Applica il mapping a tutte le colonne delle tecniche
    for col in tech_columns:
        df[col] = df[col].str.lower().map(technique_mapping).fillna(0).astype(int)

    return df


def decode_techniques(df, mapping_path='./kaggle/working/models/technique_mapping.joblib'):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping not found at {mapping_path}")

    technique_mapping = joblib.load(mapping_path)
    reverse_mapping = {v: k for k, v in technique_mapping.items()}
    reverse_mapping[0] = ''  # Aggiungi un mapping per 0 a stringa vuota

    # Trova tutte le colonne delle tecniche
    tech_columns = [col for col in df.columns if col.endswith('_tech')]

    # Applica il reverse mapping a tutte le colonne delle tecniche
    for col in tech_columns:
        df[col] = df[col].map(reverse_mapping)

    return df


def decode_single_technique(technique_value, mapping_path='./kaggle/working/models/technique_mapping.joblib'):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping not found at {mapping_path}")

    technique_mapping = joblib.load(mapping_path)
    reverse_mapping = {v: k for k, v in technique_mapping.items()}
    reverse_mapping[0] = ''

    return reverse_mapping.get(technique_value, '')

def preprocess_weather_data(weather_df):
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

    monthly_weather.columns = ['year', 'month'] + [f'{col[0]}_{col[1]}' for col in monthly_weather.columns[2:]]
    return monthly_weather