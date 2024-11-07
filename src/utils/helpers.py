import psutil
import multiprocessing
import re
import pandas as pd
from typing import List


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