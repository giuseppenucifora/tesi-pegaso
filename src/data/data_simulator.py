import pandas as pd
import numpy as np
from typing import Dict
from src.utils.helpers import clean_column_name



def calculate_weather_effect(row: pd.Series, optimal_temp: float) -> float:
    """
    Calcola l'effetto delle condizioni meteorologiche sulla produzione.

    Parameters
    ----------
    row : pd.Series
        Serie contenente i dati meteorologici
    optimal_temp : float
        Temperatura ottimale per la varietà

    Returns
    -------
    float
        Effetto combinato delle condizioni meteo
    """
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
        combined_effect -= 0.5 * max(0, row['precip_sum'] - 50)  # Penalità per pioggia eccessiva
    elif row['growth_phase'] == 'fruit_set':
        combined_effect += 0.3 * max(0, row['temp_mean'] - (optimal_temp + 5))  # Bonus temperature alte

    return combined_effect


def calculate_water_need(weather_data: pd.Series, base_need: float, optimal_temp: float) -> float:
    """
    Calcola il fabbisogno idrico basato su temperatura e precipitazioni.

    Parameters
    ----------
    weather_data : pd.Series
        Serie contenente i dati meteorologici
    base_need : float
        Fabbisogno idrico base
    optimal_temp : float
        Temperatura ottimale per la varietà

    Returns
    -------
    float
        Fabbisogno idrico calcolato
    """
    temp_factor = 1 + 0.05 * (weather_data['temp_mean'] - optimal_temp)
    rain_factor = 1 - 0.001 * weather_data['precip_sum']
    return base_need * temp_factor * rain_factor


def add_olive_water_consumption_correlation(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge dati correlati al consumo d'acqua per ogni varietà di oliva.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame contenente i dati delle varietà di olive

    Returns
    -------
    pd.DataFrame
        DataFrame con dati aggiuntivi sul consumo d'acqua
    """
    # Dati simulati per il fabbisogno d'acqua e correlazione con temperatura
    fabbisogno_acqua = {
        "Nocellara dell'Etna": {"Primavera": 1200, "Estate": 2000, "Autunno": 1000, "Inverno": 500,
                                "Temperatura Ottimale": 18, "Resistenza": "Media"},
        "Leccino": {"Primavera": 1000, "Estate": 1800, "Autunno": 800, "Inverno": 400, "Temperatura Ottimale": 20,
                    "Resistenza": "Alta"},
        "Frantoio": {"Primavera": 1100, "Estate": 1900, "Autunno": 900, "Inverno": 450, "Temperatura Ottimale": 19,
                     "Resistenza": "Alta"},
        "Coratina": {"Primavera": 1300, "Estate": 2200, "Autunno": 1100, "Inverno": 550, "Temperatura Ottimale": 17,
                     "Resistenza": "Media"},
        "Moraiolo": {"Primavera": 1150, "Estate": 2100, "Autunno": 900, "Inverno": 480, "Temperatura Ottimale": 18,
                     "Resistenza": "Media"},
        "Pendolino": {"Primavera": 1050, "Estate": 1850, "Autunno": 850, "Inverno": 430, "Temperatura Ottimale": 20,
                      "Resistenza": "Alta"},
        "Taggiasca": {"Primavera": 1000, "Estate": 1750, "Autunno": 800, "Inverno": 400, "Temperatura Ottimale": 19,
                      "Resistenza": "Alta"},
        "Canino": {"Primavera": 1100, "Estate": 1900, "Autunno": 900, "Inverno": 450, "Temperatura Ottimale": 18,
                   "Resistenza": "Media"},
        "Itrana": {"Primavera": 1200, "Estate": 2000, "Autunno": 1000, "Inverno": 500, "Temperatura Ottimale": 17,
                   "Resistenza": "Media"},
        "Ogliarola": {"Primavera": 1150, "Estate": 1950, "Autunno": 900, "Inverno": 480, "Temperatura Ottimale": 18,
                      "Resistenza": "Media"},
        "Biancolilla": {"Primavera": 1050, "Estate": 1800, "Autunno": 850, "Inverno": 430, "Temperatura Ottimale": 19,
                        "Resistenza": "Alta"}
    }

    # Calcola fabbisogno idrico annuale
    for varieta in fabbisogno_acqua:
        fabbisogno_acqua[varieta]["Annuale"] = sum(
            fabbisogno_acqua[varieta][stagione]
            for stagione in ["Primavera", "Estate", "Autunno", "Inverno"]
        )

    # Aggiungi colonne al dataset
    dataset["Fabbisogno Acqua Primavera (m³/ettaro)"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Primavera"])
    dataset["Fabbisogno Acqua Estate (m³/ettaro)"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Estate"])
    dataset["Fabbisogno Acqua Autunno (m³/ettaro)"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Autunno"])
    dataset["Fabbisogno Acqua Inverno (m³/ettaro)"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Inverno"])
    dataset["Fabbisogno Idrico Annuale (m³/ettaro)"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Annuale"])
    dataset["Temperatura Ottimale"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Temperatura Ottimale"])
    dataset["Resistenza alla Siccità"] = dataset["Varietà di Olive"].apply(
        lambda x: fabbisogno_acqua[x]["Resistenza"])

    return dataset


def simulate_zone(base_weather: pd.DataFrame,
                  olive_varieties: pd.DataFrame,
                  year: int,
                  zone: int,
                  all_varieties: np.ndarray,
                  variety_techniques: Dict) -> Dict:
    """
    Simula la produzione di olive per una singola zona.

    Parameters
    ----------
    base_weather : pd.DataFrame
        DataFrame contenente i dati meteo di base
    olive_varieties : pd.DataFrame
        DataFrame con le informazioni sulle varietà
    year : int
        Anno della simulazione
    zone : int
        ID della zona
    all_varieties : np.ndarray
        Array con tutte le varietà disponibili
    variety_techniques : Dict
        Dizionario con le tecniche disponibili per ogni varietà

    Returns
    -------
    Dict
        Dizionario con i risultati della simulazione
    """
    # Crea una copia dei dati meteo per questa zona
    zone_weather = base_weather.copy()

    # Genera variazioni meteorologiche specifiche per questa zona
    zone_weather['temp_mean'] *= np.random.uniform(0.95, 1.05, len(zone_weather))
    zone_weather['precip_sum'] *= np.random.uniform(0.9, 1.1, len(zone_weather))
    zone_weather['solarenergy_sum'] *= np.random.uniform(0.95, 1.05, len(zone_weather))

    # Genera caratteristiche specifiche della zona
    num_varieties = np.random.randint(1, 4)  # 1-3 varietà per zona
    selected_varieties = np.random.choice(all_varieties, size=num_varieties, replace=False)
    hectares = np.random.uniform(1, 10)  # Dimensione del terreno
    percentages = np.random.dirichlet(np.ones(num_varieties))  # Distribuzione delle varietà

    # Inizializzazione contatori annuali
    annual_production = 0
    annual_min_oil = 0
    annual_max_oil = 0
    annual_avg_oil = 0
    annual_water_need = 0

    # Inizializzazione dizionario dati varietà
    variety_data = {clean_column_name(variety): {
        'tech': '',
        'pct': 0,
        'prod_t_ha': 0,
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
        'olive_prod': 0,
        'min_oil_prod': 0,
        'max_oil_prod': 0,
        'avg_oil_prod': 0,
        'water_need': 0
    } for variety in all_varieties}

    # Simula produzione per ogni varietà selezionata
    for i, variety in enumerate(selected_varieties):
        # Seleziona tecnica di coltivazione casuale per questa varietà
        technique = np.random.choice(variety_techniques[variety])
        percentage = percentages[i]

        # Ottieni informazioni specifiche della varietà
        variety_info = olive_varieties[
            (olive_varieties['Varietà di Olive'] == variety) &
            (olive_varieties['Tecnica di Coltivazione'] == technique)
            ].iloc[0]

        # Calcola produzione base con variabilità
        base_production = variety_info['Produzione (tonnellate/ettaro)'] * 1000 * percentage * hectares / 12
        base_production *= np.random.uniform(0.9, 1.1)

        # Calcola effetti meteo sulla produzione
        weather_effect = zone_weather.apply(
            lambda row: calculate_weather_effect(row, variety_info['Temperatura Ottimale']),
            axis=1
        )
        monthly_production = base_production * (1 + weather_effect / 10000)
        monthly_production *= np.random.uniform(0.95, 1.05, len(zone_weather))

        # Calcola produzione annuale per questa varietà
        annual_variety_production = monthly_production.sum()

        # Calcola rese di olio con variabilità
        min_yield_factor = np.random.uniform(0.95, 1.05)
        max_yield_factor = np.random.uniform(0.95, 1.05)
        avg_yield_factor = (min_yield_factor + max_yield_factor) / 2

        min_oil_production = annual_variety_production * variety_info[
            'Min Litri per Tonnellata'] / 1000 * min_yield_factor
        max_oil_production = annual_variety_production * variety_info[
            'Max Litri per Tonnellata'] / 1000 * max_yield_factor
        avg_oil_production = annual_variety_production * variety_info[
            'Media Litri per Tonnellata'] / 1000 * avg_yield_factor

        # Calcola fabbisogno idrico
        base_water_need = (
                                  variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Estate (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Autunno (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']
                          ) / 4

        monthly_water_need = zone_weather.apply(
            lambda row: calculate_water_need(row, base_water_need, variety_info['Temperatura Ottimale']),
            axis=1
        )
        monthly_water_need *= np.random.uniform(0.95, 1.05, len(monthly_water_need))
        annual_variety_water_need = monthly_water_need.sum() * percentage * hectares

        # Aggiorna totali annuali
        annual_production += annual_variety_production
        annual_min_oil += min_oil_production
        annual_max_oil += max_oil_production
        annual_avg_oil += avg_oil_production
        annual_water_need += annual_variety_water_need

        # Aggiorna dati varietà
        clean_variety = clean_column_name(variety)
        variety_data[clean_variety].update({
            'tech': clean_column_name(technique),
            'pct': percentage,
            'prod_t_ha': variety_info['Produzione (tonnellate/ettaro)'] * np.random.uniform(0.95, 1.05),
            'oil_prod_t_ha': variety_info['Produzione Olio (tonnellate/ettaro)'] * np.random.uniform(0.95, 1.05),
            'oil_prod_l_ha': variety_info['Produzione Olio (litri/ettaro)'] * np.random.uniform(0.95, 1.05),
            'min_yield_pct': variety_info['Min % Resa'] * min_yield_factor,
            'max_yield_pct': variety_info['Max % Resa'] * max_yield_factor,
            'min_oil_prod_l_ha': variety_info['Min Produzione Olio (litri/ettaro)'] * min_yield_factor,
            'max_oil_prod_l_ha': variety_info['Max Produzione Olio (litri/ettaro)'] * max_yield_factor,
            'avg_oil_prod_l_ha': variety_info['Media Produzione Olio (litri/ettaro)'] * avg_yield_factor,
            'l_per_t': variety_info['Litri per Tonnellata'] * np.random.uniform(0.98, 1.02),
            'min_l_per_t': variety_info['Min Litri per Tonnellata'] * min_yield_factor,
            'max_l_per_t': variety_info['Max Litri per Tonnellata'] * max_yield_factor,
            'avg_l_per_t': variety_info['Media Litri per Tonnellata'] * avg_yield_factor,
            'olive_prod': annual_variety_production,
            'min_oil_prod': min_oil_production,
            'max_oil_prod': max_oil_production,
            'avg_oil_prod': avg_oil_production,
            'water_need': annual_variety_water_need
        })

    # Appiattisci i dati delle varietà
    flattened_variety_data = {
        f'{variety}_{key}': value
        for variety, data in variety_data.items()
        for key, value in data.items()
    }

    # Restituisci il risultato della zona
    return {
        'year': year,
        'zone_id': zone + 1,
        'temp_mean': zone_weather['temp_mean'].mean(),
        'precip_sum': zone_weather['precip_sum'].sum(),
        'solar_energy_sum': zone_weather['solarenergy_sum'].sum(),
        'ha': hectares,
        'zone': f"zone_{zone + 1}",
        'olive_prod': annual_production,
        'min_oil_prod': annual_min_oil,
        'max_oil_prod': annual_max_oil,
        'avg_oil_prod': annual_avg_oil,
        'total_water_need': annual_water_need,
        **flattened_variety_data
    }