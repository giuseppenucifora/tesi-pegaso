import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil
from tqdm import tqdm
import os
import argparse
import sys
import gc
from utils.helpers import clean_column_name, get_growth_phase, calculate_weather_effect, calculate_water_need, \
    create_technique_mapping, preprocess_weather_data


def get_optimal_workers():
    """Calcola il numero ottimale di workers basato sulle risorse del sistema"""
    cpu_count = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024 ** 3)
    memory_per_worker_gb = 2

    max_workers_by_memory = int(available_memory_gb / memory_per_worker_gb)
    optimal_workers = min(
        cpu_count - 1,
        max_workers_by_memory,
        32
    )
    print(f'CPU count : {cpu_count} - Memory : {memory} = Max Worker by memory : {max_workers_by_memory}')

    return max(1, optimal_workers)


def simulate_zone(base_weather, olive_varieties, year, zone, all_varieties, variety_techniques):
    """
    Simula la produzione di olive per una singola zona.

    Args:
        base_weather: DataFrame con dati meteo di base per l'anno selezionato
        olive_varieties: DataFrame con le informazioni sulle varietà di olive
        zone: ID della zona
        all_varieties: Array con tutte le varietà disponibili
        variety_techniques: Dict con le tecniche disponibili per ogni varietà

    Returns:
        Dict con i risultati della simulazione per la zona
    """
    # Crea una copia dei dati meteo per questa zona specifica
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


def simulate_olive_production_parallel(weather_data, olive_varieties, num_simulations=5, num_zones=None,
                                       random_seed=None,
                                       max_workers=None, batch_size=500,
                                       output_path='olive_simulation_dataset.parquet'):
    """
    Versione corretta della simulazione parallelizzata con gestione batch e salvataggio file

    Args:
        weather_data: DataFrame con dati meteo
        olive_varieties: DataFrame con varietà di olive
        num_simulations: numero di simulazioni da eseguire (default: 5)
        num_zones: numero di zone per simulazione (default: None, usa num_simulations se non specificato)
        random_seed: seed per riproducibilità (default: None)
        max_workers: numero massimo di workers (default: None, usa get_optimal_workers)
        batch_size: dimensione del batch per gestione memoria (default: 500)
        output_path: percorso del file di output (default: 'olive_simulation_dataset.parquet')

    Returns:
        DataFrame con i risultati delle simulazioni
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Se num_zones non è specificato, usa num_simulations
    if num_zones is None:
        num_zones = num_simulations

    # Preparazione dati
    create_technique_mapping(olive_varieties)
    monthly_weather = preprocess_weather_data(weather_data)
    all_varieties = olive_varieties['Varietà di Olive'].unique()
    variety_techniques = {
        variety: olive_varieties[olive_varieties['Varietà di Olive'] == variety]['Tecnica di Coltivazione'].unique()
        for variety in all_varieties
    }

    # Calcolo workers ottimali usando get_optimal_workers
    if max_workers is None:
        max_workers = get_optimal_workers()
        print(f"Utilizzando {max_workers} workers ottimali basati sulle risorse del sistema")

    # Calcolo numero di batch
    num_batches = (num_simulations + batch_size - 1) // batch_size
    print(f"Elaborazione di {num_simulations} simulazioni con {num_zones} zone in {num_batches} batch")
    print(f"Totale record attesi: {num_simulations * num_zones:,}")

    # Lista per contenere tutti i DataFrame dei batch
    all_batches = []

    # Elaborazione per batch
    for batch_num in range(num_batches):
        start_sim = batch_num * batch_size
        end_sim = min((batch_num + 1) * batch_size, num_simulations)
        current_batch_size = end_sim - start_sim

        batch_results = []

        # Parallelizzazione usando ProcessPoolExecutor per il batch corrente
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Calcola il numero totale di task per questo batch
            # Ogni simulazione nel batch corrente genererà num_zones zone
            total_tasks = current_batch_size * num_zones

            with tqdm(total=total_tasks,
                      desc=f"Batch {batch_num + 1}/{num_batches}") as pbar:
                # Dizionario per tenere traccia delle futures e dei loro sim_id
                future_to_sim_id = {}

                # Sottometti i lavori per tutte le simulazioni e zone nel batch corrente
                for sim in range(start_sim, end_sim):
                    selected_year = np.random.choice(monthly_weather['year'].unique())
                    base_weather = monthly_weather[monthly_weather['year'] == selected_year].copy()
                    base_weather.loc[:, 'growth_phase'] = base_weather['month'].apply(get_growth_phase)

                    # Sottometti i lavori per tutte le zone di questa simulazione
                    for zone in range(num_zones):
                        future = executor.submit(
                            simulate_zone,
                            base_weather=base_weather,
                            olive_varieties=olive_varieties,
                            year=selected_year,
                            zone=zone,
                            all_varieties=all_varieties,
                            variety_techniques=variety_techniques
                        )
                        future_to_sim_id[future] = (sim + 1, zone + 1)

                # Raccogli i risultati man mano che vengono completati
                for future in as_completed(future_to_sim_id.keys()):
                    sim_id, zone_id = future_to_sim_id[future]
                    try:
                        result = future.result()
                        result['simulation_id'] = sim_id
                        result['zone_id'] = zone_id
                        batch_results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Errore nella simulazione {sim_id}, zona {zone_id}: {str(e)}")
                        continue

        # Converti batch_results in DataFrame e aggiungi alla lista dei batch
        batch_df = pd.DataFrame(batch_results)
        all_batches.append(batch_df)

        # Stampa statistiche del batch
        print(f"\nStatistiche Batch {batch_num + 1}:")
        print(f"Righe processate: {len(batch_df):,}")
        print(f"Memoria utilizzata: {batch_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Libera memoria
        del batch_results
        del batch_df
        gc.collect()  # Forza garbage collection

    # Concatena tutti i batch e salva
    print("\nConcatenazione dei batch e salvataggio...")
    final_df = pd.concat(all_batches, ignore_index=True)

    # Crea directory output se necessario
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Salva il dataset
    final_df.to_parquet(output_path)

    # Stampa statistiche finali
    print("\nStatistiche Finali:")
    print(f"Totale simulazioni completate: {len(final_df):,}")
    print(f"Memoria totale utilizzata: {final_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"\nDataset salvato in: {output_path}")

    return final_df


def calculate_production(variety_info, weather, percentage, hectares, seed):
    """Calcola produzione e parametri correlati per una varietà"""
    np.random.seed(seed)

    base_production = variety_info['Produzione (tonnellate/ettaro)'] * percentage * hectares
    base_production *= np.random.uniform(0.8, 1.2)

    # Effetti ambientali
    temp_effect = calculate_temperature_effect(
        weather['temp_mean'],
        variety_info['Temperatura Ottimale']
    )
    water_effect = calculate_water_effect(
        weather['precip_sum'],
        variety_info['Resistenza alla Siccità']
    )
    solar_effect = calculate_solar_effect(
        weather['solarradiation_mean']
    )

    actual_production = base_production * temp_effect * water_effect * solar_effect

    # Calcolo olio
    oil_yield = np.random.uniform(
        variety_info['Min % Resa'],
        variety_info['Max % Resa']
    )
    oil_production = actual_production * oil_yield

    # Calcolo acqua
    base_water_need = (
                              variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'] +
                              variety_info['Fabbisogno Acqua Estate (m³/ettaro)'] +
                              variety_info['Fabbisogno Acqua Autunno (m³/ettaro)'] +
                              variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']
                      ) / 4 * percentage * hectares

    water_need = (
            base_water_need *
            (1 + max(0, (weather['temp_mean'] - 20) / 50)) *
            max(0.6, 1 - (weather['precip_sum'] / 1000))
    )

    return {
        'variety': variety_info['Varietà di Olive'],
        'technique': variety_info['Tecnica di Coltivazione'],
        'percentage': percentage,
        'production': actual_production,
        'oil_production': oil_production,
        'water_need': water_need,
        'temp_effect': temp_effect,
        'water_effect': water_effect,
        'solar_effect': solar_effect,
        'yield': oil_yield
    }


# Funzioni di effetto ambientale rimangono invariate
def calculate_temperature_effect(temp, optimal_temp):
    temp_diff = abs(temp - optimal_temp)
    if temp_diff <= 5:
        return np.random.uniform(0.95, 1.0)
    elif temp_diff <= 10:
        return np.random.uniform(0.8, 0.9)
    else:
        return np.random.uniform(0.6, 0.8)


def calculate_water_effect(precip, drought_resistance):
    if 'alta' in str(drought_resistance).lower():
        min_precip = 20
    elif 'media' in str(drought_resistance).lower():
        min_precip = 30
    else:
        min_precip = 40

    if precip >= min_precip:
        return np.random.uniform(0.95, 1.0)
    else:
        base_factor = max(0.6, precip / min_precip)
        return base_factor * np.random.uniform(0.8, 1.2)


def calculate_solar_effect(radiation):
    if radiation >= 200:
        return np.random.uniform(0.95, 1.0)
    else:
        base_factor = max(0.7, radiation / 200)
        return base_factor * np.random.uniform(0.8, 1.2)


def parse_arguments():
    """
    Configura e gestisce i parametri da riga di comando
    """
    parser = argparse.ArgumentParser(
        description='Generatore dataset di training per produzione olive',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Mostra i valori default nell'help
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Seed per la riproducibilità dei risultati'
    )

    parser.add_argument(
        '--num-simulations',
        type=int,
        default=100000,
        help='Numero totale di simulazioni da eseguire'
    )

    parser.add_argument(
        '--num-zones',
        type=int,
        default=None,
        help='Numero di zone per simulazione (default: uguale a num-simulations)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Dimensione di ogni batch di simulazioni'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='./sources/olive_training_dataset.parquet',
        help='Percorso del file di output'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Quantità di workers (default: usa get_optimal_workers)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Generazione dataset di training...")

    # Parsing argomenti
    args = parse_arguments()

    # Carica dati
    try:
        weather_data = pd.read_parquet('./sources/weather_data_complete.parquet')
        olive_varieties = pd.read_parquet('./sources/olive_varieties.parquet')
    except Exception as e:
        print(f"Errore nel caricamento dei dati: {str(e)}")
        sys.exit(1)

    # Stampa configurazione
    print("\nConfigurazione:")
    print(f"Random seed: {args.random_seed}")
    print(f"Numero simulazioni: {args.num_simulations:,}")
    print(f"Numero zone per simulazione: {args.num_zones if args.num_zones is not None else args.num_simulations:,}")
    print(f"Workers: {args.max_workers if args.max_workers is not None else 'auto'}")
    print(f"Dimensione batch: {args.batch_size:,}")
    print(f"File output: {args.output_path}")

    # Genera dataset
    try:
        df = simulate_olive_production_parallel(
            weather_data=weather_data,
            olive_varieties=olive_varieties,
            num_simulations=args.num_simulations,
            num_zones=args.num_zones,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            output_path=args.output_path,
            max_workers=args.max_workers
        )
    except Exception as e:
        print(f"Errore durante la generazione del dataset: {str(e)}")
        sys.exit(1)