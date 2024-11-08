import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil
from tqdm import tqdm
import os
import argparse
import sys

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


def simulate_single_year(params):
    """
    Simula un singolo anno di produzione.

    Args:
        params: dict contenente:
            - weather_annual: dati meteo annuali
            - olive_varieties: informazioni sulle varietà
            - sim_id: ID simulazione
            - random_seed: seed per riproducibilità
    """
    np.random.seed(params['random_seed'] + params['sim_id'])

    # Seleziona anno base e applica variazioni
    weather = params['weather_annual'].sample(n=1, random_state=params['random_seed'] + params['sim_id']).iloc[0].copy()

    # Applica variazioni meteorologiche (±20%)
    for col in weather.index:
        if col != 'year':
            weather[col] *= np.random.uniform(0.8, 1.2)

    # Genera caratteristiche dell'oliveto
    num_varieties = np.random.randint(1, 4)
    selected_varieties = np.random.choice(
        params['olive_varieties']['Varietà di Olive'].unique(),
        size=num_varieties,
        replace=False
    )
    hectares = np.random.uniform(1, 10)
    percentages = np.random.dirichlet(np.ones(num_varieties))

    annual_results = {
        'simulation_id': params['sim_id'],
        'year': weather['year'],
        'hectares': hectares,
        'num_varieties': num_varieties,
        'total_olive_production': 0,
        'total_oil_production': 0,
        'total_water_need': 0
    }

    # Aggiungi dati meteorologici
    for col in weather.index:
        if col != 'year':
            annual_results[f'weather_{col}'] = weather[col]

    variety_details = []
    for i, variety in enumerate(selected_varieties):
        variety_data = params['olive_varieties'][
            params['olive_varieties']['Varietà di Olive'] == variety
            ]
        technique = np.random.choice(variety_data['Tecnica di Coltivazione'].unique())
        percentage = percentages[i]

        variety_info = variety_data[
            variety_data['Tecnica di Coltivazione'] == technique
            ].iloc[0]

        # Calcoli produzione con variabilità
        production_data = calculate_production(
            variety_info, weather, percentage, hectares,
            params['sim_id'] + i
        )

        variety_details.append(production_data)

        # Aggiorna totali
        annual_results['total_olive_production'] += production_data['production']
        annual_results['total_oil_production'] += production_data['oil_production']
        annual_results['total_water_need'] += production_data['water_need']

    # Aggiungi dettagli varietà
    for i, detail in enumerate(variety_details):
        prefix = f'variety_{i + 1}'
        for key, value in detail.items():
            annual_results[f'{prefix}_{key}'] = value

    # Calcola metriche per ettaro e KPI
    annual_results['olive_production_ha'] = annual_results['total_olive_production'] / hectares
    annual_results['oil_production_ha'] = annual_results['total_oil_production'] / hectares
    annual_results['water_need_ha'] = annual_results['total_water_need'] / hectares

    # Calcola efficienze
    if annual_results['total_olive_production'] > 0:
        annual_results['yield_efficiency'] = annual_results['total_oil_production'] / annual_results[
            'total_olive_production']
    else:
        annual_results['yield_efficiency'] = 0

    if annual_results['total_water_need'] > 0:
        annual_results['water_efficiency'] = annual_results['total_olive_production'] / annual_results[
            'total_water_need']
    else:
        annual_results['water_efficiency'] = 0

    return annual_results


def generate_training_dataset_parallel(weather_data, olive_varieties, num_simulations=1000,
                                       random_seed=42, max_workers=None, batch_size=500,
                                       output_path='olive_training_dataset.parquet'):
    """
    Genera dataset di training utilizzando multiprocessing.

    Args:
        weather_data: DataFrame dati meteo
        olive_varieties: DataFrame varietà olive
        num_simulations: numero di simulazioni
        random_seed: seed per riproducibilità
        max_workers: numero massimo di workers
        batch_size: dimensione batch
        output_path: percorso file output
    """
    np.random.seed(random_seed)

    # Prepara dati meteo annuali
    weather_annual = weather_data.groupby('year').agg({
        'temp': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max'],
        'precip': ['sum', 'mean', 'std'],
        'solarradiation': ['mean', 'sum', 'std'],
        'cloudcover': ['mean']
    }).reset_index()

    weather_annual.columns = ['year'] + [
        f'{col[0]}_{col[1]}' for col in weather_annual.columns[1:]
    ]

    # Calcola workers ottimali
    if max_workers is None:
        max_workers = get_optimal_workers()

    print(f"Utilizzando {max_workers} workers")

    # Calcola numero di batch
    num_batches = (num_simulations + batch_size - 1) // batch_size
    print(f"Elaborazione di {num_simulations} simulazioni in {num_batches} batch")

    # Crea directory output se necessario
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Lista per contenere tutti i DataFrame dei batch
    all_batches = []

    for batch_num in range(num_batches):
        start_sim = batch_num * batch_size
        end_sim = min((batch_num + 1) * batch_size, num_simulations)
        current_batch_size = end_sim - start_sim

        batch_results = []

        # Preparazione parametri per ogni simulazione
        simulation_params = [
            {
                'weather_annual': weather_annual,
                'olive_varieties': olive_varieties,
                'sim_id': sim_id,
                'random_seed': random_seed
            }
            for sim_id in range(start_sim, end_sim)
        ]

        # Esegui simulazioni in parallelo
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(simulate_single_year, params)
                       for params in simulation_params]

            with tqdm(total=current_batch_size,
                      desc=f"Batch {batch_num + 1}/{num_batches}") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Errore in simulazione: {str(e)}")
                        continue

        # Converti risultati in DataFrame
        batch_df = pd.DataFrame(batch_results)
        all_batches.append(batch_df)

        # Libera memoria
        del batch_results

    # Concatena tutti i batch e salva
    final_df = pd.concat(all_batches, ignore_index=True)
    final_df.to_parquet(output_path)

    print(f"\nDataset salvato in: {output_path}")

    # Statistiche finali
    print("\nStatistiche finali:")
    print(f"Righe totali: {len(final_df)}")
    print("\nAnalisi variabilità:")
    for col in ['olive_production_ha', 'oil_production_ha', 'water_need_ha']:
        cv = final_df[col].std() / final_df[col].mean()
        print(f"{col}: CV = {cv:.2%}")

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
        default=1000000,
        help='Numero totale di simulazioni da eseguire'
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
        default=2,
        help='Quantità di workers'
    )

    return parser.parse_args()


# Esempio di utilizzo
if __name__ == "__main__":
    print("Generazione dataset di training...")



    # Parsing argomenti
    args = parse_arguments()

    # Carica dati
    try:
        # Carica dati
        weather_data = pd.read_parquet('./sources/weather_data_complete.parquet')
        olive_varieties = pd.read_parquet('./sources/olive_varieties.parquet')
    except Exception as e:
        print(f"Errore nel caricamento dei dati: {str(e)}")
        sys.exit(1)

    # Stampa configurazione
    print("\nConfigurazione:")
    print(f"Random seed: {args.random_seed}")
    print(f"Numero simulazioni: {args.num_simulations:,}")
    print(f"Workers: {args.max_workers:,}")
    print(f"Dimensione batch: {args.batch_size:,}")
    print(f"File output: {args.output_path}")

    # Genera dataset
    try:
        df = generate_training_dataset_parallel(
            weather_data=weather_data,
            olive_varieties=olive_varieties,
            random_seed=args.random_seed,
            num_simulations=args.num_simulations,
            batch_size=args.batch_size,
            output_path=args.output_path,
            max_workers=args.max_workers
        )
    except Exception as e:
        print(f"Errore durante la generazione del dataset: {str(e)}")
        sys.exit(1)

    print("\nShape dataset:", df.shape)
    print("\nColonne disponibili:")
    print(df.columns.tolist())

    print("\nStatistiche di base:")
    print(df.describe())

    # Analisi variabilità
    print("\nAnalisi coefficienti di variazione:")
    for col in ['olive_production_ha', 'oil_production_ha', 'water_need_ha']:
        cv = df[col].std() / df[col].mean()
        print(f"{col}: {cv:.2%}")

    print("\nDataset salvato './sources/olive_training_dataset.parquet'")