import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_training_dataset(weather_data, olive_varieties, num_simulations=1000, random_seed=42):
    """
    Genera dataset di training combinando le migliori caratteristiche di entrambi gli approcci.

    Args:
        weather_data: DataFrame con dati meteorologici
        olive_varieties: DataFrame con informazioni sulle varietà
        num_simulations: Numero di simulazioni da generare
        random_seed: Seme per riproducibilità
    """
    np.random.seed(random_seed)

    # Prepara dati meteorologici annuali
    weather_annual = weather_data.groupby('year').agg({
        'temp': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max'],
        'precip': ['sum', 'mean', 'std'],
        'solarradiation': ['mean', 'sum', 'std'],
        'cloudcover': ['mean']
    }).reset_index()

    # Appiattisci i nomi delle colonne
    weather_annual.columns = ['year'] + [
        f'{col[0]}_{col[1]}' for col in weather_annual.columns[1:]
    ]

    all_results = []
    all_varieties = olive_varieties['Varietà di Olive'].unique()

    with tqdm(total=num_simulations, desc="Generazione dataset") as pbar:
        for sim in range(num_simulations):
            # Seleziona anno base e applica variazioni
            selected_year = np.random.choice(weather_annual['year'])
            weather = weather_annual[weather_annual['year'] == selected_year].iloc[0].copy()

            # Applica variazioni meteorologiche (±20%)
            for col in weather.index:
                if col != 'year':
                    weather[col] *= np.random.uniform(0.8, 1.2)

            # Genera caratteristiche dell'oliveto
            num_varieties = np.random.randint(1, 4)  # 1-3 varietà
            selected_varieties = np.random.choice(all_varieties, size=num_varieties, replace=False)
            hectares = np.random.uniform(1, 10)
            percentages = np.random.dirichlet(np.ones(num_varieties))

            # Inizializza contatori per l'anno
            annual_results = {
                'simulation_id': sim + 1,
                'year': selected_year,
                'hectares': hectares,
                'num_varieties': num_varieties,
                'total_olive_production': 0,
                'total_oil_production': 0,
                'total_water_need': 0,
            }

            # Aggiungi dati meteorologici
            for col in weather.index:
                if col != 'year':
                    annual_results[f'weather_{col}'] = weather[col]

            # Simula per ogni varietà
            variety_details = []
            for i, variety in enumerate(selected_varieties):
                # Seleziona tecnica di coltivazione
                variety_data = olive_varieties[olive_varieties['Varietà di Olive'] == variety]
                technique = np.random.choice(variety_data['Tecnica di Coltivazione'].unique())
                percentage = percentages[i]

                # Ottieni dati specifici varietà
                variety_info = variety_data[
                    variety_data['Tecnica di Coltivazione'] == technique
                    ].iloc[0]

                # Calcola produzione base con variabilità
                base_variation = np.random.uniform(0.8, 1.2)
                base_production = variety_info['Produzione (tonnellate/ettaro)'] * base_variation

                # Applica effetti meteorologici
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

                # Calcola produzione effettiva
                actual_production = (
                        base_production *
                        temp_effect *
                        water_effect *
                        solar_effect *
                        percentage *
                        hectares
                )

                # Calcola resa olio con variabilità
                oil_yield = np.random.uniform(
                    variety_info['Min % Resa'],
                    variety_info['Max % Resa']
                )
                oil_production = actual_production * oil_yield

                # Calcola fabbisogno idrico
                base_water_need = (
                                          variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'] +
                                          variety_info['Fabbisogno Acqua Estate (m³/ettaro)'] +
                                          variety_info['Fabbisogno Acqua Autunno (m³/ettaro)'] +
                                          variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']
                                  ) / 4

                # Adatta fabbisogno idrico alle condizioni
                actual_water_need = (
                        base_water_need *
                        (1 + max(0, (weather['temp_mean'] - 20) / 50)) *
                        max(0.6, 1 - (weather['precip_sum'] / 1000)) *
                        percentage *
                        hectares
                )

                # Salva dettagli varietà
                variety_details.append({
                    'variety': variety,
                    'technique': technique,
                    'percentage': percentage,
                    'production': actual_production,
                    'oil_production': oil_production,
                    'water_need': actual_water_need,
                    'yield': oil_yield,
                    'base_production': base_production,
                    'temp_effect': temp_effect,
                    'water_effect': water_effect,
                    'solar_effect': solar_effect
                })

                # Aggiorna totali annuali
                annual_results['total_olive_production'] += actual_production
                annual_results['total_oil_production'] += oil_production
                annual_results['total_water_need'] += actual_water_need

            # Calcola metriche per ettaro
            annual_results['olive_production_ha'] = annual_results['total_olive_production'] / hectares
            annual_results['oil_production_ha'] = annual_results['total_oil_production'] / hectares
            annual_results['water_need_ha'] = annual_results['total_water_need'] / hectares

            # Aggiungi KPI di efficienza
            annual_results['yield_efficiency'] = annual_results['total_oil_production'] / annual_results[
                'total_olive_production']
            annual_results['water_efficiency'] = annual_results['total_olive_production'] / annual_results[
                'total_water_need']

            # Aggiungi dettagli varietà al risultato
            for i, detail in enumerate(variety_details):
                prefix = f'variety_{i + 1}'
                for key, value in detail.items():
                    annual_results[f'{prefix}_{key}'] = value

            all_results.append(annual_results)
            pbar.update(1)

    # Crea DataFrame finale
    df = pd.DataFrame(all_results)

    return df


def calculate_temperature_effect(temp, optimal_temp):
    """Calcola effetto temperatura con variabilità"""
    temp_diff = abs(temp - optimal_temp)
    if temp_diff <= 5:
        return np.random.uniform(0.95, 1.0)
    elif temp_diff <= 10:
        return np.random.uniform(0.8, 0.9)
    else:
        return np.random.uniform(0.6, 0.8)


def calculate_water_effect(precip, drought_resistance):
    """Calcola effetto precipitazioni con variabilità"""
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
    """Calcola effetto radiazione solare con variabilità"""
    if radiation >= 200:
        return np.random.uniform(0.95, 1.0)
    else:
        base_factor = max(0.7, radiation / 200)
        return base_factor * np.random.uniform(0.8, 1.2)


# Test del codice
if __name__ == "__main__":
    print("Generazione dataset di training...")

    # Carica dati
    weather_data = pd.read_parquet('./sources/weather_data_complete.parquet')
    olive_varieties = pd.read_parquet('./sources/olive_varieties.parquet')

    # Genera dataset
    df = generate_training_dataset(weather_data, olive_varieties, 100000)

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

    # Salva dataset
    df.to_parquet('./sources/olive_training_dataset.parquet', index=False)
    print("\nDataset salvato come 'olive_training_dataset.parquet'")