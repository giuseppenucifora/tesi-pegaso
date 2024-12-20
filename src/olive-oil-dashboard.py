import flask
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import tensorflow as tf
import keras
import joblib
import dash_bootstrap_components as dbc
import os
import argparse
import json
from dash.exceptions import PreventUpdate

from auth import utils
from utils.helpers import clean_column_name
from dashboard.environmental_simulator import *
from dash import no_update
from auth.utils import (
    init_directory_structure, verify_user, create_token,
    verify_token, create_user, get_user_config_path, get_default_config
)
from auth.login import create_login_layout, create_register_layout
from components.ids import Ids

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set global precision policy
tf.keras.mixed_precision.set_global_policy('float32')

DEV_MODE = True
model = None
scaler_temporal = None
scaler_static = None
scaler_y = None
MODEL_LOADING = False


def load_config():
    try:
        config = None
        # Prova a leggere la sessione corrente
        session_data = check_session()
        if session_data:
            username = session_data.get('username')
            if username:
                config_path = get_user_config_path(username)
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print(f'Loaded configuration for user: {username}')

        # Se non c'è config utente, usa quella di default
        if config is None:
            config = get_default_config()
            print('Using default configuration')

        return config
    except Exception as e:
        print(f"Errore nel caricamento della configurazione: {e}")
        return get_default_config()


def save_config(config):
    """
    Salva la configurazione nel file di configurazione
    Returns: (success: bool, message: str)
    """
    try:
        config_path = None

        # Determina il percorso del file di configurazione
        if flask.has_request_context():
            try:
                session_data = check_session()
                if session_data and 'username' in session_data:
                    username = session_data['username']
                    config_path = get_user_config_path(username)
                    print(f'Using configuration path for user {username}: {config_path}')
            except Exception as e:
                print(f"Error accessing session: {e}")
                import traceback
                traceback.print_exc()

        # Se non abbiamo un percorso utente specifico
        if config_path is None:
            print("WARNING: No user found in session!")
            return False, "Nessun utente trovato nella sessione. Effettua nuovamente il login."

        # Assicurati che la directory esista
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Verifica che la configurazione sia valida
        if not isinstance(config, dict):
            return False, "Configurazione non valida"

        # Salva la configurazione
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        return True, f"Configurazione salvata con successo in {config_path}"

    except Exception as e:
        print(f"Errore nel salvataggio della configurazione: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Errore nel salvataggio: {str(e)}"


try:
    print(f"Caricamento dataset e scaler...")

    simulated_data = pd.read_parquet("./sources/olive_training_dataset.parquet")
    weather_data = pd.read_parquet("./sources/weather_data_solarenergy.parquet")
    olive_varieties = pd.read_parquet("./sources/olive_varieties.parquet")
    scaler_temporal = joblib.load('./sources/olive_oil_transformer/olive_oil_transformer_scaler_temporal.joblib')
    scaler_static = joblib.load('./sources/olive_oil_transformer/olive_oil_transformer_scaler_static.joblib')
    scaler_y = joblib.load('./sources/olive_oil_transformer/olive_oil_transformer_scaler_y.joblib')
except Exception as e:
    print(f"Errore nel caricamento: {str(e)}")
    raise e


def prepare_static_features_multiple(varieties_info, percentages, hectares, all_varieties):
    """
    Prepara le feature statiche per multiple varietà seguendo la struttura esatta della simulazione.

    Args:
    varieties_info (list): Lista di dizionari contenenti le informazioni sulle varietà selezionate
    percentages (list): Lista delle percentuali corrispondenti a ciascuna varietà selezionata
    hectares (float): Numero di ettari totali
    all_varieties (list): Lista di tutte le possibili varietà nel dataset originale

    Returns:
    np.array: Array numpy contenente tutte le feature statiche
    """
    # Inizializza un dizionario per tutte le varietà possibili
    variety_data = {variety: {
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
        'tech': ''
    } for variety in all_varieties}

    # Aggiorna i dati per le varietà selezionate
    for variety_info, percentage in zip(varieties_info, percentages):
        variety_name = clean_column_name(variety_info['Varietà di Olive'])
        technique = clean_column_name(variety_info['Tecnica di Coltivazione'])

        # Base production calculations
        annual_prod = variety_info['Produzione (tonnellate/ettaro)'] * 1000 * percentage / 100 * hectares
        min_oil_prod = annual_prod * variety_info['Min Litri per Tonnellata'] / 1000
        max_oil_prod = annual_prod * variety_info['Max Litri per Tonnellata'] / 1000
        avg_oil_prod = annual_prod * variety_info['Media Litri per Tonnellata'] / 1000

        # Water need calculation
        base_water_need = (
                                  variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Estate (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Autunno (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']
                          ) / 12 * percentage / 100 * hectares

        variety_data[variety_name].update({
            'pct': percentage / 100,
            'prod_t_ha': variety_info['Produzione (tonnellate/ettaro)'],
            'oil_prod_t_ha': variety_info['Produzione Olio (tonnellate/ettaro)'],
            'oil_prod_l_ha': variety_info['Produzione Olio (litri/ettaro)'],
            'min_yield_pct': variety_info['Min % Resa'],
            'max_yield_pct': variety_info['Max % Resa'],
            'min_oil_prod_l_ha': variety_info['Min Produzione Olio (litri/ettaro)'],
            'max_oil_prod_l_ha': variety_info['Max Produzione Olio (litri/ettaro)'],
            'avg_oil_prod_l_ha': variety_info['Media Produzione Olio (litri/ettaro)'],
            'l_per_t': variety_info['Litri per Tonnellata'],
            'min_l_per_t': variety_info['Min Litri per Tonnellata'],
            'max_l_per_t': variety_info['Max Litri per Tonnellata'],
            'avg_l_per_t': variety_info['Media Litri per Tonnellata'],
            'tech': technique
        })

    # Crea il vettore delle feature nell'ordine esatto
    static_features = [hectares]  # Inizia con gli ettari

    # Lista delle feature per ogni varietà
    variety_features = ['pct', 'prod_t_ha', 'oil_prod_t_ha', 'oil_prod_l_ha', 'min_yield_pct', 'max_yield_pct',
                        'min_oil_prod_l_ha', 'max_oil_prod_l_ha', 'avg_oil_prod_l_ha', 'l_per_t', 'min_l_per_t',
                        'max_l_per_t', 'avg_l_per_t']

    # Appiattisci i dati delle varietà mantenendo l'ordine esatto
    for variety in all_varieties:
        # Feature esistenti
        for feature in variety_features:
            static_features.append(variety_data[variety][feature])

        # Feature binarie per le tecniche di coltivazione
        for technique in ['tradizionale', 'intensiva', 'superintensiva']:
            static_features.append(1 if variety_data[variety]['tech'] == technique else 0)

    print(f"lunghezza features {len(static_features)} ")

    return np.array(static_features).reshape(1, -1)


def mock_make_prediction(weather_data, varieties_info, percentages, hectares, simulation_data=None):
    try:
        # Debug info
        print("Inizio mock_make_prediction")
        print(f"Varietà info: {len(varieties_info)}")
        print(f"Percentuali: {percentages}")
        print(f"Ettari: {hectares}")

        # Inizializza le variabili ambientali dai dati storici
        avg_temp = weather_data['temp'].mean()
        avg_radiation = weather_data['solarradiation'].mean()
        stress_factor = 1.0

        if simulation_data is not None:
            print("Usando dati dalla simulazione ambientale")
            avg_stress = simulation_data['stress_index'].mean()
            stress_factor = 1.0 - (avg_stress * 0.5)  # Riduce fino al 50%
            print(f"Fattore di stress dalla simulazione: {stress_factor:.2f}")

            # Aggiorna valori ambientali
            avg_temp = simulation_data['temperature'].mean()
            if 'radiation' in simulation_data.columns:
                avg_radiation = simulation_data['radiation'].mean()
        else:
            print("Usando dati meteorologici storici")
            # Calcola stress base dai dati storici
            recent_weather = weather_data.tail(3)
            avg_temp = recent_weather['temp'].mean()

            if avg_temp < 15:
                stress_factor *= 0.9
            elif avg_temp > 25:
                stress_factor *= 0.95

            # Precipitazioni influenzano la produzione
            total_precip = recent_weather['precip'].sum()
            if total_precip < 30:  # Siccità
                stress_factor *= 0.85
            elif total_precip > 200:  # Troppa pioggia
                stress_factor *= 0.9

            # Radiazione solare
            avg_solar = recent_weather['solarradiation'].mean()
            if avg_solar < 150:
                stress_factor *= 0.95

        print(f"Fattore di stress finale: {stress_factor}")

        # Calcola la produzione di olive basata sui dati reali delle varietà
        current_month = datetime.now().month
        seasons = {
            'Primavera': [3, 4, 5],
            'Estate': [6, 7, 8],
            'Autunno': [9, 10, 11],
            'Inverno': [12, 1, 2]
        }
        current_season = next(season for season, months in seasons.items()
                              if current_month in months)

        season_water_need = {
            'Primavera': 'Fabbisogno Acqua Primavera (m³/ettaro)',
            'Estate': 'Fabbisogno Acqua Estate (m³/ettaro)',
            'Autunno': 'Fabbisogno Acqua Autunno (m³/ettaro)',
            'Inverno': 'Fabbisogno Acqua Inverno (m³/ettaro)'
        }

        # Calcoli per ogni varietà
        variety_details = []
        total_water_need = 0
        total_olive_production = 0
        total_oil_production = 0

        for variety_info, percentage in zip(varieties_info, percentages):
            # print(f"Elaborazione varietà: {variety_info['Varietà di Olive']}")

            # Calcola la produzione di olive per ettaro
            base_prod_per_ha = float(variety_info['Produzione (tonnellate/ettaro)']) * 1000 * (percentage / 100)
            prod_per_ha = base_prod_per_ha * stress_factor
            prod_total = prod_per_ha * hectares

            # Calcola la produzione di olio
            base_oil_per_ha = float(variety_info['Produzione Olio (litri/ettaro)']) * (percentage / 100)
            oil_per_ha = base_oil_per_ha * stress_factor
            oil_total = oil_per_ha * hectares

            # Calcolo fabbisogno idrico
            water_need = float(variety_info[season_water_need[current_season]]) * (percentage / 100)

            # print(f"  Produzione olive/ha: {prod_per_ha:.2f}")
            # print(f"  Produzione olio/ha: {oil_per_ha:.2f}")
            # print(f"  Fabbisogno idrico: {water_need:.2f}")

            variety_details.append({
                'variety': variety_info['Varietà di Olive'],
                'percentage': percentage,
                'production_per_ha': prod_per_ha,
                'production_total': prod_total,
                'oil_per_ha': oil_per_ha,
                'oil_total': oil_total,
                'water_need': water_need
            })

            total_olive_production += prod_total
            total_oil_production += oil_total
            total_water_need += water_need * hectares

        # Calcola medie per ettaro
        avg_olive_production_ha = total_olive_production / hectares if hectares > 0 else 0
        avg_oil_production_ha = total_oil_production / hectares if hectares > 0 else 0
        water_need_ha = total_water_need / hectares if hectares > 0 else 0

        return {
            'olive_production': avg_olive_production_ha,
            'olive_production_total': total_olive_production,
            'min_oil_production': avg_oil_production_ha * 0.9,
            'max_oil_production': avg_oil_production_ha * 1.1,
            'avg_oil_production': avg_oil_production_ha,
            'avg_oil_production_total': total_oil_production,
            'water_need': water_need_ha,
            'water_need_total': total_water_need,
            'variety_details': variety_details,
            'hectares': hectares,
            'stress_factor': stress_factor,
            'environmental_conditions': {
                'temperature': avg_temp,
                'radiation': avg_radiation,
                'data_source': 'simulation' if simulation_data is not None else 'historical'
            }
        }

    except Exception as e:
        print(f"Errore nella funzione mock_make_prediction: {str(e)}")
        import traceback
        print("Traceback completo:")
        print(traceback.format_exc())


def make_prediction(weather_data, varieties_info, percentages, hectares, simulation_data=None):
    print(f"DEV_MODE: {DEV_MODE}")
    if DEV_MODE:
        return mock_make_prediction(weather_data, varieties_info, percentages, hectares, simulation_data)
    try:
        if MODEL_LOADING:
            return {
                'olive_production': 0,  # kg/ha
                'olive_production_total': 0 * hectares,  # kg totali
                'min_oil_production': 0,  # L/ha
                'max_oil_production': 0,  # L/ha
                'avg_oil_production': 0,  # L/ha
                'avg_oil_production_total': 0 * hectares,  # L totali
                'water_need': 0,  # m³/ha
                'water_need_total': 0,  # m³ totali
                'variety_details': 0,
                'hectares': hectares,
                'stress_factor': 0 if simulation_data is not None else 1.0
            }

        print("Inizio della funzione make_prediction")

        # Prepara i dati temporali (meteorologici)
        temporal_features = ['temp_mean', 'precip_sum', 'solar_energy_sum']

        if simulation_data is not None:
            # Usa i dati della simulazione
            print("Usando dati dalla simulazione ambientale")
            # Calcola le medie dai dati simulati
            temporal_data = np.array([[
                simulation_data['temperature'].mean(),
                simulation_data['rainfall'].sum(),
                simulation_data['radiation'].mean()
            ]])
        else:
            # Usa i dati meteorologici storici
            monthly_stats = weather_data.groupby(['year', 'month']).agg({
                'temp': 'mean',
                'precip': 'sum',
                'solarradiation': 'sum'
            }).reset_index()

            monthly_stats = monthly_stats.rename(columns={
                'temp': 'temp_mean',
                'precip': 'precip_sum',
                'solarradiation': 'solar_energy_sum'
            })

            # Prendi gli ultimi dati meteorologici
            temporal_data = monthly_stats[temporal_features].tail(1).values

        # Calcola il fattore di stress dalla simulazione
        stress_factor = 1.0
        if simulation_data is not None:
            avg_stress = simulation_data['stress_index'].mean()
            # Applica una penalità basata sullo stress
            stress_factor = 1.0 - (avg_stress * 0.5)  # Riduce fino al 50% basato sullo stress
            print(f"Fattore di stress dalla simulazione: {stress_factor:.2f}")

        # Prepara i dati statici
        static_data = []

        # Aggiungi hectares come prima feature statica
        static_data.append(hectares)

        # Ottieni tutte le possibili varietà dal dataset di training
        all_varieties = olive_varieties['Varietà di Olive'].unique()
        varieties = [clean_column_name(variety) for variety in all_varieties]

        # Per ogni varietà possibile nel dataset
        for variety in varieties:
            # Trova se questa varietà è tra quelle selezionate
            selected_variety = None
            selected_idx = None

            for idx, info in enumerate(varieties_info):
                if clean_column_name(info['Varietà di Olive']) == variety:
                    selected_variety = info
                    selected_idx = idx
                    break

            if selected_variety is not None:
                percentage = percentages[selected_idx] / 100  # Converti in decimale

                # Aggiungi tutte le feature numeriche della varietà
                static_data.extend([
                    percentage,  # pct
                    float(selected_variety['Produzione (tonnellate/ettaro)']),  # prod_t_ha
                    float(selected_variety['Produzione Olio (tonnellate/ettaro)']),  # oil_prod_t_ha
                    float(selected_variety['Produzione Olio (litri/ettaro)']),  # oil_prod_l_ha
                    float(selected_variety['Min % Resa']),  # min_yield_pct
                    float(selected_variety['Max % Resa']),  # max_yield_pct
                    float(selected_variety['Min Produzione Olio (litri/ettaro)']),  # min_oil_prod_l_ha
                    float(selected_variety['Max Produzione Olio (litri/ettaro)']),  # max_oil_prod_l_ha
                    float(selected_variety['Media Produzione Olio (litri/ettaro)']),  # avg_oil_prod_l_ha
                    float(selected_variety['Litri per Tonnellata']),  # l_per_t
                    float(selected_variety['Min Litri per Tonnellata']),  # min_l_per_t
                    float(selected_variety['Max Litri per Tonnellata']),  # max_l_per_t
                    float(selected_variety['Media Litri per Tonnellata'])  # avg_l_per_t
                ])

                # Aggiungi le feature binarie per la tecnica
                tech = str(selected_variety['Tecnica di Coltivazione']).lower()
                static_data.extend([
                    1 if tech == 'tradizionale' else 0,
                    1 if tech == 'intensiva' else 0,
                    1 if tech == 'superintensiva' else 0
                ])
            else:
                # Se la varietà non è selezionata, aggiungi zeri per tutte le sue feature
                static_data.extend([0] * 13)  # Feature numeriche
                static_data.extend([0] * 3)  # Feature tecniche binarie

        # Converti in array numpy e reshape
        temporal_data = np.array(temporal_data).reshape(1, 1, -1)  # (1, 1, n_temporal_features)
        static_data = np.array(static_data).reshape(1, -1)  # (1, n_static_features)

        print(f"Shape dei dati temporali: {temporal_data.shape}")
        print(f"Shape dei dati statici: {static_data.shape}")

        # Standardizza i dati
        temporal_data = scaler_temporal.transform(temporal_data.reshape(1, -1)).reshape(1, 1, -1)
        static_data = scaler_static.transform(static_data)

        # Prepara il dizionario di input per il modello
        input_data = {
            'temporal': temporal_data,
            'static': static_data
        }

        # Effettua la predizione
        prediction = model.predict(input_data)

        print("\nRaw prediction:", prediction)

        target_features = [
            'olive_prod',  # Produzione olive kg/ha
            'min_oil_prod',  # Produzione minima olio L/ha
            'max_oil_prod',  # Produzione massima olio L/ha
            'avg_oil_prod',  # Produzione media olio L/ha
            'total_water_need'  # Fabbisogno idrico totale m³/ha
        ]

        prediction = scaler_y.inverse_transform(prediction)[0]
        print("\nInverse transformed prediction:")
        for feature, value in zip(target_features, prediction):
            print(f"{feature}: {value:.2f}")

        # Applica il fattore di stress se disponibile
        if simulation_data is not None:
            prediction = prediction * stress_factor
            print(f"Applied stress factor: {stress_factor}")
            print(f"Prediction after stress:", prediction)

        prediction[4] = prediction[4] / 4  # correggo il bias creato dai dati di simulazione errati @todo nel prossimo modello addestrato con i dati corretti sarà dovrà essere rimosso

        # Calcola i valori per ettaro dividendo per il numero di ettari
        olive_prod_ha = prediction[0] / hectares
        min_oil_prod_ha = prediction[1] / hectares
        max_oil_prod_ha = prediction[2] / hectares
        avg_oil_prod_ha = prediction[3] / hectares
        water_need_ha = prediction[4] / hectares

        print("\nValori per ettaro:")
        print(f"Olive production per ha: {olive_prod_ha:.2f} kg/ha")
        print(f"Min oil production per ha: {min_oil_prod_ha:.2f} L/ha")
        print(f"Max oil production per ha: {max_oil_prod_ha:.2f} L/ha")
        print(f"Avg oil production per ha: {avg_oil_prod_ha:.2f} L/ha")
        print(f"Water need per ha: {water_need_ha:.2f} m³/ha")

        # Calcola i dettagli per varietà
        variety_details = []
        total_water_need = prediction[4]  # Usa il valore predetto totale

        for variety_info, percentage in zip(varieties_info, percentages):
            # Calcoli specifici per varietà
            base_prod_per_ha = float(variety_info['Produzione (tonnellate/ettaro)']) * 1000 * (percentage / 100)
            prod_per_ha = base_prod_per_ha
            if simulation_data is not None:
                prod_per_ha *= stress_factor
            prod_total = prod_per_ha * hectares

            base_oil_per_ha = float(variety_info['Produzione Olio (litri/ettaro)']) * (percentage / 100)
            oil_per_ha = base_oil_per_ha
            if simulation_data is not None:
                oil_per_ha *= stress_factor
            oil_total = oil_per_ha * hectares

            print(f"\nVariety: {variety_info['Varietà di Olive']}")
            print(f"Base production: {base_prod_per_ha:.2f} kg/ha")
            print(f"Final production: {prod_per_ha:.2f} kg/ha")
            print(f"Base oil: {base_oil_per_ha:.2f} L/ha")
            print(f"Final oil: {oil_per_ha:.2f} L/ha")

            variety_details.append({
                'variety': variety_info['Varietà di Olive'],
                'percentage': percentage,
                'production_per_ha': prod_per_ha,
                'production_total': prod_total,
                'oil_per_ha': oil_per_ha,
                'oil_total': oil_total,
                'water_need': water_need_ha * (percentage / 100),  # Distribuisci il fabbisogno idrico in base alla percentuale
                'base_production': base_prod_per_ha,  # Produzione senza stress
                'base_oil': base_oil_per_ha,  # Produzione olio senza stress
                'stress_factor': stress_factor if simulation_data is not None else 1.0
            })

        return {
            'olive_production': olive_prod_ha,  # kg/ha
            'olive_production_total': prediction[0],  # kg totali
            'min_oil_production': min_oil_prod_ha,  # L/ha
            'max_oil_production': max_oil_prod_ha,  # L/ha
            'avg_oil_production': avg_oil_prod_ha,  # L/ha
            'avg_oil_production_total': prediction[3],  # L totali
            'water_need': water_need_ha,  # m³/ha
            'water_need_total': prediction[4],  # m³ totali
            'variety_details': variety_details,
            'hectares': hectares,
            'stress_factor': stress_factor if simulation_data is not None else 1.0
        }

    except Exception as e:
        print(f"Errore durante la preparazione dei dati o la predizione: {str(e)}")
        print(f"Tipo di errore: {type(e).__name__}")
        import traceback
        print("Traceback completo:")
        print(traceback.format_exc())
        raise e


def create_phase_card(phase: str, data: dict) -> dbc.Card:
    """Crea una card per visualizzare i dati di una fase specifica"""
    return dbc.Card([
        dbc.CardHeader(phase),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Stress Medio", className="text-muted mb-0"),
                    html.H5(f"{data['stress_medio']:.1%}")
                ], md=6),
                dbc.Col([
                    html.P("Crescita Media", className="text-muted mb-0"),
                    html.H5(f"{data['crescita_media']:.1f}%")
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Temperatura Media", className="text-muted mb-0"),
                    html.H5(f"{data['temp_media']:.1f}°C")
                ], md=6),
                dbc.Col([
                    html.P("Durata", className="text-muted mb-0"),
                    html.H5(f"{data['durata_giorni']} giorni")
                ], md=6)
            ])
        ])
    ], className="mb-3")


def calculate_kpis(sim_data: pd.DataFrame) -> dict:
    """Calcola i KPI principali dalla simulazione"""

    # Calcolo KPI generali
    kpis = {
        'stress_medio': sim_data['stress_index'].mean(),
        'giorni_critici': len(sim_data[sim_data['stress_index'] > 0.7]),
        'fase_critica': sim_data.loc[sim_data['stress_index'].idxmax(), 'phase'],
        'crescita_media': sim_data['growth_rate'].mean(),
        'temperature_medie': sim_data['temperature'].mean(),
        'giorni_ottimali': len(sim_data[
                                   (sim_data['temperature'] >= 15) &
                                   (sim_data['temperature'] <= 25) &
                                   (sim_data['stress_index'] < 0.3)
                                   ])
    }

    # Analisi per fase
    phase_analysis = {}
    for phase in sim_data['phase'].unique():
        phase_data = sim_data[sim_data['phase'] == phase]
        phase_analysis[phase] = {
            'stress_medio': phase_data['stress_index'].mean(),
            'durata_giorni': len(phase_data),
            'crescita_media': phase_data['growth_rate'].mean(),
            'temp_media': phase_data['temperature'].mean()
        }

    kpis['analisi_fasi'] = phase_analysis

    return kpis


def create_kpi_indicators(kpis: dict) -> html.Div:
    """Crea gli indicatori visivi per i KPI"""

    def get_stress_color(value):
        if value < 0.3:
            return "success"
        elif value < 0.7:
            return "warning"
        return "danger"

    def get_growth_color(value):
        if value > 70:
            return "success"
        elif value > 40:
            return "warning"
        return "danger"

    indicators = dbc.Card([
        dbc.CardHeader("Indicatori Chiave di Performance"),
        dbc.CardBody([
            # Prima riga di KPI generali
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Stress Medio", className="text-muted h6"),
                            html.H3(f"{kpis['stress_medio']:.1%}",
                                    className=f"text-{get_stress_color(kpis['stress_medio'])}"),
                            html.P(f"Giorni critici: {kpis['giorni_critici']}",
                                   className="small text-muted")
                        ])
                    ], className="mb-3")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Crescita Media", className="text-muted h6"),
                            html.H3(f"{kpis['crescita_media']:.1f}%",
                                    className=f"text-{get_growth_color(kpis['crescita_media'])}"),
                            html.P(f"Giorni ottimali: {kpis['giorni_ottimali']}",
                                   className="small text-muted")
                        ])
                    ], className="mb-3")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Temperatura Media", className="text-muted h6"),
                            html.H3(f"{kpis['temperature_medie']:.1f}°C",
                                    className="text-primary"),
                            html.P("Range ottimale: 15-25°C",
                                   className="small text-muted")
                        ])
                    ], className="mb-3")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Fase Critica", className="text-muted h6"),
                            html.H3(f"{kpis['fase_critica']}",
                                    className="text-warning"),
                            html.P("Fase con maggiore stress",
                                   className="small text-muted")
                        ])
                    ], className="mb-3")
                ], md=3),
            ]),

            # Analisi per fase
            html.H5("Analisi per Fase di Crescita", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    create_phase_card(phase, data)
                ], md=4) for phase, data in kpis['analisi_fasi'].items()
            ])
        ])
    ])

    return indicators


server = flask.Flask(__name__)
server.secret_key = utils.SECRET_KEY

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    server=server,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    prevent_initial_callbacks='initial_duplicate',
    suppress_callback_exceptions=True
)

# Stili comuni
CARD_STYLE = {
    "height": "100%",
    "marginBottom": "15px"
}

CARD_BODY_STYLE = {
    "padding": "15px"
}


def create_production_tab():
    return dbc.Tab([
        dbc.Row([
            # Aggiungiamo un card per lo stato del modello
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Modalità Inferenza", className="text-primary mb-0"),
                    ], className="bg-light"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.P("Stato corrente:", className="mb-2"),
                                    html.H5(id=Ids.PRODUCTION_INFERENCE_MODE, className="text-info")
                                ], className="text-center")
                            ], width=8),
                            dbc.Col([
                                dbc.Switch(
                                    id=Ids.PRODUCTION_DEBUG_SWITCH,
                                    label="Modalità Debug",
                                    value=True,
                                    className="mt-2"
                                ),
                            ], width=4),
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.P("Richieste totali:", className="mb-2"),
                                html.H5(id=Ids.PRODUCTION_INFERENCE_REQUESTS, className="text-muted")
                            ], className="text-center")
                        ])
                    ])
                ], className="mb-4"),
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Simulatore Condizioni Ambientali",
                                className="text-primary mb-0")
                    ]),
                    dbc.CardBody([
                        # Controlli simulazione
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Temperatura (°C)"),
                                dcc.RangeSlider(
                                    id='temp-slider',
                                    min=0,
                                    max=40,
                                    step=0.5,
                                    value=[15, 25],
                                    marks={i: f'{i}°C' for i in range(0, 41, 5)}
                                ),
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Umidità (%)"),
                                dcc.Slider(
                                    id='humidity-slider',
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=60,
                                    marks={i: f'{i}%' for i in range(0, 101, 10)}
                                ),
                            ], md=6),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Precipitazioni (mm/mese)"),
                                dbc.Input(
                                    id='rainfall-input',
                                    type='number',
                                    value=50,
                                    min=0,
                                    max=500,
                                    className="mb-2"
                                ),
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Radiazione Solare (W/m²)"),
                                dbc.Input(
                                    id='radiation-input',
                                    type='number',
                                    value=250,
                                    min=0,
                                    max=1000,
                                    className="mb-2"
                                ),
                            ], md=6),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-play me-2"),
                                        "Avvia Simulazione"
                                    ],
                                    id="simulate-btn",
                                    color="primary",
                                    className="w-100"
                                ),
                            ], md=12),
                        ]),
                    ]),
                ], className="mb-4"),
            ], md=12),
        ]),
        # Metriche principali
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Produzione Olive/ha"),
                    dbc.CardBody([
                        html.H3(
                            id='olive-production_ha',
                            className="text-center text-primary"
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Produzione Olio/ha"),
                    dbc.CardBody([
                        html.H3(
                            id='oil-production_ha',
                            className="text-center text-success"
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fabbisogno Idrico/ha"),
                    dbc.CardBody([
                        html.H3(
                            id='water-need_ha',
                            className="text-center text-info"
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=4),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Produzione Olive"),
                    dbc.CardBody([
                        html.H3(
                            id='olive-production',
                            className="text-center text-primary"
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Produzione Olio"),
                    dbc.CardBody([
                        html.H3(
                            id='oil-production',
                            className="text-center text-success"
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fabbisogno Idrico"),
                    dbc.CardBody([
                        html.H3(
                            id='water-need',
                            className="text-center text-info"
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=4),
        ]),

        # Grafici
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Dettagli Produzione"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='production-details',
                            config={'displayModeBar': False}
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analisi Meteorologica"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='weather-impact',
                            config={'displayModeBar': False}
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fabbisogno Idrico Mensile"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='water-needs',
                            config={'displayModeBar': False}
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6)
        ]),

        # Info dettagliate
        dbc.Row([
            dbc.Col([
                html.Div(
                    id='extra-info',
                    className="mt-3"
                )
            ])
        ])
    ], label="Produzione", tab_id="tab-production")


def create_environmental_simulation_tab():
    return dbc.Tab([
        # Container per KPI
        dbc.Row([
            dbc.Col([
                html.Div(id='kpi-container')
            ], md=12),
        ], className="mb-4"),

        # Grafici simulazione
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Simulazione Crescita"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='growth-simulation-chart',
                            config={'displayModeBar': True,
                                    'scrollZoom': True}
                        )
                    ])
                ])
            ], md=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Impatto sulla Produzione"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='production-simulation-chart',
                            config={'displayModeBar': True}
                        )
                    ])
                ])
            ], md=12),
        ], className="mb-4"),

        # Riepilogo simulazione
        dbc.Row([
            dbc.Col([
                html.Div(id='simulation-summary')
            ], md=12),
        ]),

        # Store per dati simulazione
        dcc.Store(id='simulation-data')

    ], label="Simulazione Ambientale")


def create_economic_analysis_tab():
    return dbc.Tab([
        # Sezione Costi di Trasformazione
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Costi di Trasformazione"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Frantoio", className="mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Molitura: "),
                                        "€0.15/kg olive"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Stoccaggio: "),
                                        "€0.20/L olio"
                                    ])
                                ], flush=True)
                            ], md=6),
                            dbc.Col([
                                html.H5("Imbottigliamento", className="mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Bottiglia (1L): "),
                                        "€1.20/unità"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Etichettatura: "),
                                        "€0.30/bottiglia"
                                    ])
                                ], flush=True)
                            ], md=6)
                        ])
                    ])
                ], style=CARD_STYLE)
            ], md=12)
        ]),

        # Sezione Ricavi e Guadagni
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analisi Economica"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Ricavi", className="mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Prezzo vendita olio: "),
                                        "€12.00/L"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Ricavo totale: "),
                                        "€48,000.00"
                                    ])
                                ], flush=True)
                            ], md=4),
                            dbc.Col([
                                html.H5("Costi Totali", className="mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Costi produzione: "),
                                        "€25,000.00"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Costi trasformazione: "),
                                        "€8,000.00"
                                    ])
                                ], flush=True)
                            ], md=4),
                            dbc.Col([
                                html.H5("Margini", className="mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Margine lordo: "),
                                        "€15,000.00"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Margine per litro: "),
                                        "€3.75/L"
                                    ])
                                ], flush=True)
                            ], md=4)
                        ])
                    ])
                ], style=CARD_STYLE)
            ], md=12)
        ]),

        # Grafici Finanziari
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Distribuzione Costi"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.pie(
                                values=[2000, 500, 800, 1500, 600, 400],
                                names=['Ammortamento', 'Assicurazione', 'Manutenzione',
                                       'Raccolta', 'Potatura', 'Fertilizzanti'],
                                title='Distribuzione Costi per Ettaro'
                            ),
                            config={'displayModeBar': False}
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analisi Break-Even"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=px.line(
                                x=[0, 1000, 2000, 3000, 4000],
                                y=[[0, 12000, 24000, 36000, 48000],
                                   [5000, 15000, 25000, 35000, 45000]],
                                title='Analisi Break-Even',
                                labels={'x': 'Litri di olio', 'y': 'Euro'}
                            ),
                            config={'displayModeBar': False}
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6)
        ])
    ], label="Analisi Economica", tab_id="tab-financial")


def create_configuration_tab():
    return dbc.Tab([
        dbc.Row([
            # Configurazione Oliveto
            dbc.Col([
                create_costs_config_section()
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Configurazione Oliveto", className="text-primary mb-0"),
                    ], className="bg-light"),
                    dbc.CardBody([
                        # Hectares input
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Ettari totali:", className="fw-bold"),
                                dbc.Input(
                                    id='hectares-input',
                                    type='number',
                                    value=1,
                                    min=1,
                                    className="mb-3"
                                )
                            ])
                        ]),

                        # Variety sections
                        html.Div([
                            # Variety 1
                            html.Div([
                                html.H6("Varietà 1", className="text-primary mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Varietà:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='variety-1-dropdown',
                                            options=[{'label': v, 'value': v}
                                                     for v in olive_varieties['Varietà di Olive'].unique()],
                                            value=olive_varieties['Varietà di Olive'].iloc[0],
                                            className="mb-2"
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Tecnica:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='technique-1-dropdown',
                                            options=[
                                                {'label': 'Tradizionale', 'value': 'tradizionale'},
                                                {'label': 'Intensiva', 'value': 'intensiva'},
                                                {'label': 'Superintensiva', 'value': 'superintensiva'}
                                            ],
                                            value='Tradizionale',
                                            className="mb-2"
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Percentuale:", className="fw-bold"),
                                        dbc.Input(
                                            id='percentage-1-input',
                                            type='number',
                                            min=1,
                                            max=100,
                                            value=100,
                                            className="mb-2"
                                        )
                                    ], md=4)
                                ])
                            ], className="mb-4"),

                            # Variety 2
                            html.Div([
                                html.H6("Varietà 2 (opzionale)", className="text-primary mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Varietà:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='variety-2-dropdown',
                                            options=[{'label': v, 'value': v}
                                                     for v in olive_varieties['Varietà di Olive'].unique()],
                                            value=None,
                                            className="mb-2"
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Tecnica:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='technique-2-dropdown',
                                            options=[
                                                {'label': 'Tradizionale', 'value': 'tradizionale'},
                                                {'label': 'Intensiva', 'value': 'intensiva'},
                                                {'label': 'Superintensiva', 'value': 'superintensiva'}
                                            ],
                                            value=None,
                                            disabled=True,
                                            className="mb-2"
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Percentuale:", className="fw-bold"),
                                        dbc.Input(
                                            id='percentage-2-input',
                                            type='number',
                                            min=0,
                                            max=99,
                                            value=0,
                                            disabled=True,
                                            className="mb-2"
                                        )
                                    ], md=4)
                                ])
                            ], className="mb-4"),

                            # Variety 3
                            html.Div([
                                html.H6("Varietà 3 (opzionale)", className="text-primary mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Varietà:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='variety-3-dropdown',
                                            options=[{'label': v, 'value': v}
                                                     for v in olive_varieties['Varietà di Olive'].unique()],
                                            value=None,
                                            className="mb-2"
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Tecnica:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='technique-3-dropdown',
                                            options=[
                                                {'label': 'Tradizionale', 'value': 'tradizionale'},
                                                {'label': 'Intensiva', 'value': 'intensiva'},
                                                {'label': 'Superintensiva', 'value': 'superintensiva'}
                                            ],
                                            value=None,
                                            disabled=True,
                                            className="mb-2"
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Percentuale:", className="fw-bold"),
                                        dbc.Input(
                                            id='percentage-3-input',
                                            type='number',
                                            min=0,
                                            max=99,
                                            value=0,
                                            disabled=True,
                                            className="mb-2"
                                        )
                                    ], md=4)
                                ])
                            ], className="mb-4"),
                        ]),

                        # Warning message
                        html.Div(
                            id='percentage-warning',
                            className="text-danger mt-3"
                        )
                    ])
                ], className="mb-4")
            ], md=6),
            dbc.Row([
                dbc.Col([
                    create_inference_config_section()
                ], md=12)
            ]),
            # Configurazione Costi
            html.Div([
                dbc.Button(
                    "Salva Configurazione",
                    id="save-config-button",
                    color="primary",
                    className="mt-3"
                ),
                html.Div(
                    id="save-config-message",
                    className="mt-2"
                )
            ], className="text-center")
        ])
    ], label="Configurazione", tab_id="tab-config")


def create_costs_config_section():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Configurazione Costi e Marketing", className="text-primary mb-0")
        ], className="bg-light"),
        dbc.CardBody([
            # Costi Fissi Annuali (totali)
            html.H5("Costi Fissi Annuali Totali", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Ammortamento impianto (€):", className="fw-bold"),
                    dbc.Input(
                        id='cost-ammortamento',  # Mantenuto ID originale
                        type='number',
                        value=10000,
                        min=0,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Assicurazioni (€):", className="fw-bold"),
                    dbc.Input(
                        id='cost-assicurazione',  # Mantenuto ID originale
                        type='number',
                        value=2500,
                        min=0,
                        className="mb-2"
                    )
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Manutenzione attrezzature (€):", className="fw-bold"),
                    dbc.Input(
                        id='cost-manutenzione',  # Mantenuto ID originale
                        type='number',
                        value=4000,
                        min=0,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Certificazioni e licenze (€):", className="fw-bold"),
                    dbc.Input(
                        id='cost-certificazioni',  # Nuovo campo
                        type='number',
                        value=3000,
                        min=0,
                        className="mb-2"
                    )
                ], md=6)
            ], className="mb-4"),

            # Costi Variabili (per ettaro)
            html.H5("Costi Variabili", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Manodopera raccolta (€/kg):", className="fw-bold"),
                    dbc.Input(
                        id='cost-raccolta',  # Mantenuto ID originale
                        type='number',
                        value=0.35,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Potatura (€/ha):", className="fw-bold"),
                    dbc.Input(
                        id='cost-potatura',  # Mantenuto ID originale
                        type='number',
                        value=600,
                        min=0,
                        className="mb-2"
                    )
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fertilizzanti (€/ha):", className="fw-bold"),
                    dbc.Input(
                        id='cost-fertilizzanti',  # Mantenuto ID originale
                        type='number',
                        value=400,
                        min=0,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Irrigazione (€/ha):", className="fw-bold"),
                    dbc.Input(
                        id='cost-irrigazione',  # Nuovo campo
                        type='number',
                        value=300,
                        min=0,
                        className="mb-2"
                    )
                ], md=6)
            ], className="mb-4"),

            # Costi di Trasformazione
            html.H5("Costi di Trasformazione", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Molitura (€/kg olive):", className="fw-bold"),
                    dbc.Input(
                        id='cost-molitura',  # Mantenuto ID originale
                        type='number',
                        value=0.15,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Stoccaggio (€/L olio):", className="fw-bold"),
                    dbc.Input(
                        id='cost-stoccaggio',  # Mantenuto ID originale
                        type='number',
                        value=0.20,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Bottiglia 1L (€/unità):", className="fw-bold"),
                    dbc.Input(
                        id='cost-bottiglia',  # Mantenuto ID originale
                        type='number',
                        value=1.20,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Etichettatura (€/bottiglia):", className="fw-bold"),
                    dbc.Input(
                        id='cost-etichettatura',  # Mantenuto ID originale
                        type='number',
                        value=0.30,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6)
            ], className="mb-4"),

            # Marketing e Vendita (nuova sezione)
            html.H5("Marketing e Vendita", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Budget Marketing Annuale (€):", className="fw-bold"),
                    dbc.Input(
                        id='cost-marketing',  # Nuovo campo
                        type='number',
                        value=15000,
                        min=0,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Costi commerciali (€/L):", className="fw-bold"),
                    dbc.Input(
                        id='cost-commerciali',  # Nuovo campo
                        type='number',
                        value=0.50,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Prezzo vendita olio (€/L):", className="fw-bold"),
                    dbc.Input(
                        id='price-olio',  # Mantenuto ID originale
                        type='number',
                        value=12.00,
                        min=0,
                        step=0.01,
                        className="mb-2"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("% Vendita diretta:", className="fw-bold"),
                    dbc.Input(
                        id='perc-vendita-diretta',  # Nuovo campo
                        type='number',
                        value=30,
                        min=0,
                        max=100,
                        className="mb-2"
                    )
                ], md=6)
            ], className="mb-4"),
        ])
    ])


def create_inference_config_section():
    return dbc.Card([
        html.Div(id=Ids.INFERENCE_CONTAINER, children=[
            dbc.CardHeader([
                html.H4("Configurazione Inferenza", className="text-primary mb-0"),
                dbc.Switch(
                    id=Ids.DEBUG_SWITCH,
                    label="Modalità Debug",
                    value=True,
                    className="mt-2"
                ),
            ], className="bg-light"),
            dbc.CardBody([
                # Stato del servizio
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("Stato Servizio", className="mb-3"),
                            html.Div(id=Ids.INFERENCE_STATUS, className="mb-3"),
                        ])
                    ])
                ], className="mb-4"),

                # Metriche e monitoraggio
                dbc.Row([
                    dbc.Col([
                        html.H5("Metriche", className="mb-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("Modalità: "),
                                html.Span(id=Ids.INFERENCE_MODE)
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Latenza media: "),
                                html.Span(id=Ids.INFERENCE_LATENCY)
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Richieste totali: "),
                                html.Span(id=Ids.INFERENCE_REQUESTS)
                            ])
                        ], flush=True)
                    ])
                ], className="mt-4")
            ])
        ])
    ])


def create_growth_simulation_figure(sim_data: pd.DataFrame) -> go.Figure:
    """Crea il grafico della simulazione di crescita"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Aggiunge la linea di crescita
    fig.add_trace(
        go.Scatter(
            x=sim_data['date'],
            y=sim_data['growth_rate'],
            name="Tasso di Crescita",
            line=dict(color='#2E86C1', width=2)
        ),
        secondary_y=False
    )

    # Aggiunge l'indice di stress
    fig.add_trace(
        go.Scatter(
            x=sim_data['date'],
            y=sim_data['stress_index'],
            name="Indice di Stress",
            line=dict(color='#E74C3C', width=2)
        ),
        secondary_y=True
    )

    # Aggiungi indicatori delle fasi
    for phase in sim_data['phase'].unique():
        phase_data = sim_data[sim_data['phase'] == phase]
        fig.add_trace(
            go.Scatter(
                x=[phase_data['date'].iloc[0]],
                y=[0],
                name=phase,
                mode='markers+text',
                text=[phase],
                textposition='top center',
                marker=dict(size=10)
            ),
            secondary_y=False
        )

    # Configurazione layout
    fig.update_layout(
        title='Simulazione Crescita e Stress Ambientale',
        xaxis_title='Data',
        yaxis_title='Tasso di Crescita (%)',
        yaxis2_title='Indice di Stress',
        hovermode='x unified',
        showlegend=True,
        height=500
    )

    return fig


def create_production_impact_figure(sim_data: pd.DataFrame) -> go.Figure:
    """
    Crea il grafico dell'impatto sulla produzione, con gestione corretta del resampling mensile.

    Parameters
    ----------
    sim_data : pd.DataFrame
        DataFrame contenente i dati della simulazione con colonne:
        - date: datetime
        - stress_index: float
        - phase: str
        - temperature: float
        - growth_rate: float

    Returns
    -------
    go.Figure
        Figura Plotly con il grafico dell'impatto sulla produzione
    """
    # Verifica che la colonna date sia datetime
    if not pd.api.types.is_datetime64_any_dtype(sim_data['date']):
        sim_data['date'] = pd.to_datetime(sim_data['date'])

    # Setta l'indice come datetime
    sim_data_indexed = sim_data.set_index('date')

    # Calcola medie mensili solo per le colonne numeriche
    numeric_columns = ['stress_index', 'temperature', 'growth_rate']
    monthly_means = {}

    for col in numeric_columns:
        if col in sim_data_indexed.columns:
            monthly_means[col] = sim_data_indexed[col].resample('ME').mean()

    # Crea DataFrame con le medie mensili
    monthly_data = pd.DataFrame(monthly_means)

    # Crea la figura
    fig = go.Figure()

    # Calcola la produzione stimata (100% - stress%)
    production_estimate = 100 * (1 - monthly_data['stress_index'])

    # Aggiungi traccia principale
    fig.add_trace(
        go.Bar(
            x=monthly_data.index,
            y=production_estimate,
            name='Produzione Stimata (%)',
            marker_color='#27AE60'
        )
    )

    # Aggiungi linea di trend
    fig.add_trace(
        go.Scatter(
            x=monthly_data.index,
            y=production_estimate.rolling(window=3, min_periods=1).mean(),
            name='Trend (Media Mobile 3 mesi)',
            line=dict(color='#2C3E50', width=2, dash='dot')
        )
    )

    # Configura il layout
    fig.update_layout(
        title={
            'text': 'Impatto Stimato sulla Produzione',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
        },
        xaxis_title='Mese',
        yaxis_title='Produzione Stimata (%)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white',
        yaxis=dict(
            range=[0, 100],
            tickformat='.0f',
            ticksuffix='%'
        ),
        xaxis=dict(
            tickformat='%B %Y',
            tickangle=45
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, b=80)
    )

    # Aggiungi annotazioni per valori estremi
    min_prod = production_estimate.min()
    max_prod = production_estimate.max()

    fig.add_annotation(
        x=production_estimate.idxmin(),
        y=min_prod,
        text=f"Min: {min_prod:.1f}%",
        showarrow=True,
        arrowhead=1
    )

    fig.add_annotation(
        x=production_estimate.idxmax(),
        y=max_prod,
        text=f"Max: {max_prod:.1f}%",
        showarrow=True,
        arrowhead=1
    )

    return fig


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session', storage_type='local'),
    dcc.Store(id='user-data', storage_type='local'),
    dcc.Store(id=Ids.INFERENCE_COUNTER, storage_type='session', data={'count': 0}),
    dcc.Store(id=Ids.DEV_MODE, storage_type='session', data={'count': 0}),
    html.Div(id=Ids.AUTH_CONTAINER),
    html.Div(id=Ids.DASHBOARD_CONTAINER),
])


def create_main_layout():
    return dbc.Container([
        dcc.Location(id='_pages_location'),
        # Navbar con logout button
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand([
                    html.I(className="fas fa-leaf me-2"),  # Icona olivo
                    "Dashboard Produzione Olio d'Oliva"
                ], className="text-white"),
                dbc.Nav([
                    dbc.Button(
                        [
                            html.I(className="fas fa-sign-out-alt me-2"),  # Icona logout
                            "Logout"
                        ],
                        id="logout-button",
                        color="light",
                        outline=True,
                        size="sm",
                        className="ms-2"
                    )
                ], className="ms-auto d-flex align-items-center")  # Modificato qui per un migliore allineamento
            ],
                fluid=True,  # Aggiunto per sfruttare tutta la larghezza
            ),
            color="primary",
            dark=True,
            className="mb-3"
        ),
        html.Div(id='loading-alert'),
        dbc.Tooltip(
            "Seleziona una seconda varietà per creare un mix",
            target="variety-2-dropdown",
            placement="top"
        ),
        dbc.Tooltip(
            "Seleziona una terza varietà per completare il mix",
            target="variety-3-dropdown",
            placement="top"
        ),
        # Header
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    create_production_tab(),
                    create_environmental_simulation_tab(),
                    create_economic_analysis_tab(),
                    create_configuration_tab(),
                ], id="tabs", active_tab="tab-production")
            ], md=12, lg=12)
        ]),

        # Store per dati simulazione
        dcc.Store(id='simulation-data')
    ], fluid=True, className="px-4 py-3")


def create_extra_info_component(prediction, varieties_info):
    """Crea il componente delle informazioni dettagliate con il nuovo stile"""
    cards = []

    # Card per ogni varietà
    for detail, variety_info in zip(prediction['variety_details'], varieties_info):
        # Calcola la resa in modo sicuro
        resa = (detail['oil_total'] / detail['production_total'] * 100) if detail['production_total'] > 0 else 0

        variety_card = dbc.Card([
            dbc.CardHeader(
                html.H5(f"{detail['variety']} - {detail['percentage']}%",
                        className="mb-0 text-primary")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Produzione", className="text-muted mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Strong("Olive: "),
                                    f"{detail['production_total']:.0f} kg"
                                ], className="px-2 py-1"),
                                dbc.ListGroupItem([
                                    html.Strong("Olio: "),
                                    f"{detail['oil_total']:.0f} L"
                                ], className="px-2 py-1"),
                                dbc.ListGroupItem([
                                    html.Strong("Resa: "),
                                    f"{resa:.1f}%"
                                ], className="px-2 py-1")
                            ], flush=True)
                        ])
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.H6("Produzione/ha", className="text-muted mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Strong("Olive: "),
                                    f"{detail['production_per_ha']:.0f} kg/ha"
                                ], className="px-2 py-1"),
                                dbc.ListGroupItem([
                                    html.Strong("Olio: "),
                                    f"{detail['oil_per_ha']:.0f} L/ha"
                                ], className="px-2 py-1")
                            ], flush=True)
                        ])
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.H6("Caratteristiche", className="text-muted mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Strong("Resa: "),
                                    f"{variety_info['Min % Resa']:.1f}% - {variety_info['Max % Resa']:.1f}%"
                                ], className="px-2 py-1"),
                                dbc.ListGroupItem([
                                    html.Strong("L/t: "),
                                    f"{variety_info['Min Litri per Tonnellata']:.0f} - {variety_info['Max Litri per Tonnellata']:.0f}"
                                ], className="px-2 py-1")
                            ], flush=True)
                        ])
                    ], md=4)
                ]),

                # Fabbisogno Idrico
                dbc.Row([
                    dbc.Col([
                        html.H6("Fabbisogno Idrico Stagionale",
                                className="text-muted mb-3 mt-3"),
                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Stagione"),
                                    html.Th("m³/ha")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Primavera"),
                                    html.Td(f"{variety_info['Fabbisogno Acqua Primavera (m³/ettaro)']:.0f}")
                                ]),
                                html.Tr([
                                    html.Td("Estate"),
                                    html.Td(f"{variety_info['Fabbisogno Acqua Estate (m³/ettaro)']:.0f}")
                                ]),
                                html.Tr([
                                    html.Td("Autunno"),
                                    html.Td(f"{variety_info['Fabbisogno Acqua Autunno (m³/ettaro)']:.0f}")
                                ]),
                                html.Tr([
                                    html.Td("Inverno"),
                                    html.Td(f"{variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']:.0f}")
                                ])
                            ])
                        ], size="sm", bordered=True)
                    ])
                ])
            ])
        ], className="mb-3", style=CARD_STYLE)
        cards.append(variety_card)

    # Calcola la resa media in modo sicuro
    resa_media = (
        (prediction['avg_oil_production_total'] / prediction['olive_production_total'] * 100)
        if prediction['olive_production_total'] > 0 else 0
    )

    # Card riepilogo totali
    summary_card = dbc.Card([
        dbc.CardHeader(
            html.H5("Riepilogo Totali", className="mb-0 text-primary")
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong("Produzione Olive: "),
                            f"{prediction['olive_production']:.0f} kg/ha"
                        ], className="px-2 py-1"),
                        dbc.ListGroupItem([
                            html.Strong("Produzione Olio: "),
                            f"{prediction['avg_oil_production']:.0f} L/ha"
                        ], className="px-2 py-1"),
                        dbc.ListGroupItem([
                            html.Strong("Resa Media: "),
                            f"{resa_media:.1f}%"
                        ], className="px-2 py-1"),
                        dbc.ListGroupItem([
                            html.Strong("Fabbisogno Idrico: "),
                            f"{prediction['water_need']:.0f} m³/ha"
                        ], className="px-2 py-1")
                    ], flush=True)
                ])
            ])
        ])
    ], className="mb-3", style=CARD_STYLE)

    return html.Div([
        dbc.Row([
            dbc.Col(card, md=12 if len(cards) == 1 else 6 if len(cards) == 2 else 4)
            for card in cards
        ]),
        dbc.Row([
            dbc.Col(summary_card)
        ])
    ])


def create_figure_layout(fig, title):
    fig.update_layout(
        title=title,
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        font=dict(family="Helvetica, Arial, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def create_production_details_figure(prediction):
    """Crea il grafico dei dettagli produzione con il nuovo stile"""
    details_data = prepare_details_data(prediction)
    fig = px.bar(
        details_data,
        x='Varietà',
        y='Produzione',
        color='Tipo',
        barmode='group',
        color_discrete_map={'Olive': '#2185d0', 'Olio': '#21ba45'}
    )
    return create_figure_layout(fig, 'Dettagli Produzione per Varietà')


def create_weather_impact_figure(weather_data):
    """Crea il grafico dell'impatto meteorologico con il nuovo stile"""
    recent_weather = weather_data.tail(41).copy()
    fig = px.scatter(
        recent_weather,
        x='temp',
        y='solarradiation',
        size='precip',
        color_discrete_sequence=['#2185d0']
    )
    return create_figure_layout(fig, 'Condizioni Meteorologiche')


def create_water_needs_figure(prediction):
    """Crea il grafico del fabbisogno idrico con il nuovo stile"""
    # Definisci i mesi in italiano
    months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
              'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']

    water_data = []
    for detail in prediction['variety_details']:
        for month in months:
            season = get_season_from_month(month)
            variety_info = olive_varieties[
                olive_varieties['Varietà di Olive'] == detail['variety']
                ].iloc[0]

            water_need = variety_info[f'Fabbisogno Acqua {season} (m³/ettaro)']
            water_data.append({
                'Month': month,
                'Variety': detail['variety'],
                'Water_Need': water_need * (detail['percentage'] / 100)
            })

    water_df = pd.DataFrame(water_data)
    fig = px.bar(
        water_df,
        x='Month',
        y='Water_Need',
        color='Variety',
        barmode='stack',
        color_discrete_sequence=['#2185d0', '#21ba45', '#6435c9']
    )

    return create_figure_layout(fig, 'Fabbisogno Idrico Mensile')


def prepare_details_data(prediction):
    """Prepara i dati per il grafico dei dettagli di produzione"""
    details_data = []

    # Dati per ogni varietà
    for detail in prediction['variety_details']:
        details_data.extend([
            {
                'Varietà': f"{detail['variety']} ({detail['percentage']}%)",
                'Tipo': 'Olive',
                'Produzione': detail['production_per_ha']
            },
            {
                'Varietà': f"{detail['variety']} ({detail['percentage']}%)",
                'Tipo': 'Olio',
                'Produzione': detail['oil_per_ha']
            }
        ])

    # Aggiungi totali
    details_data.extend([
        {
            'Varietà': 'Totale',
            'Tipo': 'Olive',
            'Produzione': prediction['olive_production']
        },
        {
            'Varietà': 'Totale',
            'Tipo': 'Olio',
            'Produzione': prediction['avg_oil_production']
        }
    ])

    return pd.DataFrame(details_data)


def get_season_from_month(month):
    """Helper function per determinare la stagione dal mese."""
    seasons = {
        'Gen': 'Inverno', 'Feb': 'Inverno', 'Mar': 'Primavera',
        'Apr': 'Primavera', 'Mag': 'Primavera', 'Giu': 'Estate',
        'Lug': 'Estate', 'Ago': 'Estate', 'Set': 'Autunno',
        'Ott': 'Autunno', 'Nov': 'Autunno', 'Dic': 'Inverno'
    }
    return seasons[month]


@app.callback(
    Output('loading-alert', 'children'),
    [Input('simulate-btn', 'n_clicks'),
     Input('debug-switch', 'value')],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def update_loading_status(n_clicks, debug_mode):
    global DEV_MODE

    config = load_config()

    print(config)
    DEV_MODE = config['inference']['debug_mode']
    if MODEL_LOADING:
        return dbc.Alert(
            [
                html.I(className="fas fa-spinner fa-spin me-2"),
                "Caricamento del modello in corso..."
            ],
            color="warning",
            is_open=True
        )
    return None


@app.callback(
    [
        Output(Ids.PRODUCTION_INFERENCE_MODE, 'children'),
        Output(Ids.PRODUCTION_INFERENCE_REQUESTS, 'children'),
        Output(Ids.INFERENCE_COUNTER, 'data', allow_duplicate=True)
    ],
    [Input(Ids.PRODUCTION_DEBUG_SWITCH, 'value')],
    [State(Ids.INFERENCE_COUNTER, 'data')],
    prevent_initial_call=True
)
def update_inference_status(debug_mode, counter_data):
    try:
        toggle_inference_mode(debug_mode)
        mode_text = "Debug (Mock)" if debug_mode else "Produzione (Model)"
        # Resetta il contatore
        new_counter_data = {'count': 0}
        return mode_text, "0", new_counter_data
    except Exception as e:
        print(f"Errore nell'aggiornamento dello stato di inferenza: {e}")
        return "Errore", "N/A", {'count': 0}


@app.callback(
    [
        Output(Ids.INFERENCE_STATUS, 'children'),
        Output(Ids.INFERENCE_MODE, 'children', allow_duplicate=True),
        Output(Ids.INFERENCE_LATENCY, 'children'),
        Output(Ids.INFERENCE_REQUESTS, 'children', allow_duplicate=True),
        Output(Ids.INFERENCE_COUNTER, 'data', allow_duplicate=True)
    ],
    [Input(Ids.INFERENCE_CONTAINER, 'value')],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ],
    prevent_initial_call=True
)
def toggle_inference_mode(debug_mode):
    global DEV_MODE, model, MODEL_LOADING, scaler_temporal, scaler_static, scaler_y
    new_counter_data = {'count': 0}
    try:
        config = load_config()
        print(f"debug mode: {debug_mode}")
        # Aggiorna la modalità debug nella configurazione
        config['inference'] = config.get('inference', {})  # Crea la sezione se non esiste
        config['inference']['debug_mode'] = debug_mode

        DEV_MODE = debug_mode
        print(f"DEV_MODE: {DEV_MODE}")
        dcc.Store(id=Ids.INFERENCE_COUNTER, data=new_counter_data)
        if debug_mode:

            MODEL_LOADING = False
            return (
                dbc.Alert("Modalità Debug attiva - Using mock predictions", color="info"),
                "Debug (Mock)",
                "< 1ms",
                "N/A",
                new_counter_data
            )
        else:
            if model is None:
                try:
                    MODEL_LOADING = True
                    print(f"Keras version: {keras.__version__}")
                    print(f"TensorFlow version: {tf.__version__}")
                    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
                    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

                    # GPU memory configuration
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)

                            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                        except RuntimeError as e:
                            print(e)

                    @keras.saving.register_keras_serializable()
                    class DataAugmentation(tf.keras.layers.Layer):
                        """Custom layer per l'augmentation dei dati"""

                        def __init__(self, noise_stddev=0.03, **kwargs):
                            super().__init__(**kwargs)
                            self.noise_stddev = noise_stddev

                        def call(self, inputs, training=None):
                            if training:
                                return inputs + tf.random.normal(
                                    shape=tf.shape(inputs),
                                    mean=0.0,
                                    stddev=self.noise_stddev
                                )
                            return inputs

                        def get_config(self):
                            config = super().get_config()
                            config.update({"noise_stddev": self.noise_stddev})
                            return config

                    @keras.saving.register_keras_serializable()
                    class PositionalEncoding(tf.keras.layers.Layer):
                        """Custom layer per l'encoding posizionale"""

                        def __init__(self, d_model, **kwargs):
                            super().__init__(**kwargs)
                            self.d_model = d_model

                        def build(self, input_shape):
                            _, seq_length, _ = input_shape

                            # Crea la matrice di encoding posizionale
                            position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
                            div_term = tf.exp(
                                tf.range(0, self.d_model, 2, dtype=tf.float32) *
                                (-tf.math.log(10000.0) / self.d_model)
                            )

                            # Calcola sin e cos
                            pos_encoding = tf.zeros((1, seq_length, self.d_model))
                            pos_encoding_even = tf.sin(position * div_term)
                            pos_encoding_odd = tf.cos(position * div_term)

                            # Assegna i valori alle posizioni pari e dispari
                            pos_encoding = tf.concat(
                                [tf.expand_dims(pos_encoding_even, -1),
                                 tf.expand_dims(pos_encoding_odd, -1)],
                                axis=-1
                            )
                            pos_encoding = tf.reshape(pos_encoding, (1, seq_length, -1))
                            pos_encoding = pos_encoding[:, :, :self.d_model]

                            # Salva l'encoding come peso non trainabile
                            self.pos_encoding = self.add_weight(
                                shape=(1, seq_length, self.d_model),
                                initializer=tf.keras.initializers.Constant(pos_encoding),
                                trainable=False,
                                name='positional_encoding'
                            )

                            super().build(input_shape)

                        def call(self, inputs):
                            # Broadcast l'encoding posizionale sul batch
                            batch_size = tf.shape(inputs)[0]
                            pos_encoding_tiled = tf.tile(self.pos_encoding, [batch_size, 1, 1])
                            return inputs + pos_encoding_tiled

                        def get_config(self):
                            config = super().get_config()
                            config.update({"d_model": self.d_model})
                            return config

                    @keras.saving.register_keras_serializable()
                    class WarmUpLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
                        """Custom learning rate schedule with linear warmup and exponential decay."""

                        def __init__(self, initial_learning_rate=1e-3, warmup_steps=500, decay_steps=5000):
                            super().__init__()
                            self.initial_learning_rate = initial_learning_rate
                            self.warmup_steps = warmup_steps
                            self.decay_steps = decay_steps

                        def __call__(self, step):
                            warmup_pct = tf.cast(step, tf.float32) / self.warmup_steps
                            warmup_lr = self.initial_learning_rate * warmup_pct
                            decay_factor = tf.pow(0.1, tf.cast(step, tf.float32) / self.decay_steps)
                            decayed_lr = self.initial_learning_rate * decay_factor
                            return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)

                        def get_config(self):
                            return {
                                'initial_learning_rate': self.initial_learning_rate,
                                'warmup_steps': self.warmup_steps,
                                'decay_steps': self.decay_steps
                            }

                    @keras.saving.register_keras_serializable()
                    def weighted_huber_loss(y_true, y_pred):
                        # Pesi per diversi output
                        weights = tf.constant([1.0, 0.8, 0.8, 1.0, 0.6], dtype=tf.float32)
                        huber = tf.keras.losses.Huber(delta=1.0)
                        loss = huber(y_true, y_pred)
                        weighted_loss = tf.reduce_mean(loss * weights)
                        return weighted_loss

                    print("Caricamento modello...")

                    # Verifica che il modello sia disponibile
                    model_path = './sources/olive_oil_transformer/olive_oil_transformer_model.keras'
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Modello non trovato in: {model_path}")

                    # Prova a caricare il modello
                    model = tf.keras.models.load_model(model_path, custom_objects={
                        'DataAugmentation': DataAugmentation,
                        'PositionalEncoding': PositionalEncoding,
                        'WarmUpLearningRateSchedule': WarmUpLearningRateSchedule,
                        'weighted_huber_loss': weighted_huber_loss
                    })
                    MODEL_LOADING = False
                    return (
                        dbc.Alert("Modello caricato correttamente", color="success"),
                        "Produzione (Local Model)",
                        "~ 100ms",
                        "0",
                        new_counter_data
                    )
                except Exception as e:
                    print(f"Errore nel caricamento del modello: {str(e)}")
                    # Se c'è un errore nel caricamento del modello, torna in modalità debug
                    DEV_MODE = True
                    MODEL_LOADING = False
                    return (
                        dbc.Alert(f"Errore nel caricamento del modello: {str(e)}", color="danger"),
                        "Debug (Mock) - Fallback",
                        "N/A",
                        "N/A",
                        new_counter_data
                    )

            else:
                MODEL_LOADING = False
    except Exception as e:
        print(f"Errore nella configurazione inferenza: {str(e)}")
        MODEL_LOADING = False
        return (
            dbc.Alert(f"Errore: {str(e)}", color="danger"),
            "Errore",
            "Errore",
            "Errore",
            new_counter_data
        )


@app.callback(
    [
        Output(Ids.DEBUG_SWITCH, 'value'),
        Output(Ids.PRODUCTION_DEBUG_SWITCH, 'value')
    ],
    [Input('url', 'pathname')],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def init_debug_switch(pathname):
    if pathname != '/':
        raise PreventUpdate
    try:
        config = load_config()
        return config.get('inference', {}).get('debug_mode', True), config.get('inference', {}).get('debug_mode', True)  # Default a True se non configurato
    except Exception as e:
        print(f"Errore nel caricamento della configurazione debug: {str(e)}")
        return True, True  # Default a True in caso di errore


@app.callback(
    [Output(Ids.AUTH_CONTAINER, 'children'),
     Output(Ids.DASHBOARD_CONTAINER, 'children')],
    [Input('url', 'pathname'),
     Input('session', 'data')],
)
def display_page(pathname, session_data):
    # print(f"Session data: {session_data}")  # Debug print

    if pathname == '/register':
        return create_register_layout(), html.Div()

    if not session_data:
        print("No session data found")  # Debug print
        return create_login_layout(), html.Div()

    if 'token' not in session_data:
        print("No token in session data")  # Debug print
        return create_login_layout(), html.Div()

    is_valid, username = verify_token(session_data['token'])
    if not is_valid:
        print("Invalid token")  # Debug print
        return create_login_layout(), html.Div()

    # print(f"Valid session for user: {username}")  # Debug print
    return html.Div(), create_main_layout()


def check_session():
    """Verifica lo stato della sessione e restituisce i dati se disponibili"""
    if not flask.has_request_context():
        print("No request context available")
        return None

    try:
        print("\nChecking session...")
        # Prima prova a leggere dalla sessione corrente
        if 'user_session' in flask.session:
            print(f"Found session in flask.session: {flask.session['user_session']}")
            return flask.session['user_session']

        # Poi prova a leggere dai cookies
        session_data = flask.request.cookies.get('session')
        print(f"Session cookie data: {session_data}")

        if session_data:
            try:
                session_data = json.loads(session_data)
                print(f"Parsed session data: {session_data}")
                if 'username' in session_data:
                    print(f"Found username in session: {session_data['username']}")
                    return session_data
            except json.JSONDecodeError as e:
                print(f"Error decoding session data: {e}")
                # Prova a decodificare in base64
                try:
                    import base64
                    decoded = base64.b64decode(session_data)
                    session_data = json.loads(decoded)
                    if 'username' in session_data:
                        print(f"Found username in decoded session: {session_data['username']}")
                        return session_data
                except Exception as be:
                    print(f"Error decoding base64 session: {be}")
                return None
    except Exception as e:
        print(f"Error checking session: {str(e)}")
        import traceback
        traceback.print_exc()
    print("No session data found")
    return None


def save_session_data(username, token):
    """Salva i dati di sessione sia in Flask che nello store Dash"""
    session_data = {
        'token': token,
        'username': username,
        'authenticated': True
    }

    if flask.has_request_context():
        try:
            flask.session['user_session'] = session_data
            print(f'Saved session data in Flask session: {session_data}')
        except Exception as e:
            print(f"Error saving to Flask session: {e}")

    return session_data


@app.callback(
    [Output('session', 'data'),
     Output(Ids.LOGIN_ERROR, 'children'),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input(Ids.LOGIN_BUTTON, 'n_clicks')],
    [State(Ids.LOGIN_USERNAME, 'value'),
     State(Ids.LOGIN_PASSWORD, 'value')],
    prevent_initial_call=True
)
def login(n_clicks, username, password):
    if n_clicks is None:
        raise PreventUpdate

    print(f"\nAttempting login for user: {username}")

    if verify_user(username, password):
        try:
            token = create_token(username)
            print(f"Token created: {token}")

            # Salva i dati di sessione
            session_data = save_session_data(username, token)
            print(f"Session data saved: {session_data}")

            # Verifica che i dati siano stati salvati
            current_session = check_session()
            print(f"Current session after save: {current_session}")

            return session_data, '', '/'

        except Exception as e:
            print(f"Error in login process: {e}")
            import traceback
            traceback.print_exc()
            return None, dbc.Alert(f"Errore durante il login: {str(e)}", color="danger"), no_update

    return None, dbc.Alert("Credenziali non valide", color="danger"), '/login'


@app.callback(
    [Output(Ids.REGISTER_ERROR, 'children'),
     Output(Ids.REGISTER_SUCCESS, 'children'),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input(Ids.REGISTER_BUTTON, 'n_clicks')],
    [State(Ids.REGISTER_USERNAME, 'value'),
     State(Ids.REGISTER_PASSWORD, 'value'),
     State(Ids.REGISTER_CONFIRM, 'value')],
    prevent_initial_call=True
)
def register(n_clicks, username, password, password_confirm):
    if n_clicks is None:
        raise PreventUpdate

    if password != password_confirm:
        return dbc.Alert("Le password non coincidono", color="danger"), None, no_update

    success, message = create_user(username, password)
    if success:
        return None, dbc.Alert(message, color="success"), '/login'
    return dbc.Alert(message, color="danger"), None, no_update


@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input(Ids.SHOW_REGISTER_BUTTON, 'n_clicks')],
    prevent_initial_call=True
)
def navigate_to_register(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return '/register'


@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input(Ids.SHOW_LOGIN_BUTTON, 'n_clicks')],
    prevent_initial_call=True
)
def navigate_to_login(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return '/login'


@app.callback(
    [Output('session', 'clear_data'),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input('logout-button', 'n_clicks')],
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    print("Performing logout")  # Debug print
    try:
        # Pulisci eventuali dati di sessione Flask
        if flask.has_request_context():
            flask.session.clear()
    except Exception as e:
        print(f"Error clearing Flask session: {e}")

    return True, '/login'


@app.server.before_request
def before_request_func():
    # print("\n=== Request Info ===")
    # print(f"Path: {flask.request.path}")
    # print(f"Cookies: {flask.request.cookies}")
    if flask.has_request_context():
        session_data = flask.request.cookies.get('session')
    #    if session_data:
    #        print(f"Session Data: {session_data}")
    # print("==================\n")


@app.callback(
    Output("save-config-message", "children"),
    [Input("save-config-button", "n_clicks")],
    [State("hectares-input", "value"),
     State("variety-1-dropdown", "value"),
     State("technique-1-dropdown", "value"),
     State("percentage-1-input", "value"),
     State("variety-2-dropdown", "value"),
     State("technique-2-dropdown", "value"),
     State("percentage-2-input", "value"),
     State("variety-3-dropdown", "value"),
     State("technique-3-dropdown", "value"),
     State("percentage-3-input", "value"),
     # Costi fissi
     State("cost-ammortamento", "value"),
     State("cost-assicurazione", "value"),
     State("cost-manutenzione", "value"),
     State("cost-certificazioni", "value"),
     # Costi variabili
     State("cost-raccolta", "value"),
     State("cost-potatura", "value"),
     State("cost-fertilizzanti", "value"),
     State("cost-irrigazione", "value"),
     # Costi trasformazione
     State("cost-molitura", "value"),
     State("cost-stoccaggio", "value"),
     State("cost-bottiglia", "value"),
     State("cost-etichettatura", "value"),
     # Marketing e vendita
     State("cost-marketing", "value"),
     State("cost-commerciali", "value"),
     State("price-olio", "value"),
     State("perc-vendita-diretta", "value"),
     # Debug mode
     State(Ids.DEBUG_SWITCH, "value")],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def save_configuration(n_clicks, hectares, var1, tech1, perc1, var2, tech2, perc2,
                       var3, tech3, perc3, amm, ass, man, cert, rac, pot, fer, irr,
                       mol, sto, bot, eti, mark, comm, price, perc_dir, debug_mode):
    if n_clicks is None:
        return no_update

    # Prepara la configurazione
    varieties = [{"variety": var1, "technique": tech1, "percentage": perc1}]
    if var2:
        varieties.append({"variety": var2, "technique": tech2, "percentage": perc2})
    if var3:
        varieties.append({"variety": var3, "technique": tech3, "percentage": perc3})

    config = {
        'oliveto': {
            'hectares': hectares,
            'varieties': varieties
        },
        'costs': {
            'fixed': {
                'ammortamento': amm,
                'assicurazione': ass,
                'manutenzione': man,
                'certificazioni': cert
            },
            'variable': {
                'raccolta': rac,
                'potatura': pot,
                'fertilizzanti': fer,
                'irrigazione': irr
            },
            'transformation': {
                'molitura': mol,
                'stoccaggio': sto,
                'bottiglia': bot,
                'etichettatura': eti
            },
            'marketing': {
                'budget_annuale': mark,
                'costi_commerciali': comm,
                'prezzo_vendita': price,
                'perc_vendita_diretta': perc_dir
            }
        },
        'inference': {
            'debug_mode': debug_mode,
            'model_path': './sources/olive_oil_transformer/olive_oil_transformer_model.keras'
        }
    }

    success, message = save_config(config)
    if success:
        return dbc.Alert(
            "Configurazione salvata con successo!",
            color="success",
            duration=4000,
            is_open=True
        )
    else:
        return dbc.Alert(
            f"Errore nel salvataggio della configurazione: {message}",
            color="danger",
            duration=4000,
            is_open=True
        )


@app.callback(
    Output("percentage-warning", "children"),
    [
        Input("percentage-1-input", "value"),
        Input("percentage-2-input", "value"),
        Input("percentage-3-input", "value")
    ],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def check_percentages(perc1, perc2, perc3):
    try:
        # Calcola la somma delle percentuali, considerando solo i valori non nulli
        total = sum(p for p in [perc1 or 0, perc2 or 0, perc3 or 0])

        if total > 100:
            return dbc.Alert(
                f"La somma delle percentuali è {total}% (non può superare 100%)",
                color="danger",
                className="mt-2"
            )
        if total < 100:
            return dbc.Alert(
                f"La somma delle percentuali è {total}% (non può essere inferiore a 100%)",
                color="danger",
                className="mt-2"
            )
        return ""

    except Exception as e:
        print(f"Errore nel controllo delle percentuali: {str(e)}")
        return ""


@app.callback(
    [
        # Outputs per i costi e configurazioni base (15 outputs, escluso warning)
        Output("hectares-input", "value"),
        Output("variety-1-dropdown", "value"),
        Output("technique-1-dropdown", "value"),
        Output("percentage-1-input", "value"),
        Output("variety-2-dropdown", "value"),
        Output("technique-2-dropdown", "value"),
        Output("technique-2-dropdown", "disabled"),
        Output("percentage-2-input", "value"),
        Output("percentage-2-input", "disabled"),
        Output("variety-3-dropdown", "value"),
        Output("variety-3-dropdown", "disabled"),
        Output("technique-3-dropdown", "value"),
        Output("technique-3-dropdown", "disabled"),
        Output("percentage-3-input", "value"),
        Output("percentage-3-input", "disabled"),
        # Outputs per i costi fissi (4 outputs)
        Output("cost-ammortamento", "value"),
        Output("cost-assicurazione", "value"),
        Output("cost-manutenzione", "value"),
        Output("cost-certificazioni", "value"),
        # Outputs per i costi variabili (4 outputs)
        Output("cost-raccolta", "value"),
        Output("cost-potatura", "value"),
        Output("cost-fertilizzanti", "value"),
        Output("cost-irrigazione", "value"),
        # Outputs per i costi di trasformazione (4 outputs)
        Output("cost-molitura", "value"),
        Output("cost-stoccaggio", "value"),
        Output("cost-bottiglia", "value"),
        Output("cost-etichettatura", "value"),
        # Outputs per marketing e vendita (4 outputs)
        Output("cost-marketing", "value"),
        Output("cost-commerciali", "value"),
        Output("price-olio", "value"),
        Output("perc-vendita-diretta", "value")
    ],
    [
        Input("tabs", "active_tab"),
        Input("variety-2-dropdown", "value"),
        Input("variety-3-dropdown", "value"),
        Input('_pages_location', 'pathname')
    ],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def load_configuration(active_tab, variety2, variety3, pathname):
    try:
        # Carica la configurazione
        config = load_config()

        # Carica dati varietà
        varieties = config['oliveto']['varieties']
        var1 = varieties[0] if len(varieties) > 0 else {"variety": None, "technique": None, "percentage": 0}
        var2 = varieties[1] if len(varieties) > 1 else {"variety": None, "technique": None, "percentage": 0}
        var3 = varieties[2] if len(varieties) > 2 else {"variety": None, "technique": None, "percentage": 0}

        # Carica costi e marketing
        costs = config['costs']
        fixed = costs.get('fixed', {})
        variable = costs.get('variable', {})
        transformation = costs.get('transformation', {})
        marketing = costs.get('marketing', {})

        var2_exists = var2["variety"] is not None
        var3_exists = var3["variety"] is not None

        return [
            # Configurazioni base (15 valori)
            config['oliveto']['hectares'],
            var1["variety"],
            var1["technique"],
            var1["percentage"],
            var2["variety"],
            var2["technique"],
            not var2_exists,
            var2["percentage"],
            not var2_exists,
            var3["variety"],
            not var2_exists,
            var3["technique"],
            not var2_exists or not var3_exists,
            var3["percentage"],
            not var2_exists or not var3_exists,
            # Costi fissi (4 valori)
            fixed.get('ammortamento', 10000),
            fixed.get('assicurazione', 2500),
            fixed.get('manutenzione', 4000),
            fixed.get('certificazioni', 3000),
            # Costi variabili (4 valori)
            variable.get('raccolta', 0.35),
            variable.get('potatura', 600),
            variable.get('fertilizzanti', 400),
            variable.get('irrigazione', 300),
            # Costi trasformazione (4 valori)
            transformation.get('molitura', 0.15),
            transformation.get('stoccaggio', 0.20),
            transformation.get('bottiglia', 1.20),
            transformation.get('etichettatura', 0.30),
            # Marketing e vendita (4 valori)
            marketing.get('budget_annuale', 15000),
            marketing.get('costi_commerciali', 0.50),
            marketing.get('prezzo_vendita', 12.00),
            marketing.get('perc_vendita_diretta', 30)
        ]

    except Exception as e:
        print(f"Errore in load_configuration: {str(e)}")
        return [no_update] * 31


@app.callback(
    [
        Output(Ids.GROWTH_CHART, 'figure'),
        Output(Ids.PRODUCTION_CHART, 'figure'),
        Output(Ids.SIMULATION_SUMMARY, 'children'),
        Output(Ids.KPI_CONTAINER, 'children', allow_duplicate=True),
        Output(Ids.OLIVE_PRODUCTION_HA, 'children'),
        Output(Ids.OIL_PRODUCTION_HA, 'children'),
        Output(Ids.WATER_NEED_HA, 'children'),
        Output(Ids.OLIVE_PRODUCTION, 'children'),
        Output(Ids.OIL_PRODUCTION, 'children'),
        Output(Ids.WATER_NEED, 'children'),
        Output(Ids.PRODUCTION_DETAILS, 'figure'),
        Output(Ids.WEATHER_IMPACT, 'figure'),
        Output(Ids.WATER_NEEDS, 'figure'),
        Output(Ids.EXTRA_INFO, 'children'),
        Output(Ids.INFERENCE_REQUESTS, 'children')
    ],
    [Input(Ids.SIMULATE_BUTTON, 'n_clicks')],
    [State(Ids.TEMP_SLIDER, 'value'),
     State(Ids.HUMIDITY_SLIDER, 'value'),
     State(Ids.RAINFALL_INPUT, 'value'),
     State(Ids.RADIATION_INPUT, 'value'),
     State(Ids.INFERENCE_COUNTER, 'data')],
    prevent_initial_call='initial_duplicate',
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def update_simulation(n_clicks, temp_range, humidity, rainfall, radiation, counter_data):
    """
    Callback principale per aggiornare tutti i componenti della simulazione
    """
    if n_clicks is None or MODEL_LOADING:
        # Crea grafici vuoti per l'inizializzazione
        empty_growth_fig = go.Figure()
        empty_production_fig = go.Figure()
        empty_summary = html.Div()
        empty_kpis = html.Div()
        return empty_growth_fig, empty_production_fig, empty_summary, empty_kpis, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", {}, {}, {}, "", 0

    try:
        # Inizializza il simulatore
        simulator = EnvironmentalSimulator()

        # Esegui simulazione
        sim_data = simulator.simulate_growth(temp_range, humidity, rainfall, radiation)

        # Crea i grafici
        growth_fig = create_growth_simulation_figure(sim_data)
        production_fig = create_production_impact_figure(sim_data)

        # Calcola KPI
        kpis = calculate_kpis(sim_data)
        kpi_indicators = create_kpi_indicators(kpis)

        # Crea riepilogo
        avg_stress = sim_data['stress_index'].mean()
        estimated_production = simulator.calculate_production_impact(sim_data['stress_index'])

        summary = dbc.Card([
            dbc.CardHeader("Riepilogo Simulazione"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Stress Medio:"),
                        html.P(f"{avg_stress:.2%}"),
                    ], md=4),
                    dbc.Col([
                        html.H6("Produzione Stimata:"),
                        html.P(f"{estimated_production:.1f} kg/albero"),
                    ], md=4),
                    dbc.Col([
                        html.H6("Fase Critica:"),
                        html.P(sim_data.loc[sim_data['stress_index'].idxmax(), 'phase']),
                    ], md=4),
                ]),
            ]),
        ])

        try:
            config = load_config()

            hectares = config['oliveto']['hectares']
            varieties_info = []
            percentages = []

            # Estrai le informazioni dalle varietà configurate
            for variety_config in config['oliveto']['varieties']:
                variety_data = olive_varieties[
                    (olive_varieties['Varietà di Olive'] == variety_config['variety']) &
                    (olive_varieties['Tecnica di Coltivazione'].str.lower() == variety_config['technique'].lower())
                    ]
                if not variety_data.empty:
                    varieties_info.append(variety_data.iloc[0])
                    percentages.append(variety_config['percentage'])

            current_count = counter_data.get('count', 0) + 1

            prediction = make_prediction(weather_data, varieties_info, percentages, hectares, sim_data)

            dcc.Store(id=Ids.INFERENCE_COUNTER, data={'count': current_count})

            # Formattazione output con valori per ettaro e totali
            olive_prod_text_ha = f"{prediction['olive_production']:.0f} kg/ha\n"
            olive_prod_text = f"Totale: {prediction['olive_production_total']:.0f} kg"

            oil_prod_text_ha = f"{prediction['avg_oil_production']:.0f} L/ha\n"
            oil_prod_text = f"Totale: {(prediction['avg_oil_production_total'] * hectares):.0f} L"

            water_need_text_ha = f"{prediction['water_need']:.0f} m³/ha\n"
            water_need_text = f"Totale: {prediction['water_need_total']:.0f} m³"

            # Creazione grafici con il nuovo stile
            details_fig = create_production_details_figure(prediction)
            weather_fig = create_weather_impact_figure(weather_data)
            water_fig = {}  # create_water_needs_figure(prediction)

            # Creazione info extra con il nuovo stile
            extra_info = create_extra_info_component(prediction, varieties_info)

            return (
                growth_fig, production_fig, summary, kpi_indicators,
                olive_prod_text_ha, oil_prod_text_ha, water_need_text_ha,
                olive_prod_text, oil_prod_text, water_need_text,
                details_fig, weather_fig, water_fig, extra_info, f"{current_count}")

        except Exception as e:
            print(f"Errore nell'aggiornamento dashboard: {str(e)}")
            return growth_fig, production_fig, summary, kpi_indicators, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", {}, {}, {}, "", 0

    except Exception as e:
        print(f"Errore nella simulazione: {str(e)}")
        # In caso di errore, ritorna componenti vuoti
        empty_growth_fig = go.Figure()
        empty_production_fig = go.Figure()
        error_summary = dbc.Alert(
            f"Errore durante la simulazione: {str(e)}",
            color="danger"
        )
        empty_kpis = html.Div()
        return empty_growth_fig, empty_production_fig, error_summary, empty_kpis, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", {}, {}, {}, "", 0


@app.callback(
    Output(Ids.SIMULATE_BUTTON, 'disabled'),
    [Input(Ids.TEMP_SLIDER, 'value'),
     Input(Ids.HUMIDITY_SLIDER, 'value'),
     Input(Ids.RAINFALL_INPUT, 'value'),
     Input(Ids.RADIATION_INPUT, 'value')],
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def update_button_state(temp_range, humidity, rainfall, radiation):
    """
    Disabilita il pulsante se i parametri non sono validi
    """
    if None in [temp_range, humidity, rainfall, radiation]:
        return True

    # Verifica range validi
    if not (0 <= humidity <= 100):
        return True
    if not (0 <= rainfall <= 1000):
        return True
    if not (0 <= radiation <= 1200):
        return True

    return False


@app.callback(
    Output(Ids.KPI_CONTAINER, 'children', allow_duplicate=True),
    [Input(Ids.GROWTH_CHART, 'figure'),
     Input(Ids.PRODUCTION_CHART, 'figure')],
    [State(Ids.TEMP_SLIDER, 'value'),
     State(Ids.HUMIDITY_SLIDER, 'value'),
     State(Ids.RAINFALL_INPUT, 'value'),
     State(Ids.RADIATION_INPUT, 'value')],
    prevent_initial_callbacks='initial_duplicate',
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def update_kpis(growth_fig, prod_fig, temp_range, humidity, rainfall, radiation):
    """Aggiorna i KPI quando cambia la simulazione"""

    simulator = EnvironmentalSimulator()
    sim_data = simulator.simulate_growth(temp_range, humidity, rainfall, radiation)

    kpis = calculate_kpis(sim_data)

    return create_kpi_indicators(kpis)


@app.callback(
    Output('growth-simulation-chart', 'style'),
    Input('growth-simulation-chart', 'id'),
    running=[
        (Output(Ids.DASHBOARD_CONTAINER, 'children'),
         [Input('url', 'pathname')],
         lambda x: x == '/')
    ]
)
def update_graph_style(graph_id):
    return {
        'height': '60vh',  # Altezza responsive
        'minHeight': '400px',
        'maxHeight': '800px'
    }


@app.callback(
    Output(Ids.INFERENCE_COUNTER, 'data'),
    [Input(Ids.SIMULATE_BUTTON, 'n_clicks')],
    [State(Ids.INFERENCE_COUNTER, 'data')],
    prevent_initial_call=True
)
def update_inference_counter(n_clicks, current_data):
    if n_clicks is None:
        raise PreventUpdate

    try:
        current_count = current_data.get('count', 0) + 1
        return {'count': current_count}
    except Exception as e:
        print(f"Errore nell'aggiornamento del contatore: {str(e)}")
        return current_data


@app.callback(
    [
        Output(Ids.INFERENCE_REQUESTS, 'children', allow_duplicate=True),
        Output(Ids.PRODUCTION_INFERENCE_REQUESTS, 'children', allow_duplicate=True)
    ],
    [Input(Ids.INFERENCE_COUNTER, 'data')],
    prevent_initial_call=True
)
def display_inference_count(counter_data):
    try:
        count = counter_data.get('count', 0)
        return f"{count}", f"{count}"
    except Exception as e:
        print(f"Errore nella visualizzazione del contatore: {str(e)}")
        return "0", "0"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    env_port = int(os.environ.get('DASH_PORT', 8888))
    env_debug = os.environ.get('DASH_DEBUG', '').lower() == 'true'

    port = args.port if args.port is not None else env_port
    debug = args.debug if args.debug else env_debug

    print(f"Starting server on port {port} with debug={'on' if debug else 'off'}")

    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
