import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import tensorflow as tf
import joblib
import dash_bootstrap_components as dbc
import os
import json
from utils.helpers import clean_column_name
from dashboard.environmental_simulator import *

DEV_MODE = os.getenv('DEV_MODE', 'True').lower() == 'true'

CONFIG_FILE = 'olive_config.json'


def load_config():
    default_config = {
        'oliveto': {
            'hectares': 1,
            'varieties': [
                {
                    'variety': olive_varieties['Varietà di Olive'].iloc[0],
                    'technique': 'Tradizionale',
                    'percentage': 100
                }
            ]
        },
        'costs': {
            'fixed': {
                'ammortamento': 2000,
                'assicurazione': 500,
                'manutenzione': 800
            },
            'variable': {
                'raccolta': 0.35,
                'potatura': 600,
                'fertilizzanti': 400
            },
            'transformation': {
                'molitura': 0.15,
                'stoccaggio': 0.20,
                'bottiglia': 1.20,
                'etichettatura': 0.30
            },
            'selling_price': 12.00
        }
    }

    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return default_config
    except Exception as e:
        print(f"Errore nel caricamento della configurazione: {e}")
        return default_config


# Funzione per salvare la configurazione
def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Errore nel salvataggio della configurazione: {e}")
        return False


# Caricamento dati
print("Inizializzazione della dashboard...")

try:
    # Caricamento dati e modello
    print("Caricamento dati...")
    simulated_data = pd.read_parquet("../sources/simulated_data.parquet")
    weather_data = pd.read_parquet("../sources/weather_data_complete.parquet")
    olive_varieties = pd.read_parquet("../sources/olive_varieties.parquet")
    if not DEV_MODE:

        print("Caricamento modello e scaler...")
        model = tf.keras.models.load_model('./models/oli_transformer/olive_transformer.keras')

        scaler_temporal = joblib.load('../old_model_train/models/oli_transformer/scaler_temporal.joblib')
        scaler_static = joblib.load('../old_model_train/models/oli_transformer/scaler_static.joblib')
        scaler_y = joblib.load('../old_model_train/models/oli_transformer/scaler_y.joblib')

    else:
        print("Modalità sviluppo attiva - Modelli non caricati")

except Exception as e:
    print(f"Errore nel caricamento: {str(e)}")
    raise e


# Funzioni di supporto per la dashboard
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
                          ) / 4 * percentage / 100 * hectares

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


def mock_make_prediction(weather_data, varieties_info, percentages, hectares):
    """
    Versione mock della funzione make_prediction che simula risultati realistici
    basati sui dati reali delle varietà e tiene conto degli ettari
    """
    try:
        # Calcola la produzione di olive basata sui dati reali delle varietà per ettaro
        olive_production_per_ha = sum(
            variety_info['Produzione (tonnellate/ettaro)'] * 1000 * (percentage / 100)
            for variety_info, percentage in zip(varieties_info, percentages)
        )

        # Applica il fattore ettari
        olive_production = olive_production_per_ha * hectares

        # Aggiungi una variabilità realistica basata sulle condizioni meteorologiche recenti
        recent_weather = weather_data.tail(3)
        weather_factor = 1.0

        # Temperatura influenza la produzione
        avg_temp = recent_weather['temp'].mean()
        if avg_temp < 15:
            weather_factor *= 0.9
        elif avg_temp > 25:
            weather_factor *= 0.95

        # Precipitazioni influenzano la produzione
        total_precip = recent_weather['precip'].sum()
        if total_precip < 30:  # Siccità
            weather_factor *= 0.85
        elif total_precip > 200:  # Troppa pioggia
            weather_factor *= 0.9

        # Radiazione solare influenza la produzione
        avg_solar = recent_weather['solarradiation'].mean()
        if avg_solar < 150:
            weather_factor *= 0.95

        # Applica il fattore meteorologico alla produzione totale
        olive_production = olive_production * weather_factor

        # Calcola la produzione di olio basata sulle rese delle varietà
        oil_per_ha = 0
        oil_total = 0

        for variety_info, percentage in zip(varieties_info, percentages):
            # Calcolo della produzione di olio per ettaro per questa varietà
            variety_oil_per_ha = variety_info['Produzione Olio (litri/ettaro)'] * (percentage / 100)
            oil_per_ha += variety_oil_per_ha

            # Calcolo della produzione totale di olio per questa varietà
            variety_oil_total = variety_oil_per_ha * hectares
            oil_total += variety_oil_total

        # Applica il fattore meteorologico anche alla produzione di olio
        oil_total = oil_total * weather_factor
        oil_per_ha = oil_per_ha * weather_factor

        # Calcola il fabbisogno idrico considerando la stagione attuale e gli ettari
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

        # Calcolo del fabbisogno idrico per ettaro
        water_need_per_ha = sum(
            variety_info[season_water_need[current_season]] * (percentage / 100)
            for variety_info, percentage in zip(varieties_info, percentages)
        )

        # Applica il fattore ettari al fabbisogno idrico
        total_water_need = water_need_per_ha * hectares

        # Prepara i dettagli per varietà
        variety_details = []
        for variety_info, percentage in zip(varieties_info, percentages):
            variety_prod_per_ha = variety_info['Produzione (tonnellate/ettaro)'] * 1000 * (percentage / 100)
            variety_prod_total = variety_prod_per_ha * hectares * weather_factor

            # Calcolo corretto dell'olio per varietà
            variety_oil_per_ha = variety_info['Produzione Olio (litri/ettaro)'] * (percentage / 100)
            variety_oil_total = variety_oil_per_ha * hectares * weather_factor

            variety_details.append({
                'variety': variety_info['Varietà di Olive'],
                'percentage': percentage,
                'production_per_ha': variety_prod_per_ha,
                'production_total': variety_prod_total,
                'oil_per_ha': variety_oil_per_ha,
                'oil_total': variety_oil_total,
                'water_need': variety_info[season_water_need[current_season]] * hectares
            })

        return {
            'olive_production': olive_production_per_ha,
            'olive_production_total': olive_production,
            'min_oil_production': oil_per_ha * 0.9,
            'max_oil_production': oil_per_ha * 1.1,
            'avg_oil_production': oil_per_ha,
            'avg_oil_production_total': oil_total,
            'water_need': water_need_per_ha,
            'water_need_total': total_water_need,
            'variety_details': variety_details,
            'hectares': hectares
        }

    except Exception as e:
        print(f"Errore nella funzione mock_make_prediction: {str(e)}")
        import traceback
        print("Traceback completo:")
        print(traceback.format_exc())
        raise e


def make_prediction(weather_data, varieties_info, percentages, hectares):
    if DEV_MODE:
        return mock_make_prediction(weather_data, varieties_info, percentages, hectares)
    else:
        """Effettua una predizione usando il modello."""
        try:
            print("Inizio della funzione make_prediction")

            # Prepara i dati meteorologici mensili
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

            print(f"Shape dei dati meteorologici mensili: {monthly_stats.shape}")

            # Definisci la dimensione della finestra temporale
            window_size = 41

            # Prendi gli ultimi window_size mesi di dati
            if len(monthly_stats) >= window_size:
                temporal_data = monthly_stats[['temp_mean', 'precip_sum', 'solar_energy_sum']].values[-window_size:]
            else:
                raise ValueError(f"Non ci sono abbastanza dati meteorologici. Necessari almeno {window_size} mesi.")

            print(f"Shape dei dati temporali prima della trasformazione: {temporal_data.shape}")

            temporal_data = scaler_temporal.transform(temporal_data)
            print(f"Shape dei dati temporali dopo la trasformazione: {temporal_data.shape}")

            temporal_data = np.expand_dims(temporal_data, axis=0)
            print(f"Shape finale dei dati temporali: {temporal_data.shape}")

            all_varieties = olive_varieties['Varietà di Olive'].unique()
            varieties = [clean_column_name(variety) for variety in all_varieties]

            # Prepara i dati statici
            print("Preparazione dei dati statici")
            static_data = prepare_static_features_multiple(varieties_info, percentages, hectares, varieties)

            # Verifica che il numero di feature statiche sia corretto
            if static_data.shape[1] != scaler_static.n_features_in_:
                print("ATTENZIONE: Il numero di feature statiche non corrisponde a quello atteso dallo scaler!")
                print(f"Feature generate: {static_data.shape[1]}, Feature attese: {scaler_static.n_features_in_}")

            static_data = scaler_static.transform(static_data)
            print(f"Shape dei dati statici dopo la trasformazione: {static_data.shape}")

            # Effettua la predizione
            print("Effettuazione della predizione")
            prediction = model.predict({'temporal': temporal_data, 'static': static_data})
            prediction = scaler_y.inverse_transform(prediction)[0]

            # Calcola i dettagli per varietà
            variety_details = []
            for variety_info, percentage in zip(varieties_info, percentages):
                # Calcoli specifici per varietà
                prod_per_ha = variety_info['Produzione (tonnellate/ettaro)'] * 1000
                oil_per_ha = variety_info['Produzione Olio (litri/ettaro)']
                water_need = (
                                     variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'] +
                                     variety_info['Fabbisogno Acqua Estate (m³/ettaro)'] +
                                     variety_info['Fabbisogno Acqua Autunno (m³/ettaro)'] +
                                     variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']
                             ) / 4

                variety_details.append({
                    'variety': variety_info['Varietà di Olive'],
                    'percentage': percentage,
                    'production_per_ha': prod_per_ha,
                    'oil_per_ha': oil_per_ha,
                    'water_need': water_need
                })

            return {
                'olive_production': prediction[0],
                'min_oil_production': prediction[1],
                'max_oil_production': prediction[2],
                'avg_oil_production': prediction[3],
                'water_need': prediction[4],
                'variety_details': variety_details
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


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    prevent_initial_callbacks='initial_duplicate'
)

# Stili comuni
CARD_STYLE = {
    "height": "100%",
    "margin-bottom": "15px"
}

CARD_BODY_STYLE = {
    "padding": "15px"
}

# Modifiche al layout - aggiungi tooltips per chiarire la funzionalità
variety2_tooltip = dbc.Tooltip(
    "Seleziona una seconda varietà per creare un mix",
    target="variety-2-dropdown",
    placement="top"
)

variety3_tooltip = dbc.Tooltip(
    "Seleziona una terza varietà per completare il mix",
    target="variety-3-dropdown",
    placement="top"
)


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
                                                {'label': 'Tradizionale', 'value': 'Tradizionale'},
                                                {'label': 'Intensiva', 'value': 'Intensiva'},
                                                {'label': 'Superintensiva', 'value': 'Superintensiva'}
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
                                                {'label': 'Tradizionale', 'value': 'Tradizionale'},
                                                {'label': 'Intensiva', 'value': 'Intensiva'},
                                                {'label': 'Superintensiva', 'value': 'Superintensiva'}
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
                                                {'label': 'Tradizionale', 'value': 'Tradizionale'},
                                                {'label': 'Intensiva', 'value': 'Intensiva'},
                                                {'label': 'Superintensiva', 'value': 'Superintensiva'}
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


def create_production_tab():
    return dbc.Tab([
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


# Layout della dashboard modernizzato e responsive
app.layout = dbc.Container([
    dcc.Location(id='_pages_location'),
    variety2_tooltip,
    variety3_tooltip,
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Dashboard Produzione Olio d'Oliva",
                        className="text-primary text-center mb-3")
            ], className="mt-4 mb-4")
        ])
    ]),

    # Main content
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                create_production_tab(),
                create_environmental_simulation_tab(),
                create_economic_analysis_tab(),
                create_configuration_tab(),
            ], id="tabs", active_tab="tab-production")
        ], md=12, lg=12)
    ])
], fluid=True, className="px-4 py-3")


def create_extra_info_component(prediction, varieties_info):
    """Crea il componente delle informazioni dettagliate con il nuovo stile"""
    cards = []

    # Card per ogni varietà
    for detail, variety_info in zip(prediction['variety_details'], varieties_info):
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
                                    f"{detail['oil_total'] / detail['production_total']:.3f} %"
                                ], className="px-2 py-1")
                            ], flush=True)
                        ])
                    ], md=4),
                    # Colonna Produzione/ha
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
                    # Colonna Rese
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
                            f"{(prediction['avg_oil_production_total'] / prediction['olive_production_total']):.3f}%"
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
     State("perc-vendita-diretta", "value")]
)
def save_configuration(n_clicks, hectares, var1, tech1, perc1, var2, tech2, perc2,
                       var3, tech3, perc3, amm, ass, man, cert, rac, pot, fer, irr,
                       mol, sto, bot, eti, mark, comm, price, perc_dir):
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
        }
    }

    if save_config(config):
        return dbc.Alert(
            "Configurazione salvata con successo!",
            color="success",
            duration=4000,
            is_open=True
        )
    else:
        return dbc.Alert(
            "Errore nel salvataggio della configurazione",
            color="danger",
            duration=4000,
            is_open=True
        )


# Aggiorna la configurazione dei grafici per essere più responsive
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


from dash import no_update


@app.callback(
    [
        # Outputs per i costi e configurazioni base (16 outputs)
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
        Output("percentage-warning", "children"),
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
        Input("percentage-1-input", "value"),
        Input("percentage-2-input", "value"),
        Input("percentage-3-input", "value"),
        Input('_pages_location', 'pathname')  # Aggiunto per trigger all'avvio
    ]
)
def unified_config_manager(active_tab, variety2, variety3, perc1, perc2, perc3, pathname):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    try:
        # Carica sempre la configurazione
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

        # Calcolo warning percentuali
        total = sum(v["percentage"] for v in varieties if v["variety"] is not None)
        warning = ""
        if total != 100:
            warning = f"La somma delle percentuali è {total}% (dovrebbe essere 100%)"

        return [
            # Configurazioni base (16 valori)
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
            warning,
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
        print(f"Errore in unified_config_manager: {str(e)}")
        # In caso di errore, ritorna no_update per tutti i valori
        return [no_update] * 32


# Modifica il callback update_dashboard per utilizzare il nuovo layout dei grafici
@app.callback(
    [Output('olive-production_ha', 'children'),
     Output('oil-production_ha', 'children'),
     Output('water-need_ha', 'children'),
     Output('olive-production', 'children'),
     Output('oil-production', 'children'),
     Output('water-need', 'children'),
     Output('production-details', 'figure'),
     Output('weather-impact', 'figure'),
     Output('water-needs', 'figure'),
     Output('extra-info', 'children')],
    [Input('variety-1-dropdown', 'value'),
     Input('technique-1-dropdown', 'value'),
     Input('percentage-1-input', 'value'),
     Input('variety-2-dropdown', 'value'),
     Input('technique-2-dropdown', 'value'),
     Input('percentage-2-input', 'value'),
     Input('variety-3-dropdown', 'value'),
     Input('technique-3-dropdown', 'value'),
     Input('percentage-3-input', 'value'),
     Input('hectares-input', 'value')]
)
def update_dashboard(variety1, tech1, perc1, variety2, tech2, perc2,
                     variety3, tech3, perc3, hectares):
    if not variety1 or not tech1 or perc1 is None or hectares is None or hectares <= 0:
        return "N/A", "N/A", "N/A", {}, {}, {}, ""

    # Raccogli le informazioni delle varietà
    varieties_info = []
    percentages = []

    # Prima varietà
    variety_data = olive_varieties[
        (olive_varieties['Varietà di Olive'] == variety1) &
        (olive_varieties['Tecnica di Coltivazione'] == tech1)
        ]
    if not variety_data.empty:
        varieties_info.append(variety_data.iloc[0])
        percentages.append(perc1)

    # Seconda varietà
    if variety2 and tech2 and perc2:
        variety_data = olive_varieties[
            (olive_varieties['Varietà di Olive'] == variety2) &
            (olive_varieties['Tecnica di Coltivazione'] == tech2)
            ]
        if not variety_data.empty:
            varieties_info.append(variety_data.iloc[0])
            percentages.append(perc2)

    # Terza varietà
    if variety3 and tech3 and perc3:
        variety_data = olive_varieties[
            (olive_varieties['Varietà di Olive'] == variety3) &
            (olive_varieties['Tecnica di Coltivazione'] == tech3)
            ]
        if not variety_data.empty:
            varieties_info.append(variety_data.iloc[0])
            percentages.append(perc3)

    try:
        prediction = make_prediction(weather_data, varieties_info, percentages, hectares)

        # Formattazione output principale
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
        water_fig = create_water_needs_figure(prediction)

        # Creazione info extra con il nuovo stile
        extra_info = create_extra_info_component(prediction, varieties_info)

        return (
            olive_prod_text_ha, oil_prod_text_ha, water_need_text_ha,
            olive_prod_text, oil_prod_text, water_need_text,
            details_fig, weather_fig, water_fig, extra_info)

    except Exception as e:
        print(f"Errore nell'aggiornamento dashboard: {str(e)}")
        return "Errore", "Errore", "Errore", {}, {}, {}, "Errore nella generazione dei dati"


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
                'Mese': month,
                'Varietà': detail['variety'],
                'Fabbisogno': water_need * (detail['percentage'] / 100)
            })

    water_df = pd.DataFrame(water_data)
    fig = px.bar(
        water_df,
        x='Mese',
        y='Fabbisogno',
        color='Varietà',
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
    [Output('growth-simulation-chart', 'figure'),
     Output('production-simulation-chart', 'figure'),
     Output('simulation-summary', 'children'),
     Output('kpi-container', 'children', allow_duplicate=True)],
    [Input('simulate-btn', 'n_clicks')],
    [State('temp-slider', 'value'),
     State('humidity-slider', 'value'),
     State('rainfall-input', 'value'),
     State('radiation-input', 'value')],
    prevent_initial_call='initial_duplicate'
)
def update_simulation(n_clicks, temp_range, humidity, rainfall, radiation):
    """
    Callback principale per aggiornare tutti i componenti della simulazione
    """
    if n_clicks is None:
        # Crea grafici vuoti per l'inizializzazione
        empty_growth_fig = go.Figure()
        empty_production_fig = go.Figure()
        empty_summary = html.Div()
        empty_kpis = html.Div()
        return empty_growth_fig, empty_production_fig, empty_summary, empty_kpis

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

        return growth_fig, production_fig, summary, kpi_indicators

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
        return empty_growth_fig, empty_production_fig, error_summary, empty_kpis


# Aggiungiamo un callback per gestire l'abilitazione del pulsante di simulazione
@app.callback(
    Output('simulate-btn', 'disabled'),
    [Input('temp-slider', 'value'),
     Input('humidity-slider', 'value'),
     Input('rainfall-input', 'value'),
     Input('radiation-input', 'value')]
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
    if not (0 <= rainfall <= 500):
        return True
    if not (0 <= radiation <= 1000):
        return True

    return False


# Callback per aggiornare il layout dei grafici in base alle dimensioni della finestra
'''@app.clientside_callback(
    """
    function(value) {
        return {
            'height': window.innerHeight * 0.6
        }
    }
    """,
    Output('growth-simulation-chart', 'style'),
    Input('growth-simulation-chart', 'id')
)
'''


@app.callback(
    Output('kpi-container', 'children', allow_duplicate=True),
    [Input('growth-simulation-chart', 'figure'),
     Input('production-simulation-chart', 'figure')],
    [State('temp-slider', 'value'),
     State('humidity-slider', 'value'),
     State('rainfall-input', 'value'),
     State('radiation-input', 'value')],
    prevent_initial_callbacks='initial_duplicate'
)
def update_kpis(growth_fig, prod_fig, temp_range, humidity, rainfall, radiation):
    """Aggiorna i KPI quando cambia la simulazione"""

    # Ricalcola la simulazione
    simulator = EnvironmentalSimulator()
    sim_data = simulator.simulate_growth(temp_range, humidity, rainfall, radiation)

    # Calcola i KPI
    kpis = calculate_kpis(sim_data)

    # Crea gli indicatori
    return create_kpi_indicators(kpis)


@app.callback(
    Output('growth-simulation-chart', 'style'),
    Input('growth-simulation-chart', 'id')
)
def update_graph_style(graph_id):
    return {
        'height': '60vh',  # Altezza responsive
        'min-height': '400px',
        'max-height': '800px'
    }

if __name__ == '__main__':
    app.run_server(debug=True)
