import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import re
import os
import json

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
    simulated_data = pd.read_parquet("./data/simulated_data.parquet")
    weather_data = pd.read_parquet("./data/weather_data_complete.parquet")
    olive_varieties = pd.read_parquet("./data/olive_varieties.parquet")
    if not DEV_MODE:

        print("Caricamento modello e scaler...")
        model = tf.keras.models.load_model('./models/oli_transformer/olive_transformer.keras')

        scaler_temporal = joblib.load('./models/oli_transformer/scaler_temporal.joblib')
        scaler_static = joblib.load('./models/oli_transformer/scaler_static.joblib')
        scaler_y = joblib.load('./models/oli_transformer/scaler_y.joblib')

    else:
        print("Modalità sviluppo attiva - Modelli non caricati")

except Exception as e:
    print(f"Errore nel caricamento: {str(e)}")
    raise e


def clean_column_name(name):
    # Rimuove caratteri speciali e spazi, converte in snake_case e abbrevia
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)  # Rimuove caratteri speciali
    name = name.lower().replace(' ', '_')  # Converte in snake_case

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


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
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

# Layout della dashboard modernizzato e responsive
app.layout = dbc.Container([

    variety2_tooltip,
    variety3_tooltip,
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Dashboard Produzione Olio d'Oliva",
                        className="text-primary text-center mb-3"),
                html.P("Analisi e previsioni della produzione olivicola",
                       className="text-muted text-center")
            ], className="mt-4 mb-4")
        ])
    ]),

    # Main content
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
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
                ], label="Produzione", tab_id="tab-production"),

                dbc.Tab([
                    # Sezione Costi
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Costi di Produzione"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Costi Fissi Annuali", className="mb-3"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([
                                                    html.Strong("Ammortamento impianto: "),
                                                    "€2,000.00/ha"
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Strong("Assicurazione: "),
                                                    "€500.00/ha"
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Strong("Manutenzione: "),
                                                    "€800.00/ha"
                                                ])
                                            ], flush=True)
                                        ], md=6),
                                        dbc.Col([
                                            html.H5("Costi Variabili", className="mb-3"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([
                                                    html.Strong("Manodopera raccolta: "),
                                                    "€0.35/kg olive"
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Strong("Potatura: "),
                                                    "€600.00/ha"
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Strong("Fertilizzanti: "),
                                                    "€400.00/ha"
                                                ])
                                            ], flush=True)
                                        ], md=6)
                                    ])
                                ])
                            ], style=CARD_STYLE)
                        ], md=12)
                    ]),

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
                ], label="Analisi Economica", tab_id="tab-financial"),

                # Tab Configurazione
                dbc.Tab([
                    dbc.Row([
                        # Configurazione Oliveto
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
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("Configurazione Costi", className="text-primary mb-0")
                                ], className="bg-light"),
                                dbc.CardBody([
                                    # Costi Fissi
                                    html.H5("Costi Fissi Annuali", className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Ammortamento impianto (€/ha):", className="fw-bold"),
                                            dbc.Input(
                                                id='cost-ammortamento',
                                                type='number',
                                                value=2000,
                                                min=0,
                                                className="mb-2"
                                            )
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Assicurazione (€/ha):", className="fw-bold"),
                                            dbc.Input(
                                                id='cost-assicurazione',
                                                type='number',
                                                value=500,
                                                min=0,
                                                className="mb-2"
                                            )
                                        ], md=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Manutenzione (€/ha):", className="fw-bold"),
                                            dbc.Input(
                                                id='cost-manutenzione',
                                                type='number',
                                                value=800,
                                                min=0,
                                                className="mb-2"
                                            )
                                        ], md=6)
                                    ], className="mb-4"),

                                    # Costi Variabili
                                    html.H5("Costi Variabili", className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Manodopera raccolta (€/kg):", className="fw-bold"),
                                            dbc.Input(
                                                id='cost-raccolta',
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
                                                id='cost-potatura',
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
                                                id='cost-fertilizzanti',
                                                type='number',
                                                value=400,
                                                min=0,
                                                className="mb-2"
                                            )
                                        ], md=6)
                                    ], className="mb-4"),

                                    # Costi Trasformazione
                                    html.H5("Costi Trasformazione", className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Molitura (€/kg olive):", className="fw-bold"),
                                            dbc.Input(
                                                id='cost-molitura',
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
                                                id='cost-stoccaggio',
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
                                                id='cost-bottiglia',
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
                                                id='cost-etichettatura',
                                                type='number',
                                                value=0.30,
                                                min=0,
                                                step=0.01,
                                                className="mb-2"
                                            )
                                        ], md=6)
                                    ], className="mb-4"),

                                    # Prezzo Vendita
                                    html.H5("Prezzo di Vendita", className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Prezzo vendita olio (€/L):", className="fw-bold"),
                                            dbc.Input(
                                                id='price-olio',
                                                type='number',
                                                value=12.00,
                                                min=0,
                                                step=0.01,
                                                className="mb-2"
                                            )
                                        ], md=6)
                                    ])
                                ])
                            ])
                        ], md=6),
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
                ], label="Configurazione", tab_id="tab-config"),
            ], id="tabs", active_tab="tab-production")
        ], md=12, lg=12)
    ])
], fluid=True, className="px-4 py-3")


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
     State("cost-ammortamento", "value"),
     State("cost-assicurazione", "value"),
     State("cost-manutenzione", "value"),
     State("cost-raccolta", "value"),
     State("cost-potatura", "value"),
     State("cost-fertilizzanti", "value"),
     State("cost-molitura", "value"),
     State("cost-stoccaggio", "value"),
     State("cost-bottiglia", "value"),
     State("cost-etichettatura", "value"),
     State("price-olio", "value")]
)
def save_configuration(n_clicks, hectares, var1, tech1, perc1, var2, tech2, perc2,
                       var3, tech3, perc3, amm, ass, man, rac, pot, fer, mol, sto,
                       bot, eti, price):
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
                'manutenzione': man
            },
            'variable': {
                'raccolta': rac,
                'potatura': pot,
                'fertilizzanti': fer
            },
            'transformation': {
                'molitura': mol,
                'stoccaggio': sto,
                'bottiglia': bot,
                'etichettatura': eti
            },
            'selling_price': price
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

@app.callback(
    [
        Output("hectares-input", "value"),
        Output("variety-1-dropdown", "value"),
        Output("technique-1-dropdown", "value"),
        Output("percentage-1-input", "value"),
        Output("variety-2-dropdown", "value"),
        Output("cost-ammortamento", "value"),
        Output("cost-assicurazione", "value"),
        Output("cost-manutenzione", "value"),
        Output("cost-raccolta", "value"),
        Output("cost-potatura", "value"),
        Output("cost-fertilizzanti", "value"),
        Output("cost-molitura", "value"),
        Output("cost-stoccaggio", "value"),
        Output("cost-bottiglia", "value"),
        Output("cost-etichettatura", "value"),
        Output("price-olio", "value")
    ],
    [Input("tabs", "active_tab")]
)
def load_saved_config(tab):
    if tab != "tab-config":
        return [no_update] * 16

    config = load_config()

    varieties = config['oliveto']['varieties']
    var1 = varieties[0] if len(varieties) > 0 else {"variety": None, "technique": None, "percentage": 0}
    var2 = varieties[1] if len(varieties) > 1 else {"variety": None, "technique": None, "percentage": 0}

    costs = config['costs']

    return [
        config['oliveto']['hectares'],
        var1["variety"],
        var1["technique"],
        var1["percentage"],
        var2["variety"],
        costs['fixed']['ammortamento'],
        costs['fixed']['assicurazione'],
        costs['fixed']['manutenzione'],
        costs['variable']['raccolta'],
        costs['variable']['potatura'],
        costs['variable']['fertilizzanti'],
        costs['transformation']['molitura'],
        costs['transformation']['stoccaggio'],
        costs['transformation']['bottiglia'],
        costs['transformation']['etichettatura'],
        costs['selling_price']
    ]


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


@app.callback(
    [
        Output('technique-2-dropdown', 'disabled'),
        Output('percentage-2-input', 'disabled'),
        Output('technique-3-dropdown', 'disabled'),
        Output('percentage-3-input', 'disabled'),
        Output('percentage-warning', 'children'),
        Output('technique-2-dropdown', 'value'),
        Output('technique-3-dropdown', 'value'),
        Output('percentage-2-input', 'value'),
        Output('percentage-3-input', 'value')
    ],
    [
        Input('variety-2-dropdown', 'value'),
        Input('variety-3-dropdown', 'value'),
        Input('percentage-1-input', 'value'),
        Input('percentage-2-input', 'value'),
        Input('percentage-3-input', 'value')
    ]
)
def manage_percentages_and_techniques(variety2, variety3, perc1, perc2, perc3):
    perc1 = perc1 or 0
    perc2 = perc2 or 0
    perc3 = perc3 or 0
    total = perc1 + perc2 + perc3

    # Gestione varietà 2
    disable_2 = variety2 is None
    technique2_value = None if disable_2 else 'Tradizionale'
    percentage2_value = 0 if disable_2 else (perc2 if perc2 else 0)

    # Gestione varietà 3
    disable_3 = variety3 is None or variety2 is None
    technique3_value = None if disable_3 else 'Tradizionale'
    percentage3_value = 0 if disable_3 else (perc3 if perc3 else 0)

    # Gestione warning percentuali - ora mostra sempre il warning se non è 100%
    warning = ""
    if total != 100:
        if total > 100:
            warning = f"La somma delle percentuali è {total}% (dovrebbe essere 100%)"
        else:
            warning = f"La somma delle percentuali è {total}% (dovrebbe essere 100%)"

    return (
        disable_2,
        disable_2,
        disable_3,
        disable_3,
        warning,
        technique2_value,
        technique3_value,
        percentage2_value,
        percentage3_value
    )


# Aggiunta callback per resettare i valori della varietà 3 quando la varietà 2 viene deselezionata
@app.callback(
    [
        Output('variety-3-dropdown', 'value'),
        Output('variety-3-dropdown', 'disabled')
    ],
    [
        Input('variety-2-dropdown', 'value')
    ]
)
def manage_variety3_availability(variety2):
    if variety2 is None:
        return None, True
    return no_update, False


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


if __name__ == '__main__':
    app.run_server(debug=True)
