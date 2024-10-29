import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import re

@tf.keras.utils.register_keras_serializable()
class WarmUpLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate=1e-3, warmup_steps=1000, decay_steps=10000):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        warmup_pct = tf.cast(step, tf.float32) / self.warmup_steps
        warmup_lr = self.initial_learning_rate * warmup_pct
        decay_factor = tf.pow(0.1, tf.cast(step, tf.float32) / self.decay_steps)
        decayed_lr = self.initial_learning_rate * decay_factor
        final_lr = tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
        return final_lr

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps
        }

# Definizione delle classi del modello
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TemporalAugmentation(tf.keras.layers.Layer):
    def __init__(self, noise_factor=0.03, **kwargs):
        super().__init__(**kwargs)
        self.noise_factor = noise_factor

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.noise_factor
            )
            return inputs + noise
        return inputs

class EnhancedTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            value_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.residual_attention = tf.keras.layers.Dense(d_model, activation='sigmoid')

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        residual_weights = self.residual_attention(inputs)
        out1 = self.layernorm1(inputs + residual_weights * attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TemporalPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_pooling = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )
        self.temporal_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.max_pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, training=None):
        att_output = self.attention_pooling(inputs, inputs)
        avg_output = self.temporal_pooling(inputs)
        max_output = self.max_pooling(inputs)
        att_output = tf.reduce_mean(att_output, axis=1)
        return self.concat([att_output, avg_output, max_output])

@tf.keras.utils.register_keras_serializable()
class OliveOilTransformer(tf.keras.Model):
    def __init__(self, temporal_shape=None, static_shape=None, num_outputs=None,
                 d_model=128, num_heads=8, ff_dim=256, num_transformer_blocks=4,
                 mlp_units=[256, 128, 64], dropout=0.2, **kwargs):
        super(OliveOilTransformer, self).__init__(**kwargs)

        self.temporal_shape = temporal_shape
        self.static_shape = static_shape
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout

        if temporal_shape is not None and static_shape is not None and num_outputs is not None:
            self.build_model()

    def build_model(self):
        # Input layers
        self.temporal_input = tf.keras.layers.Input(shape=self.temporal_shape, name='temporal_input')
        self.static_input = tf.keras.layers.Input(shape=self.static_shape, name='static_input')

        # Input normalization
        self.temporal_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.static_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Data Augmentation
        self.temporal_augmentation = TemporalAugmentation(noise_factor=0.03)

        # Temporal path
        self.temporal_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_model//2, activation='gelu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.d_model, activation='gelu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        ])

        self.pos_encoding = PositionalEncoding(position=self.temporal_shape[0], d_model=self.d_model)

        # Transformer blocks
        self.transformer_blocks = [
            EnhancedTransformerBlock(self.d_model, self.num_heads, self.ff_dim, self.dropout_rate)
            for _ in range(self.num_transformer_blocks)
        ]

        # Temporal pooling
        self.temporal_pooling = TemporalPoolingLayer(
            num_heads=self.num_heads,
            key_dim=self.d_model//4
        )

        # Static path
        self.static_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='gelu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(128, activation='gelu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(64, activation='gelu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        ])

        # Feature fusion
        self.fusion_layer = tf.keras.layers.Concatenate()

        # MLP head
        self.mlp_layers = []
        for units in self.mlp_units:
            self.mlp_layers.extend([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(units, activation="gelu",
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
                tf.keras.layers.Dropout(self.dropout_rate)
            ])

        # Output layer
        self.final_layer = tf.keras.layers.Dense(
            self.num_outputs,
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        # Build model
        temporal_encoded = self.encode_temporal(self.temporal_input, training=True)
        static_encoded = self.encode_static(self.static_input)
        combined = self.fusion_layer([temporal_encoded, static_encoded])

        x = combined
        for layer in self.mlp_layers:
            x = layer(x)

        outputs = self.final_layer(x)

        self._model = tf.keras.Model(
            inputs={'temporal': self.temporal_input, 'static': self.static_input},
            outputs=outputs
        )

    def encode_temporal(self, x, training=None):
        x = self.temporal_normalization(x)
        x = self.temporal_augmentation(x, training=training)
        x = self.temporal_projection(x)
        x = self.pos_encoding(x)

        skip_connection = x
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = tf.keras.layers.Add()([x, skip_connection])

        return self.temporal_pooling(x)

    def encode_static(self, x):
        x = self.static_normalization(x)
        return self.static_encoder(x)

    def call(self, inputs, training=None):
        temporal_input = inputs['temporal']
        static_input = inputs['static']

        temporal_encoded = self.encode_temporal(temporal_input, training)
        static_encoded = self.encode_static(static_input)

        combined = self.fusion_layer([temporal_encoded, static_encoded])

        x = combined
        for layer in self.mlp_layers:
            x = layer(x, training=training)

        return self.final_layer(x)

    def model(self):
        return self._model

    def get_config(self):
        config = super().get_config()
        config.update({
            "temporal_shape": self.temporal_shape,
            "static_shape": self.static_shape,
            "num_outputs": self.num_outputs,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_units": self.mlp_units,
            "dropout": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Caricamento dati e modello
print("Caricamento dati...")
simulated_data = pd.read_parquet("./data/simulated_data.parquet")
weather_data = pd.read_parquet("./data/weather_data_complete.parquet")
olive_varieties = pd.read_parquet("./data/olive_varieties.parquet")

print("Caricamento modello e scaler...")
model = tf.keras.models.load_model('./models/oli_transformer/olive_transformer.keras',
                                   custom_objects={
                                       'OliveOilTransformer': OliveOilTransformer,
                                       'PositionalEncoding': PositionalEncoding,
                                       'TemporalAugmentation': TemporalAugmentation,
                                       'EnhancedTransformerBlock': EnhancedTransformerBlock,
                                       'TemporalPoolingLayer': TemporalPoolingLayer,
                                       'WarmUpLearningRateSchedule': WarmUpLearningRateSchedule
                                   })

scaler_temporal = joblib.load('./models/oli_transformer/scaler_temporal.joblib')
scaler_static = joblib.load('./models/oli_transformer/scaler_static.joblib')
scaler_y = joblib.load('./models/oli_transformer/scaler_y.joblib')

def prepare_monthly_weather_stats(weather_data):
    """Prepara le statistiche mensili dal weather_data."""
    monthly_stats = weather_data.groupby(['year', 'month']).agg({
        'temp': 'mean',
        'precip': 'sum',
        'solarradiation': 'sum'  # useremo questo come proxy per solar_energy_sum
    }).reset_index()

    monthly_stats.columns = ['year', 'month', 'temp_mean', 'precip_sum', 'solar_energy_sum']
    return monthly_stats

def prepare_static_features(variety_info, hectares):
    """Prepara le feature statiche nello stesso formato usato durante il training."""
    # Inizializza un array con il numero di ettari
    static_features = [hectares]

    # Aggiungi le feature della varietà
    variety_name = variety_info['Varietà di Olive'].lower().replace(" ", "_").replace("'", "")
    technique = variety_info['Tecnica di Coltivazione'].lower()

    # Feature di base della varietà
    variety_features = {
        f'{variety_name}_prod_t_ha': variety_info['Produzione (tonnellate/ettaro)'],
        f'{variety_name}_oil_prod_t_ha': variety_info['Produzione Olio (tonnellate/ettaro)'],
        f'{variety_name}_oil_prod_l_ha': variety_info['Produzione Olio (litri/ettaro)'],
        f'{variety_name}_min_yield_pct': variety_info['Min % Resa'],
        f'{variety_name}_max_yield_pct': variety_info['Max % Resa'],
        f'{variety_name}_min_oil_prod_l_ha': variety_info['Min Produzione Olio (litri/ettaro)'],
        f'{variety_name}_max_oil_prod_l_ha': variety_info['Max Produzione Olio (litri/ettaro)'],
        f'{variety_name}_avg_oil_prod_l_ha': variety_info['Media Produzione Olio (litri/ettaro)'],
        f'{variety_name}_l_per_t': variety_info['Litri per Tonnellata'],
        f'{variety_name}_min_l_per_t': variety_info['Min Litri per Tonnellata'],
        f'{variety_name}_max_l_per_t': variety_info['Max Litri per Tonnellata'],
        f'{variety_name}_avg_l_per_t': variety_info['Media Litri per Tonnellata']
    }

    # Aggiungi le feature binarie per le tecniche di coltivazione
    for tech in ['tradizionale', 'intensiva', 'superintensiva']:
        variety_features[f'{variety_name}_{tech}'] = 1 if technique == tech else 0

    # Converti il dizionario in lista mantenendo l'ordine usato durante il training
    static_features.extend([variety_features[key] for key in sorted(variety_features.keys())])

    return np.array(static_features)

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
def prepare_prediction_data(weather_data, varieties_info, percentages, hectares):
    """
    Prepara i dati per la predizione con multiple varietà.

    Args:
        weather_data: DataFrame con i dati meteorologici
        varieties_info: Lista di Series con le informazioni delle varietà selezionate
        percentages: Lista con le percentuali per ogni varietà
        hectares: Numero di ettari totali
    """
    # Prepara dati temporali
    monthly_stats = weather_data.groupby(['year', 'month']).agg({
        'temp': 'mean',
        'precip': 'sum',
        'solarradiation': 'sum'
    }).reset_index()

    temporal_features = ['temp_mean', 'precip_sum', 'solar_energy_sum']
    temporal_data = monthly_stats.rename(columns={
        'temp': 'temp_mean',
        'precip': 'precip_sum',
        'solarradiation': 'solar_energy_sum'
    })[temporal_features].values[-41:]  # Ultimi 41 timestep come nel training

    temporal_data = scaler_temporal.transform(temporal_data)
    temporal_data = np.expand_dims(temporal_data, axis=0)

    # Prepara dati statici per tutte le varietà
    static_data = prepare_static_features_multiple(varieties_info, percentages, hectares)
    static_data = np.expand_dims(static_data, axis=0)
    static_data = scaler_static.transform(static_data)

    return {'temporal': temporal_data, 'static': static_data}

def prepare_static_features_multiple(varieties_info, percentages, hectares):
    """
    Prepara le feature statiche per multiple varietà seguendo la struttura esatta della simulazione.

    Args:
        varieties_info: Lista di Series con le informazioni delle varietà selezionate
        percentages: Lista con le percentuali per ogni varietà
        hectares: Numero di ettari totali
    """
    # Inizializza un dizionario per tutte le varietà possibili
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
    } for variety in olive_varieties['Varietà di Olive'].unique()}

    # Aggiorna i dati per le varietà selezionate
    for variety_info, percentage in zip(varieties_info, percentages):
        variety_name = clean_column_name(variety_info['Varietà di Olive'])
        technique = clean_column_name(variety_info['Tecnica di Coltivazione'])

        # Base production calculations
        annual_prod = variety_info['Produzione (tonnellate/ettaro)'] * 1000 * percentage/100 * hectares
        min_oil_prod = annual_prod * variety_info['Min Litri per Tonnellata'] / 1000
        max_oil_prod = annual_prod * variety_info['Max Litri per Tonnellata'] / 1000
        avg_oil_prod = annual_prod * variety_info['Media Litri per Tonnellata'] / 1000

        # Water need calculation
        base_water_need = (
                                  variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Estate (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Autunno (m³/ettaro)'] +
                                  variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']
                          ) / 4 * percentage/100 * hectares

        variety_data[variety_name].update({
            'tech': technique,
            'pct': percentage/100,
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
            'olive_prod': annual_prod,
            'min_oil_prod': min_oil_prod,
            'max_oil_prod': max_oil_prod,
            'avg_oil_prod': avg_oil_prod,
            'water_need': base_water_need
        })

    # Crea il vettore delle feature nell'ordine esatto
    static_features = [hectares]  # Inizia con gli ettari

    # Appiattisci i dati delle varietà mantenendo l'ordine esatto
    flattened_variety_data = {
        f'{variety}_{key}': value
        for variety, data in sorted(variety_data.items())  # Ordina per nome varietà
        for key, value in sorted(data.items())  # Ordina per nome feature
    }

    # Aggiungi le feature nell'ordine corretto
    static_features.extend([flattened_variety_data[key] for key in sorted(flattened_variety_data.keys())])

    return np.array(static_features)

def make_prediction(weather_data, varieties_info, percentages, hectares):
    """Effettua una predizione usando il modello."""
    try:
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

        # Prendi gli ultimi 41 mesi di dati
        temporal_data = monthly_stats[['temp_mean', 'precip_sum', 'solar_energy_sum']].values[-41:]
        temporal_data = scaler_temporal.transform(temporal_data)
        temporal_data = np.expand_dims(temporal_data, axis=0)

        # Prepara i dati statici
        static_data = prepare_static_features_multiple(varieties_info, percentages, hectares)
        static_data = np.expand_dims(static_data, axis=0)
        static_data = scaler_static.transform(static_data)

        # Effettua la predizione
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
        raise e

# Definizione del layout della dashboard
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout della dashboard
variety_options = [
    {'label': v, 'value': v}
    for v in sorted(olive_varieties['Varietà di Olive'].unique())
]
technique_options = [
    {'label': t, 'value': t}
    for t in sorted(olive_varieties['Tecnica di Coltivazione'].unique())
]

# Layout della dashboard aggiornato
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Dashboard Produzione Olio d'Oliva", className="text-center mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Composizione Oliveto"),
                dbc.CardBody([
                    # Prima varietà (obbligatoria)
                    html.Div([
                        html.H6("Varietà 1"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Seleziona Varietà:"),
                                dcc.Dropdown(
                                    id='variety-1-dropdown',
                                    options=[{'label': v, 'value': v} for v in olive_varieties['Varietà di Olive'].unique()],
                                    value=olive_varieties['Varietà di Olive'].iloc[0]
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Tecnica:"),
                                dcc.Dropdown(
                                    id='technique-1-dropdown',
                                    options=[
                                        {'label': 'Tradizionale', 'value': 'Tradizionale'},
                                        {'label': 'Intensiva', 'value': 'Intensiva'},
                                        {'label': 'Superintensiva', 'value': 'Superintensiva'}
                                    ],
                                    value='Tradizionale'
                                ),
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Percentuale (%):"),
                                dcc.Input(
                                    id='percentage-1-input',
                                    type='number',
                                    min=1,
                                    max=100,
                                    value=100,
                                    className="form-control"
                                )
                            ], width=6)
                        ], className="mt-2")
                    ], className="mb-3"),

                    # Seconda varietà (opzionale)
                    html.Div([
                        html.H6("Varietà 2 (opzionale)"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Seleziona Varietà:"),
                                dcc.Dropdown(
                                    id='variety-2-dropdown',
                                    options=[{'label': v, 'value': v} for v in olive_varieties['Varietà di Olive'].unique()],
                                    value=None
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Tecnica:"),
                                dcc.Dropdown(
                                    id='technique-2-dropdown',
                                    options=[
                                        {'label': 'Tradizionale', 'value': 'Tradizionale'},
                                        {'label': 'Intensiva', 'value': 'Intensiva'},
                                        {'label': 'Superintensiva', 'value': 'Superintensiva'}
                                    ],
                                    value=None,
                                    disabled=True
                                ),
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Percentuale (%):"),
                                dcc.Input(
                                    id='percentage-2-input',
                                    type='number',
                                    min=0,
                                    max=99,
                                    value=0,
                                    disabled=True,
                                    className="form-control"
                                )
                            ], width=6)
                        ], className="mt-2")
                    ], className="mb-3"),

                    # Terza varietà (opzionale)
                    html.Div([
                        html.H6("Varietà 3 (opzionale)"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Seleziona Varietà:"),
                                dcc.Dropdown(
                                    id='variety-3-dropdown',
                                    options=[{'label': v, 'value': v} for v in olive_varieties['Varietà di Olive'].unique()],
                                    value=None
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Tecnica:"),
                                dcc.Dropdown(
                                    id='technique-3-dropdown',
                                    options=[
                                        {'label': 'Tradizionale', 'value': 'Tradizionale'},
                                        {'label': 'Intensiva', 'value': 'Intensiva'},
                                        {'label': 'Superintensiva', 'value': 'Superintensiva'}
                                    ],
                                    value=None,
                                    disabled=True
                                ),
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Percentuale (%):"),
                                dcc.Input(
                                    id='percentage-3-input',
                                    type='number',
                                    min=0,
                                    max=99,
                                    value=0,
                                    disabled=True,
                                    className="form-control"
                                )
                            ], width=6)
                        ], className="mt-2")
                    ], className="mb-3"),

                    html.Div(id='percentage-warning', className="text-danger"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Ettari totali:"),
                            dcc.Input(
                                id='hectares-input',
                                type='number',
                                value=5,
                                min=1,
                                max=100,
                                className="form-control"
                            )
                        ], width=6)
                    ], className="mt-3")
                ])
            ], className="mb-4")
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Previsioni di Produzione"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Produzione Olive", className="text-center"),
                            html.H2(id='olive-production', className="text-center text-primary")
                        ], width=6),
                        dbc.Col([
                            html.H4("Produzione Olio", className="text-center"),
                            html.H2(id='oil-production', className="text-center text-success")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Fabbisogno Idrico", className="text-center mt-4"),
                            html.H2(id='water-need', className="text-center text-info")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='extra-info', className="text-center mt-4")
                        ])
                    ])
                ])
            ], className="mb-4")
        ], width=8)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Dettagli Produzione"),
                dbc.CardBody([
                    dcc.Graph(id='production-details')
                ])
            ])
        ], width=12, className="mb-4")
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analisi Meteorologica"),
                dbc.CardBody([
                    dcc.Graph(id='weather-impact')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Fabbisogno Idrico Mensile"),
                dbc.CardBody([
                    dcc.Graph(id='water-needs')
                ])
            ])
        ], width=6)
    ])
], fluid=True)

def prepare_static_features_multiple(varieties_info, percentages, hectares):
    """Prepara le feature statiche per multiple varietà."""
    static_features = [hectares]  # Inizia con gli ettari totali

    # Dizionario per tenere traccia delle feature per ogni varietà possibile
    all_varieties = olive_varieties['Varietà di Olive'].unique()
    variety_features = {}

    # Inizializza tutte le feature a 0 per tutte le varietà possibili
    for variety in all_varieties:
        variety_name = variety.lower().replace(" ", "_").replace("'", "")
        # Feature base
        variety_features[f'{variety_name}_pct'] = 0
        variety_features[f'{variety_name}_prod_t_ha'] = 0
        variety_features[f'{variety_name}_oil_prod_t_ha'] = 0
        variety_features[f'{variety_name}_oil_prod_l_ha'] = 0
        variety_features[f'{variety_name}_min_yield_pct'] = 0
        variety_features[f'{variety_name}_max_yield_pct'] = 0
        variety_features[f'{variety_name}_min_oil_prod_l_ha'] = 0
        variety_features[f'{variety_name}_max_oil_prod_l_ha'] = 0
        variety_features[f'{variety_name}_avg_oil_prod_l_ha'] = 0
        variety_features[f'{variety_name}_l_per_t'] = 0
        variety_features[f'{variety_name}_min_l_per_t'] = 0
        variety_features[f'{variety_name}_max_l_per_t'] = 0
        variety_features[f'{variety_name}_avg_l_per_t'] = 0
        # Feature tecniche
        variety_features[f'{variety_name}_tradizionale'] = 0
        variety_features[f'{variety_name}_intensiva'] = 0
        variety_features[f'{variety_name}_superintensiva'] = 0

    # Aggiorna le feature per le varietà selezionate
    for variety_info, percentage in zip(varieties_info, percentages):
        if variety_info is not None and percentage > 0:
            variety_name = variety_info['Varietà di Olive'].lower().replace(" ", "_").replace("'", "")
            technique = variety_info['Tecnica di Coltivazione'].lower()

            # Aggiorna le feature della varietà
            variety_features[f'{variety_name}_pct'] = percentage / 100
            variety_features[f'{variety_name}_prod_t_ha'] = variety_info['Produzione (tonnellate/ettaro)']
            variety_features[f'{variety_name}_oil_prod_t_ha'] = variety_info['Produzione Olio (tonnellate/ettaro)']
            variety_features[f'{variety_name}_oil_prod_l_ha'] = variety_info['Produzione Olio (litri/ettaro)']
            variety_features[f'{variety_name}_min_yield_pct'] = variety_info['Min % Resa']
            variety_features[f'{variety_name}_max_yield_pct'] = variety_info['Max % Resa']
            variety_features[f'{variety_name}_min_oil_prod_l_ha'] = variety_info['Min Produzione Olio (litri/ettaro)']
            variety_features[f'{variety_name}_max_oil_prod_l_ha'] = variety_info['Max Produzione Olio (litri/ettaro)']
            variety_features[f'{variety_name}_avg_oil_prod_l_ha'] = variety_info['Media Produzione Olio (litri/ettaro)']
            variety_features[f'{variety_name}_l_per_t'] = variety_info['Litri per Tonnellata']
            variety_features[f'{variety_name}_min_l_per_t'] = variety_info['Min Litri per Tonnellata']
            variety_features[f'{variety_name}_max_l_per_t'] = variety_info['Max Litri per Tonnellata']
            variety_features[f'{variety_name}_avg_l_per_t'] = variety_info['Media Litri per Tonnellata']

            # Aggiorna la tecnica
            variety_features[f'{variety_name}_{technique}'] = 1

    # Converti il dizionario in lista mantenendo l'ordine
    static_features.extend([variety_features[key] for key in sorted(variety_features.keys())])

    return np.array(static_features)

# Callback per la gestione delle percentuali e abilitazione dei campi
@app.callback(
    [Output('technique-2-dropdown', 'disabled'),
     Output('percentage-2-input', 'disabled'),
     Output('technique-3-dropdown', 'disabled'),
     Output('percentage-3-input', 'disabled'),
     Output('percentage-warning', 'children')],
    [Input('variety-2-dropdown', 'value'),
     Input('variety-3-dropdown', 'value'),
     Input('percentage-1-input', 'value'),
     Input('percentage-2-input', 'value'),
     Input('percentage-3-input', 'value')]
)
def manage_percentages(variety2, variety3, perc1, perc2, perc3):
    perc1 = perc1 or 0
    perc2 = perc2 or 0
    perc3 = perc3 or 0
    total = perc1 + perc2 + perc3

    # Abilita/disabilita campi basati sulle selezioni
    disable_2 = variety2 is None
    disable_3 = variety3 is None or variety2 is None

    warning = ""
    if total > 100:
        warning = "La somma delle percentuali non può superare 100%"
    elif total < 100:
        warning = f"La somma delle percentuali è {total}% (dovrebbe essere 100%)"

    return disable_2, disable_2, disable_3, disable_3, warning

# Aggiorna il callback principale per utilizzare multiple varietà
@app.callback(
    [Output('olive-production', 'children'),
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
    # Verifica i dati di input
    if not variety1 or not tech1 or perc1 is None or hectares is None:
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
        # Prepara i dati e fai la predizione
        prediction = make_prediction(weather_data, varieties_info, percentages, hectares)

        # Formatta output
        olive_prod_text = f"{prediction['olive_production']:.0f} kg/ha"
        oil_prod_text = f"{prediction['avg_oil_production']:.0f} L/ha"
        water_need_text = f"{prediction['water_need']:.0f} m³/ha"

        # Crea il grafico dei dettagli di produzione
        details_data = []

        # Aggiungi dati per ogni varietà
        for detail in prediction['variety_details']:
            details_data.extend([
                {
                    'Varietà': f"{detail['variety']} ({detail['percentage']}%)",
                    'Tipo': 'Olive',
                    'Produzione': detail['production_per_ha'] * (detail['percentage']/100)
                },
                {
                    'Varietà': f"{detail['variety']} ({detail['percentage']}%)",
                    'Tipo': 'Olio',
                    'Produzione': detail['oil_per_ha'] * (detail['percentage']/100)
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

        # Crea il grafico dei dettagli
        details_df = pd.DataFrame(details_data)
        details_fig = px.bar(
            details_df,
            x='Varietà',
            y='Produzione',
            color='Tipo',
            barmode='group',
            title='Dettagli Produzione per Varietà',
            labels={'Produzione': 'kg/ha o L/ha'},
            color_discrete_map={'Olive': '#1f77b4', 'Olio': '#2ca02c'}
        )
        details_fig.update_layout(
            legend_title_text='Prodotto',
            xaxis_tickangle=-45
        )

        # Grafico impatto meteo
        recent_weather = weather_data.tail(41).copy()
        weather_impact = px.scatter(
            recent_weather,
            x='temp',
            y='solarradiation',
            size='precip',
            title='Condizioni Meteorologiche',
            labels={
                'temp': 'Temperatura (°C)',
                'solarradiation': 'Radiazione Solare (W/m²)',
                'precip': 'Precipitazioni (mm)'
            }
        )
        weather_impact.update_layout(
            legend_title_text='Precipitazioni',
            showlegend=True
        )

        # Grafico fabbisogno idrico
        water_data = []
        months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
                  'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']

        # Calcola il fabbisogno idrico mensile per ogni varietà
        for detail in prediction['variety_details']:
            variety_info = olive_varieties[
                olive_varieties['Varietà di Olive'] == detail['variety']
                ].iloc[0]

            seasonal_water = {
                'Inverno': variety_info['Fabbisogno Acqua Inverno (m³/ettaro)'],
                'Primavera': variety_info['Fabbisogno Acqua Primavera (m³/ettaro)'],
                'Estate': variety_info['Fabbisogno Acqua Estate (m³/ettaro)'],
                'Autunno': variety_info['Fabbisogno Acqua Autunno (m³/ettaro)']
            }

            for month in months:
                season = get_season_from_month(month)
                water_data.append({
                    'Mese': month,
                    'Varietà': detail['variety'],
                    'Fabbisogno': seasonal_water[season] * (detail['percentage']/100)
                })

        # Crea il grafico del fabbisogno idrico
        water_df = pd.DataFrame(water_data)
        water_needs = px.bar(
            water_df,
            x='Mese',
            y='Fabbisogno',
            color='Varietà',
            title='Fabbisogno Idrico Mensile per Varietà',
            labels={'Fabbisogno': 'm³/ettaro'},
            barmode='stack'
        )
        water_needs.update_layout(
            legend_title_text='Varietà',
            xaxis_tickangle=0
        )

        extra_info = html.Div([
            html.H5("Dettagli per Varietà", className="mb-3"),
            html.Div([
                # Crea una card per ogni varietà
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                f"{detail['variety']} - {detail['percentage']}%",
                                className="font-weight-bold"
                            ),
                            dbc.CardBody([
                                # Trova i dettagli completi della varietà dal dataset originale
                                html.Div([
                                    # Produzione
                                    html.Div([
                                        html.H6("Produzione Prevista:", className="mb-2"),
                                        html.P([
                                            html.Span("Olive: ", className="font-weight-bold"),
                                            f"{detail['production_per_ha'] * (detail['percentage']/100):.0f} kg/ha"
                                        ]),
                                        html.P([
                                            html.Span("Olio: ", className="font-weight-bold"),
                                            f"{detail['oil_per_ha'] * (detail['percentage']/100):.0f} L/ha"
                                        ]),
                                    ], className="mb-3"),

                                    # Rese
                                    html.Div([
                                        html.H6("Rese:", className="mb-2"),
                                        html.P([
                                            html.Span("Resa in Olio: ", className="font-weight-bold"),
                                            f"{variety_info['Min % Resa']:.1f}% - {variety_info['Max % Resa']:.1f}%"
                                        ]),
                                        html.P([
                                            html.Span("Litri per Tonnellata: ", className="font-weight-bold"),
                                            f"{variety_info['Min Litri per Tonnellata']:.0f} - {variety_info['Max Litri per Tonnellata']:.0f} L/t"
                                        ])
                                    ], className="mb-3"),

                                    # Caratteristiche
                                    html.Div([
                                        html.H6("Caratteristiche:", className="mb-2"),
                                        html.P([
                                            html.Span("Temperatura Ottimale: ", className="font-weight-bold"),
                                            f"{variety_info['Temperatura Ottimale']}°C"
                                        ]),
                                        html.P([
                                            html.Span("Resistenza alla Siccità: ", className="font-weight-bold"),
                                            f"{variety_info['Resistenza']}"
                                        ])
                                    ], className="mb-3"),

                                    # Fabbisogno Idrico
                                    html.Div([
                                        html.H6("Fabbisogno Idrico Stagionale:", className="mb-2"),
                                        html.P([
                                            html.Span("Primavera: ", className="font-weight-bold"),
                                            f"{variety_info['Fabbisogno Acqua Primavera (m³/ettaro)']:.0f} m³/ha"
                                        ]),
                                        html.P([
                                            html.Span("Estate: ", className="font-weight-bold"),
                                            f"{variety_info['Fabbisogno Acqua Estate (m³/ettaro)']:.0f} m³/ha"
                                        ]),
                                        html.P([
                                            html.Span("Autunno: ", className="font-weight-bold"),
                                            f"{variety_info['Fabbisogno Acqua Autunno (m³/ettaro)']:.0f} m³/ha"
                                        ]),
                                        html.P([
                                            html.Span("Inverno: ", className="font-weight-bold"),
                                            f"{variety_info['Fabbisogno Acqua Inverno (m³/ettaro)']:.0f} m³/ha"
                                        ])
                                    ])
                                ])
                            ])
                        ], className="h-100")
                    ], width=12 if len(prediction['variety_details']) == 1 else
                    6 if len(prediction['variety_details']) == 2 else 4,
                        className="mb-3")
                    for detail in prediction['variety_details']
                    for variety_info in [olive_varieties[
                                             olive_varieties['Varietà di Olive'] == detail['variety']
                                             ].iloc[0]]
                ], className="mb-4"),

                # Sezione totali
                dbc.Card([
                    dbc.CardHeader("Totali Previsti", className="font-weight-bold"),
                    dbc.CardBody([
                        html.Div([
                            html.P([
                                html.Span("Produzione Totale Olive: ", className="font-weight-bold"),
                                f"{prediction['olive_production']:.0f} kg/ha"
                            ]),
                            html.P([
                                html.Span("Produzione Totale Olio: ", className="font-weight-bold"),
                                f"{prediction['avg_oil_production']:.0f} L/ha"
                            ]),
                            html.P([
                                html.Span("Resa Media in Olio: ", className="font-weight-bold"),
                                f"{(prediction['avg_oil_production']/prediction['olive_production']*100):.1f}%"
                            ]),
                            html.P([
                                html.Span("Fabbisogno Idrico Totale: ", className="font-weight-bold"),
                                f"{prediction['water_need']:.0f} m³/ha"
                            ])
                        ])
                    ])
                ])
            ])
        ], className="mt-4")

        return olive_prod_text, oil_prod_text, water_need_text, details_fig, weather_impact, water_needs, extra_info

    except Exception as e:
        print(f"Errore durante la predizione: {str(e)}")
        return "Errore", "Errore", "Errore", {}, {}, {}, f"Errore: {str(e)}"


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