import tensorflow as tf
from tf.keras.layers import Dense, LSTM, Conv1D, MultiHeadAttention, Dropout, BatchNormalization, LayerNormalization, GlobalAveragePooling1D, Concatenate, Input, Reshape, Activation
from tf.keras.models import Model
import tf.keras.backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tf.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.utils.class_weight import compute_class_weight


def get_season(date):
    month = date.month
    day = date.day
    if (month == 12 and day >= 21) or (month <= 3 and day < 20):
        return 'Winter'
    elif (month == 3 and day >= 20) or (month <= 6 and day < 21):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month <= 9 and day < 23):
        return 'Summer'
    elif (month == 9 and day >= 23) or (month <= 12 and day < 21):
        return 'Autumn'
    else:
        return 'Unknown'


def get_time_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


def add_time_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['timestamp'] = df['datetime'].astype(np.int64) // 10 ** 9
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['datetime'].dt.quarter
    df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = df['datetime'].dt.is_quarter_end.astype(int)
    df['is_year_end'] = df['datetime'].dt.is_year_end.astype(int)
    df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
    df['day_of_year_sin'] = np.sin(df['day_of_year'] * (2 * np.pi / 365.25))
    df['day_of_year_cos'] = np.cos(df['day_of_year'] * (2 * np.pi / 365.25))
    df['season'] = df['datetime'].apply(get_season)
    df['time_period'] = df['hour'].apply(get_time_period)
    return df


def add_solar_features(df):
    # Calcolo dell'angolo solare
    df['solar_angle'] = np.sin(df['day_of_year'] * (2 * np.pi / 365.25)) * np.sin(df['hour'] * (2 * np.pi / 24))

    # Interazioni tra features rilevanti
    df['cloud_temp_interaction'] = df['cloudcover'] * df['temp']
    df['visibility_cloud_interaction'] = df['visibility'] * (100 - df['cloudcover'])

    # Feature derivate
    df['clear_sky_index'] = (100 - df['cloudcover']) / 100
    df['temp_gradient'] = df['temp'] - df['tempmin']

    return df


def add_solar_specific_features(df):
    # Angolo solare e durata del giorno
    df['day_length'] = 12 + 3 * np.sin(2 * np.pi * (df['day_of_year'] - 81) / 365.25)
    df['solar_noon'] = 12 - df['hour']
    df['solar_elevation'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25) * np.cos(2 * np.pi * df['solar_noon'] / 24)

    # Interazioni
    df['cloud_elevation'] = df['cloudcover'] * df['solar_elevation']
    df['visibility_elevation'] = df['visibility'] * df['solar_elevation']

    # Rolling features con finestre più ampie
    df['cloud_rolling_12h'] = df['cloudcover'].rolling(window=12).mean()
    df['temp_rolling_12h'] = df['temp'].rolling(window=12).mean()

    return df


def add_advanced_features(df):
    # Features esistenti
    df = add_time_features(df)
    df = add_solar_features(df)
    df = add_solar_specific_features(df)

    # Aggiungi interazioni tra variabili meteorologiche
    df['temp_humidity'] = df['temp'] * df['humidity']
    df['temp_cloudcover'] = df['temp'] * df['cloudcover']
    df['visibility_cloudcover'] = df['visibility'] * df['cloudcover']

    # Features derivate per la radiazione solare
    df['clear_sky_factor'] = (100 - df['cloudcover']) / 100
    df['day_length'] = np.sin(df['day_of_year_sin']) * 12 + 12  # approssimazione della durata del giorno

    # Lag features
    df['temp_1h_lag'] = df['temp'].shift(1)
    df['cloudcover_1h_lag'] = df['cloudcover'].shift(1)
    df['humidity_1h_lag'] = df['humidity'].shift(1)

    # Rolling means
    df['temp_rolling_mean_6h'] = df['temp'].rolling(window=6).mean()
    df['cloudcover_rolling_mean_6h'] = df['cloudcover'].rolling(window=6).mean()

    return df

def prepare_advanced_data(df):
    # Applicazione delle funzioni di feature engineering
    df = add_advanced_features(df)

    # Selezione delle feature più rilevanti per UV index
    selected_features = [
        # Features meteorologiche base
        'temp', 'humidity', 'cloudcover', 'visibility', 'pressure',

        # Features temporali cicliche
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'day_of_year_sin', 'day_of_year_cos',

        # Features solari
        'solar_angle', 'solar_elevation', 'day_length',
        'clear_sky_index', 'solar_noon',

        # Interazioni
        'cloud_temp_interaction', 'visibility_cloud_interaction',
        'cloud_elevation', 'visibility_elevation',

        # Rolling features
        'cloud_rolling_12h', 'temp_rolling_12h',
        'temp_rolling_mean_6h', 'cloudcover_rolling_mean_6h',

        # Features categoriche (da encodare)
        'season', 'time_period'
    ]

    # One-hot encoding per le feature categoriche
    df = pd.get_dummies(df, columns=['season', 'time_period'])

    # Aggiorna la lista delle feature con le colonne one-hot
    categorical_columns = [col for col in df.columns if col.startswith(('season_', 'time_period_'))]
    final_features = [f for f in selected_features if f not in ['season', 'time_period']] + categorical_columns

    # Rimozione delle righe con valori NaN (create dai rolling features)
    df = df.dropna()

    X = df[final_features]
    y = df['uvindex']

    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling delle feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, final_features

def create_sequence_data(X, sequence_length=24):
    """
    Converte i dati in sequenze per l'input LSTM
    sequence_length rappresenta quante ore precedenti considerare
    """
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    return np.array(sequences)


def prepare_hybrid_data(df):
    # Utilizziamo la preparazione dati esistente
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = prepare_advanced_data(df)

    # Convertiamo i dati in sequenze
    sequence_length = 24  # 24 ore di dati storici

    X_train_seq = create_sequence_data(X_train_scaled, sequence_length)
    X_test_seq = create_sequence_data(X_test_scaled, sequence_length)

    # Adattiamo le y rimuovendo i primi (sequence_length-1) elementi
    y_train = y_train[sequence_length - 1:]
    y_test = y_test[sequence_length - 1:]

    return X_train_seq, X_test_seq, y_train, y_test, scaler, features


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Implementa un blocco Transformer Encoder
    """
    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed Forward Network
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)

    return LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

def custom_activation(x):
    """
    Activation function personalizzata che limita l'output tra 0 e 11
    """
    return 11 * tf.sigmoid(x)


def custom_loss(y_true, y_pred):
    """
    Loss function personalizzata che penalizza fortemente le predizioni fuori range
    """
    # MSE base
    mse = K.mean(K.square(y_true - y_pred))

    # Penalità per valori fuori range
    below_range = K.relu(0 - y_pred)
    above_range = K.relu(y_pred - 11)

    # Aggiungi una forte penalità per valori fuori range
    range_penalty = 10.0 * (K.mean(K.square(below_range)) + K.mean(K.square(above_range)))

    return mse + range_penalty


def create_hybrid_model(input_shape, n_features):
    """
    Crea un modello ibrido con output vincolato tra 0 e 11
    """
    # Input Layer
    inputs = Input(shape=input_shape)

    # CNN Branch - Estrazione pattern locali
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(conv1)
    conv3 = Conv1D(filters=256, kernel_size=7, activation='relu', padding='same')(conv2)
    conv_output = BatchNormalization()(conv3)

    # LSTM Branch - Dipendenze temporali
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm_output = BatchNormalization()(lstm2)

    # Combine CNN and LSTM branches
    combined = Concatenate()([conv_output, lstm_output])

    # Multi-Head Attention per catturare relazioni complesse
    attention_output = transformer_encoder(
        combined,
        head_size=32,
        num_heads=8,
        ff_dim=256,
        dropout=0.1
    )

    # Global Pooling
    pooled = GlobalAveragePooling1D()(attention_output)

    # Dense Layers con attivazioni vincolate
    dense1 = Dense(128)(pooled)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense2 = Dense(64)(dense1)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(0.2)(dense2)

    # Output layer con attivazione personalizzata per limitare tra 0 e 11
    outputs = Dense(1)(dense2)
    outputs = Activation(custom_activation)(outputs)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile con loss function personalizzata
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=['mae', 'mse']
    )

    return model


def evaluate_uv_predictions(y_true, y_pred):
    """
    Valutazione specifica per UV index con metriche categoriche
    """
    # Arrotonda le predizioni al più vicino intero
    y_pred_rounded = np.round(y_pred)

    # Clip dei valori tra 0 e 11
    y_pred_clipped = np.clip(y_pred_rounded, 0, 11)

    # Calcolo metriche
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    r2 = r2_score(y_true, y_pred_clipped)

    # Calcolo accuratezza per diversi margini di errore
    exact_accuracy = np.mean(y_pred_clipped == y_true)
    one_off_accuracy = np.mean(np.abs(y_pred_clipped - y_true) <= 1)
    two_off_accuracy = np.mean(np.abs(y_pred_clipped - y_true) <= 2)

    print("\nUV Index Prediction Metrics:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Exact Match Accuracy: {exact_accuracy:.3f}")
    print(f"±1 Accuracy: {one_off_accuracy:.3f}")
    print(f"±2 Accuracy: {two_off_accuracy:.3f}")

    # Confusion Matrix per livelli di UV
    def get_uv_level(value):
        if value <= 2:
            return 'Low'
        elif value <= 5:
            return 'Moderate'
        elif value <= 7:
            return 'High'
        elif value <= 10:
            return 'Very High'
        else:
            return 'Extreme'

    y_true_levels = [get_uv_level(v) for v in y_true]
    y_pred_levels = [get_uv_level(v) for v in y_pred_clipped]

    print("\nUV Level Confusion Matrix:")
    print(pd.crosstab(
        pd.Series(y_true_levels, name='Actual'),
        pd.Series(y_pred_levels, name='Predicted')
    ))

    return mae, rmse, r2, exact_accuracy, one_off_accuracy


def plot_uv_predictions(y_true, y_pred):
    """
    Visualizzazione delle predizioni specifica per UV index
    """
    plt.figure(figsize=(15, 5))

    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 11], [0, 11], 'r--', lw=2)
    plt.xlabel('Actual UV Index')
    plt.ylabel('Predicted UV Index')
    plt.title('Actual vs Predicted UV Index')
    plt.grid(True)

    # Plot 2: Distribution of Errors
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_hybrid_model(model, X_train, y_train, X_test, y_test, class_weights=None, epochs=100, batch_size=32):
    """
    Funzione di training avanzata per il modello ibrido UV index con monitoraggio dettagliato
    e gestione del training.

    Parameters:
    -----------
    model : keras.Model
        Il modello ibrido compilato
    X_train : numpy.ndarray
        Dati di training
    y_train : numpy.ndarray
        Target di training
    X_test : numpy.ndarray
        Dati di validation
    y_test : numpy.ndarray
        Target di validation
    class_weights : dict, optional
        Pesi per bilanciare le classi UV
    epochs : int, optional
        Numero massimo di epoche di training
    batch_size : int, optional
        Dimensione del batch

    Returns:
    --------
    history : keras.callbacks.History
        Storia del training con tutte le metriche
    """

    # Callbacks avanzati per il training
    callbacks = [
        # Early Stopping avanzato
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            mode='min',
            verbose=1,
            min_delta=1e-4
        ),

        # Learning Rate Schedule
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            mode='min',
            min_delta=1e-4,
            cooldown=5,
            min_lr=1e-6
        ),

        # Model Checkpoint per salvare i migliori modelli
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_uv_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),

        # TensorBoard callback per il monitoraggio
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

        # Custom Callback per monitorare le predizioni fuori range
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\nEpoch {epoch + 1}: Predizioni fuori range: "
                f"{np.sum((model.predict(X_test) < 0) | (model.predict(X_test) > 11))}"
            ) if epoch % 10 == 0 else None
        )
    ]

    # Calcolo dei class weights se non forniti
    if class_weights is None:
        # Discretizziamo i valori UV per il calcolo dei pesi
        y_discrete = np.round(y_train).astype(int)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_discrete),
            y=y_discrete
        )
        class_weights = dict(enumerate(class_weights))

    # Training con gestione degli errori e logging
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            shuffle=True,
            workers=4,
            use_multiprocessing=True
        )

        # Analisi post-training
        print("\nTraining completato con successo!")

        # Valutazione finale sul test set
        test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nMetriche finali sul test set:")
        print(f"Loss: {test_loss:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"MSE: {test_mse:.4f}")

        # Analisi delle predizioni
        predictions = model.predict(X_test)
        out_of_range = np.sum((predictions < 0) | (predictions > 11))
        print(f"\nPredizioni fuori range: {out_of_range} ({out_of_range / len(predictions) * 100:.2f}%)")

        # Plot della loss durante il training
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Salvataggio dei risultati del training
        training_results = {
            'final_loss': test_loss,
            'final_mae': test_mae,
            'final_mse': test_mse,
            'out_of_range_predictions': out_of_range,
            'training_time': len(history.history['loss']),
            'best_epoch': np.argmin(history.history['val_loss']) + 1
        }

        # Salvataggio su file
        with open('training_results.json', 'w') as f:
            json.dump(training_results, f, indent=4)

        return history

    except Exception as e:
        print(f"\nErrore durante il training: {str(e)}")
        raise

    finally:
        # Pulizia della memoria
        tf.keras.backend.clear_session()


def train_uvindex_bounded_model(df):
    """
    Training completo del modello UV index con preparazione dati, training,
    valutazione e visualizzazione dei risultati.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenente i dati meteorologici e UV index

    Returns:
    --------
    tuple:
        - model: modello addestrato
        - scaler: scaler utilizzato per la normalizzazione
        - features: lista delle feature utilizzate
        - history: storia del training
        - predictions: predizioni sul test set
        - y_test: valori reali del test set
        - metrics: metriche di valutazione
        - training_results: dizionario con i risultati dettagliati del training
    """
    print("Inizializzazione del training del modello UV index...")

    try:
        # Preparazione dei dati
        print("\n1. Preparazione dei dati...")
        X_train_seq, X_test_seq, y_train, y_test, scaler, features = prepare_hybrid_data(df)

        print(f"Shape dei dati di training: {X_train_seq.shape}")
        print(f"Shape dei dati di test: {X_test_seq.shape}")
        print(f"Numero di feature utilizzate: {len(features)}")

        # Verifica della qualità dei dati
        if np.isnan(X_train_seq).any() or np.isnan(y_train).any():
            raise ValueError("Trovati valori NaN nei dati di training")

        # Verifica del range dei valori UV
        if not (0 <= y_train.max() <= 11 and 0 <= y_test.max() <= 11):
            print("WARNING: Trovati valori UV index fuori range (0-11)")

        # Creazione del modello
        print("\n2. Creazione del modello...")
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = create_hybrid_model(input_shape, len(features))
        model.summary()

        # Calcolo class weights per bilanciare il dataset
        y_discrete = np.round(y_train).astype(int)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_discrete),
            y=y_discrete
        )
        class_weights_dict = dict(enumerate(class_weights))

        print("\n3. Avvio del training...")
        history = train_hybrid_model(
            model=model,
            X_train=X_train_seq,
            y_train=y_train,
            X_test=X_test_seq,
            y_test=y_test,
            class_weights=class_weights_dict,
            epochs=100,
            batch_size=32
        )

        print("\n4. Generazione delle predizioni...")
        predictions = model.predict(X_test_seq)

        # Clip delle predizioni nel range corretto
        predictions = np.clip(predictions, 0, 11)

        print("\n5. Valutazione del modello...")
        metrics = evaluate_uv_predictions(y_test, predictions)

        print("\n6. Creazione delle visualizzazioni...")
        plot_uv_predictions(y_test, predictions)

        # Creazione del dizionario dei risultati
        training_results = {
            'model_params': {
                'input_shape': input_shape,
                'n_features': len(features),
                'sequence_length': X_train_seq.shape[1]
            },
            'training_params': {
                'batch_size': 32,
                'total_epochs': len(history.history['loss']),
                'best_epoch': np.argmin(history.history['val_loss']) + 1
            },
            'performance_metrics': {
                'final_loss': float(history.history['val_loss'][-1]),
                'final_mae': float(history.history['val_mae'][-1]),
                'best_val_loss': float(min(history.history['val_loss'])),
                'out_of_range_predictions': int(np.sum((predictions < 0) | (predictions > 11))),
                'accuracy_metrics': metrics
            },
            'feature_importance': {
                feature: float(importance)
                for feature, importance in zip(features, model.layers[0].get_weights()[0].mean(axis=1))
            }
        }

        # Salvataggio dei risultati
        print("\n7. Salvataggio dei risultati...")

        # Salva il modello
        model.save('uv_index_model.h5')

        # Salva i risultati del training
        with open('training_results.json', 'w') as f:
            json.dump(training_results, f, indent=4)

        # Salva lo scaler
        joblib.dump(scaler, 'scaler.pkl')

        print("\nTraining completato con successo!")

        return (
            model, scaler, features, history,
            predictions, y_test, metrics, training_results
        )

    except Exception as e:
        print(f"\nErrore durante il training: {str(e)}")
        raise

    finally:
        # Pulizia della memoria
        tf.keras.backend.clear_session()

df = pd.read_parquet('../data/weather_data.parquet')

# Esegui il training
(model, scaler, features, history,
 predictions, y_test, metrics,
 training_results) = train_uvindex_bounded_model(df)
