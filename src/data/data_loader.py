import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from src.models.solar_models import create_uv_model, create_energy_model, create_radiation_model
from typing import Tuple, Optional
import datetime

def read_json_files(folder_path):
    all_data = []

    file_list = sorted(os.listdir(folder_path))

    for filename in file_list:
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    all_data.extend(data['days'])
            except Exception as e:
                print(f"Error processing file '{filename}': {str(e)}")

    return all_data


def save_single_model_and_scalers(model, model_name, scalers=None, base_path='./kaggle/working/models'):
    """
    Salva un singolo modello con tutti i suoi artefatti associati e multipli scaler.

    Parameters:
    -----------
    model : keras.Model
        Il modello da salvare
    model_name : str
        Nome del modello (es. 'solarradiation', 'solarenergy', 'uvindex')
    scalers : dict, optional
        Dizionario degli scaler associati al modello (es. {'X': x_scaler, 'y': y_scaler})
    base_path : str
        Percorso base dove salvare il modello
    """
    if isinstance(base_path, list):
        base_path = './kaggle/working/models'

    # Crea la cartella base se non esiste
    os.makedirs(base_path, exist_ok=True)

    # Crea la sottocartella per il modello specifico
    model_path = os.path.join(base_path, model_name)
    os.makedirs(model_path, exist_ok=True)

    try:
        print(f"\nSalvataggio modello {model_name}...")

        # 1. Salva il modello completo
        model_file = os.path.join(model_path, 'model.keras')
        model.save(model_file, save_format='keras')
        print(f"- Salvato modello completo: {model_file}")

        # 2. Salva i pesi separatamente
        weights_path = os.path.join(model_path, 'weights')
        os.makedirs(weights_path, exist_ok=True)
        weight_file = os.path.join(weights_path, 'weights')
        model.save_weights(weight_file)
        print(f"- Salvati pesi: {weight_file}")

        # 3. Salva il plot del modello
        plot_path = os.path.join(model_path, f'{model_name}_architecture.png')
        tf.keras.utils.plot_model(
            model,
            to_file=plot_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=150
        )
        print(f"- Salvato plot architettura: {plot_path}")

        # 4. Salva il summary del modello
        summary_path = os.path.join(model_path, f'{model_name}_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"- Salvato summary modello: {summary_path}")

        # 5. Salva gli scaler se forniti
        if scalers is not None:
            scaler_path = os.path.join(model_path, 'scalers')
            os.makedirs(scaler_path, exist_ok=True)

            for scaler_name, scaler in scalers.items():
                scaler_file = os.path.join(scaler_path, f'{scaler_name}_scaler.joblib')
                joblib.dump(scaler, scaler_file)
                print(f"- Salvato scaler {scaler_name}: {scaler_file}")

        # 6. Salva la configurazione del modello
        model_config = {
            'has_solar_params': True if model_name == 'solarradiation' else False,
            'scalers': list(scalers.keys()) if scalers else []
        }
        config_path = os.path.join(model_path, 'model_config.joblib')
        joblib.dump(model_config, config_path)
        print(f"- Salvata configurazione: {config_path}")

        # 7. Crea un README specifico per il modello
        readme_path = os.path.join(model_path, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(f"{model_name.upper()} Model Artifacts\n")
            f.write("=" * (len(model_name) + 15) + "\n\n")
            f.write("Directory structure:\n")
            f.write("- model.keras: Complete model\n")
            f.write("- weights/: Model weights\n")
            f.write(f"- {model_name}_architecture.png: Visual representation of model architecture\n")
            f.write(f"- {model_name}_summary.txt: Detailed model summary\n")
            f.write("- model_config.joblib: Model configuration\n")
            if scalers:
                f.write("- scalers/: Directory containing model scalers\n")
                for scaler_name in scalers.keys():
                    f.write(f"  - {scaler_name}_scaler.joblib: {scaler_name} scaler\n")

        print(f"\nTutti gli artefatti per {model_name} salvati in: {model_path}")
        print(f"Consulta {readme_path} per i dettagli sulla struttura")

    except Exception as e:
        print(f"Errore nel salvataggio degli artefatti per {model_name}: {str(e)}")
        raise

    return model_path


def load_single_model_and_scalers(model_name, base_path='./kaggle/working/models'):
    """
    Carica un singolo modello con tutti i suoi artefatti e scaler associati.

    Parameters:
    -----------
    model_name : str
        Nome del modello da caricare (es. 'solarradiation', 'solarenergy', 'uvindex')
    base_path : str
        Percorso base dove sono salvati i modelli

    Returns:
    --------
    tuple
        (model, scalers, model_config)
    """
    model_path = os.path.join(base_path, model_name)

    if not os.path.exists(model_path):
        print(f"Directory del modello non trovata: {model_path}")
        return None, None, None

    try:
        print(f"\nCaricamento modello {model_name}...")

        # 1. Carica la configurazione del modello
        config_path = os.path.join(model_path, 'model_config.joblib')
        try:
            model_config = joblib.load(config_path)
            print("- Configurazione modello caricata")
        except:
            print("! Configurazione modello non trovata, usando configurazione di default")
            model_config = {
                'has_solar_params': True if model_name == 'solarradiation' else False,
                'scalers': ['X', 'y']
            }

        # 2. Carica il modello
        try:
            # Prima prova a caricare il modello completo
            model_file = os.path.join(model_path, 'model.keras')
            model = tf.keras.models.load_model(model_file)
            print(f"- Modello caricato da: {model_file}")

            # Verifica i pesi
            weights_path = os.path.join(model_path, 'weights', 'weights')
            if os.path.exists(weights_path + '.index'):
                model.load_weights(weights_path)
                print("- Pesi verificati con successo")

        except Exception as e:
            print(f"! Errore nel caricamento del modello: {str(e)}")
            print("Tentativo di ricostruzione del modello...")

            try:
                # Ricostruzione del modello
                if model_name == 'solarradiation':
                    model = create_radiation_model(input_shape=(24, 8))
                elif model_name == 'solarenergy':
                    model = create_energy_model(input_shape=(24, 8))
                elif model_name == 'uvindex':
                    model = create_uv_model(input_shape=(24, 8))
                else:
                    raise ValueError(f"Tipo di modello non riconosciuto: {model_name}")

                # Carica i pesi
                model.load_weights(weights_path)
                print("- Modello ricostruito dai pesi con successo")
            except Exception as e:
                print(f"! Errore nella ricostruzione del modello: {str(e)}")
                return None, None, None

        # 3. Carica gli scaler
        scalers = {}
        scaler_path = os.path.join(model_path, 'scalers')
        if os.path.exists(scaler_path):
            print("\nCaricamento scaler:")
            for scaler_file in os.listdir(scaler_path):
                if scaler_file.endswith('_scaler.joblib'):
                    scaler_name = scaler_file.replace('_scaler.joblib', '')
                    scaler_file_path = os.path.join(scaler_path, scaler_file)
                    try:
                        scalers[scaler_name] = joblib.load(scaler_file_path)
                        print(f"- Caricato scaler {scaler_name}")
                    except Exception as e:
                        print(f"! Errore nel caricamento dello scaler {scaler_name}: {str(e)}")
        else:
            print("! Directory degli scaler non trovata")

        # 4. Verifica integrità del modello
        try:
            # Verifica che il modello possa fare predizioni
            if model_name == 'solarradiation':
                dummy_input = [np.zeros((1, 24, 8)), np.zeros((1, 3))]
            else:
                dummy_input = np.zeros((1, 24, 8))

            model.predict(dummy_input, verbose=0)
            print("\n✓ Verifica integrità modello completata con successo")
        except Exception as e:
            print(f"\n! Attenzione: il modello potrebbe non funzionare correttamente: {str(e)}")

        # 5. Carica e verifica il summary del modello
        summary_path = os.path.join(model_path, f'{model_name}_summary.txt')
        if os.path.exists(summary_path):
            print("\nSummary del modello disponibile in:", summary_path)

        # 6. Verifica il plot dell'architettura
        plot_path = os.path.join(model_path, f'{model_name}_architecture.png')
        if os.path.exists(plot_path):
            print("Plot dell'architettura disponibile in:", plot_path)

        print(f"\nCaricamento di {model_name} completato con successo!")
        return model, scalers, model_config

    except Exception as e:
        print(f"\nErrore critico nel caricamento del modello {model_name}: {str(e)}")
        return None, None, None



def load_weather_data(
        data_path: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Carica e preprocessa i dati meteorologici da file JSON o Parquet.

    Parameters
    ----------
    data_path : str
        Percorso al file dei dati (può essere .json o .parquet)
    start_year : int, optional
        Anno di inizio per filtrare i dati
    end_year : int, optional
        Anno di fine per filtrare i dati

    Returns
    -------
    pd.DataFrame
        DataFrame contenente i dati meteo preprocessati

    Examples
    --------
    >>> weather_data = load_weather_data('./data/weather_data.parquet', start_year=2010)
    """
    try:
        # Determina il tipo di file e carica di conseguenza
        if data_path.endswith('.parquet'):
            weather_data = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            # Se è un file JSON, prima lo convertiamo in DataFrame
            with open(data_path, 'r') as f:
                raw_data = json.load(f)
            weather_data = create_weather_dataset(raw_data)
        else:
            raise ValueError(f"Formato file non supportato: {data_path}")

        # Converti la colonna datetime
        weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], errors='coerce')

        # Filtra per anno se specificato
        if start_year is not None:
            weather_data = weather_data[weather_data['datetime'].dt.year >= start_year]
        if end_year is not None:
            weather_data = weather_data[weather_data['datetime'].dt.year <= end_year]

        # Aggiungi colonne di data
        weather_data['date'] = weather_data['datetime'].dt.date
        weather_data['year'] = weather_data['datetime'].dt.year
        weather_data['month'] = weather_data['datetime'].dt.month
        weather_data['day'] = weather_data['datetime'].dt.day

        # Rimuovi righe con datetime nullo
        weather_data = weather_data.dropna(subset=['datetime'])

        # Ordina per datetime
        weather_data = weather_data.sort_values('datetime')

        # Gestione valori mancanti nelle colonne principali
        numeric_columns = weather_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if weather_data[col].isnull().any():
                # Interpolazione lineare per i valori mancanti
                weather_data[col] = weather_data[col].interpolate(method='linear')

        # Rimuovi eventuali duplicati
        weather_data = weather_data.drop_duplicates(subset=['datetime'])

        # Verifica la completezza dei dati
        print(f"Dati caricati dal {weather_data['datetime'].min()} al {weather_data['datetime'].max()}")
        print(f"Numero totale di records: {len(weather_data)}")

        return weather_data

    except Exception as e:
        print(f"Errore nel caricamento dei dati meteo: {str(e)}")
        raise


def create_weather_dataset(raw_data: list) -> pd.DataFrame:
    """
    Converte i dati JSON grezzi in un DataFrame strutturato.

    Parameters
    ----------
    raw_data : list
        Lista di dizionari contenenti i dati meteo

    Returns
    -------
    pd.DataFrame
        DataFrame strutturato con i dati meteo
    """
    dataset = []
    seen_datetimes = set()

    for day in raw_data:
        date = day['datetime']
        for hour in day['hours']:
            datetime_str = f"{date} {hour['datetime']}"

            # Verifica duplicati
            if datetime_str in seen_datetimes:
                continue

            seen_datetimes.add(datetime_str)

            # Gestione preciptype
            if isinstance(hour['preciptype'], list):
                preciptype = "__".join(hour['preciptype'])
            else:
                preciptype = hour['preciptype'] if hour['preciptype'] else ""

            # Gestione conditions
            conditions = hour['conditions'].replace(', ', '__').replace(' ', '_').lower()

            # Crea la riga
            row = {
                'datetime': datetime_str,
                'temp': hour['temp'],
                'feelslike': hour['feelslike'],
                'humidity': hour['humidity'],
                'dew': hour['dew'],
                'precip': hour['precip'],
                'snow': hour['snow'],
                'preciptype': preciptype.lower(),
                'windspeed': hour['windspeed'],
                'winddir': hour['winddir'],
                'pressure': hour['pressure'],
                'cloudcover': hour['cloudcover'],
                'visibility': hour['visibility'],
                'solarradiation': hour['solarradiation'],
                'solarenergy': hour['solarenergy'],
                'uvindex': hour['uvindex'],
                'conditions': conditions,
                'tempmax': day['tempmax'],
                'tempmin': day['tempmin'],
                'precipprob': day['precipprob'],
                'precipcover': day['precipcover']
            }
            dataset.append(row)

    # Ordina per datetime
    dataset.sort(key=lambda x: datetime.strptime(x['datetime'], "%Y-%m-%d %H:%M:%S"))

    return pd.DataFrame(dataset)


def load_olive_varieties(
        data_path: str,
        add_water_features: bool = True
) -> pd.DataFrame:
    """
    Carica e preprocessa i dati delle varietà di olive.

    Parameters
    ----------
    data_path : str
        Percorso al file dei dati
    add_water_features : bool
        Se True, aggiunge feature relative al consumo d'acqua

    Returns
    -------
    pd.DataFrame
        DataFrame contenente i dati delle varietà di olive
    """
    try:
        if data_path.endswith('.csv'):
            olive_varieties = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            olive_varieties = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Formato file non supportato: {data_path}")

        # Se richiesto, aggiungi feature sul consumo d'acqua
        if add_water_features and 'Fabbisogno Acqua Primavera (m³/ettaro)' not in olive_varieties.columns:
            from src.data.data_simulator import add_olive_water_consumption_correlation
            olive_varieties = add_olive_water_consumption_correlation(olive_varieties)

        print(f"Dati varietà olive caricati: {len(olive_varieties)} varietà")

        return olive_varieties

    except Exception as e:
        print(f"Errore nel caricamento dei dati delle varietà: {str(e)}")
        raise