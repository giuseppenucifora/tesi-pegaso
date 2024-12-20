import json
import os
import hashlib
import jwt as pyjwt
from datetime import datetime, timedelta
import secrets

# Costanti
SECRET_KEY = 'M!3EmyJ@P$yqt$dYRQ#73QtxFy$aTn8M98P8i5T9x9Fd5LHMcHgdfEEt#?H9EPg&9Qhokh$#pTyYLHxL'  # In produzione, usare una variabile d'ambiente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE = os.path.join(BASE_DIR, "users", "users.json")
CONFIGS_DIR = os.path.join(BASE_DIR, "users", "configs")

def init_directory_structure():
    """
    Inizializza la struttura delle directory necessarie per l'applicazione
    """
    # Crea directory per utenti e configurazioni
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    # Se il file users.json non esiste, crealo vuoto
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)

    print(f"Initialized directory structure:")
    print(f"Users file: {USERS_FILE}")
    print(f"Configs directory: {CONFIGS_DIR}")

def hash_password(password):
    """Hash la password usando SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

# In auth/utils.py
def create_user(username, password):
    """
    Crea un nuovo utente
    Args:
        username: nome utente
        password: password in chiaro
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        print(f"Tentativo di creazione utente: {username}")

        # Inizializza la struttura delle directory
        init_directory_structure()

        # Inizializza o carica il dizionario degli utenti
        users = {}
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, 'r') as f:
                    content = f.read()
                    if content.strip():  # Verifica che il file non sia vuoto
                        users = json.loads(content)
                    print(f"Utenti esistenti: {len(users)}")
            except json.JSONDecodeError as e:
                print(f"Errore nel parsing del file utenti: {e}")
                pass

        # Validazioni
        if not username or len(username) < 3:
            return False, "Username deve essere almeno 3 caratteri"
        if not password or len(password) < 6:
            return False, "Password deve essere almeno 6 caratteri"
        if username in users:
            return False, "Username già esistente"

        # Genera il salt e hash della password
        salt = secrets.token_hex(8)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()

        # Ottieni il percorso della configurazione utente
        user_config = get_user_config_path(username)
        print(f"Percorso configurazione utente: {user_config}")

        # Crea la configurazione dell'utente
        os.makedirs(os.path.dirname(user_config), exist_ok=True)

        # Salva la configurazione di default
        default_config = get_default_config()
        with open(user_config, 'w') as f:
            json.dump(default_config, f, indent=4)
        print("Configurazione default salvata")

        # Aggiungi nuovo utente con struttura semplificata
        users[username] = {
            "salt": salt,
            "password_hash": password_hash
        }

        # Salva il file utenti
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        print("File utenti aggiornato con successo")

        return True, "Utente creato con successo"

    except Exception as e:
        print(f"Errore nella creazione dell'utente: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Errore nella creazione dell'utente: {str(e)}"

def verify_user(username, password):
    """
    Verifica le credenziali utente usando salt e hash
    Args:
        username: nome utente
        password: password in chiaro
    Returns:
        bool: True se le credenziali sono valide, False altrimenti
    """
    try:
        if not os.path.exists(USERS_FILE):
            print("File utenti non trovato")
            return False

        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        if username not in users:
            print("Username non trovato")
            return False

        # Ottieni il salt e l'hash salvati
        user_data = users[username]
        stored_salt = user_data['salt']
        stored_hash = user_data['password_hash']

        # Calcola l'hash della password fornita con il salt salvato
        password_hash = hashlib.sha256((password + stored_salt).encode()).hexdigest()

        # Confronta gli hash
        return stored_hash == password_hash

    except Exception as e:
        print(f"Errore nella verifica dell'utente: {str(e)}")
        return False

def create_token(username):
    """Crea JWT token"""
    expiration = datetime.utcnow() + timedelta(hours=24)
    return pyjwt.encode(
        {"user": username, "exp": expiration},
        SECRET_KEY,
        algorithm="HS256"
    )

def verify_token(token):
    """Verifica JWT token"""
    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return True, payload["user"]
    except pyjwt.ExpiredSignatureError:
        return False, "Token scaduto"
    except pyjwt.InvalidTokenError:
        return False, "Token non valido"

def get_user_config_path(username):
    """
    Restituisce il percorso del file di configurazione per l'utente specificato
    Args:
        username: nome utente
    Returns:
        str: percorso assoluto del file di configurazione
    """
    # Sostituisci caratteri non validi nel nome utente
    safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_'))

    # Costruisci il percorso assoluto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'config', 'users')
    config_path = os.path.join(config_dir, f"{safe_username}_config.json")

    # Assicurati che la directory esista
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    return config_path

def get_default_config():
    return {
        "oliveto": {
            "hectares": 1,
            "varieties": [
                {
                    "variety": "Nocellara dell'Etna",
                    "technique": "tradizionale",
                    "percentage": 50
                },
                {
                    "variety": "Frantoio",
                    "technique": "tradizionale",
                    "percentage": 10
                },
                {
                    "variety": "Coratina",
                    "technique": "tradizionale",
                    "percentage": 40
                }
            ]
        },
        "costs": {
            "fixed": {
                "ammortamento": 2000,
                "assicurazione": 500,
                "manutenzione": 800,
                "certificazioni": 3000
            },
            "variable": {
                "raccolta": 0.35,
                "potatura": 600,
                "fertilizzanti": 400,
                "irrigazione": 300
            },
            "transformation": {
                "molitura": 0.15,
                "stoccaggio": 0.2,
                "bottiglia": 1.2,
                "etichettatura": 0.3
            },
            "marketing": {
                "budget_annuale": 15000,
                "costi_commerciali": 0.5,
                "prezzo_vendita": 12,
                "perc_vendita_diretta": 30
            }
        },
        "inference": {
            "debug_mode": True,
            'model_path': './sources/olive_oil_transformer/olive_oil_transformer_model.keras'
        }
    }