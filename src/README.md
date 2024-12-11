# Dashboard Produzione Olio d'Oliva

Questa dashboard interattiva permette di simulare e analizzare la produzione di olio d'oliva basandosi su diversi parametri ambientali e varietali.

## Requisiti di Sistema

- Python 3.8 o superiore
- Git LFS (per i file di grandi dimensioni)
- Conda (consigliato) o pip

## Configurazione dell'Ambiente

### Utilizzando Conda (Consigliato)

1. Clona il repository:
2. Crea e attiva l'ambiente conda:
   ```bash
   conda env create -f src/environment.yml
   conda activate olive-dashboard
   ```

### Utilizzando pip

1. Clona il repository:
2. Crea e attiva un ambiente virtuale:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   # oppure
   .\venv\Scripts\activate  # Windows
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r src/requirements.txt
   ```

## Configurazione del Progetto

1. Crea un file `.env` nella directory `src` con le seguenti variabili:
   ```env
   DEV_MODE=True  # Imposta su False per utilizzare i modelli ML
   ```
   
## Esecuzione della Dashboard

1. Assicurati di essere nella directory `src`:
   ```bash
   cd src
   ```

2. Avvia la dashboard:
   ```bash
   python olive-oil-dashboard.py
   ```

3. Apri un browser e vai all'indirizzo:
   ```
   http://localhost:8050
   ```

## Struttura del Progetto

```
src/
├── dashboard/               # Componenti della dashboard
├── sources/                # Dati e modelli
├── utils/                  # Utility functions
├── olive-oil-dashboard.py  # Main application
├── olive_config.json       # Configuration file
└── requirements.txt        # Python dependencies
```

## Funzionalità Principali

La dashboard offre diverse funzionalità:

1. **Configurazione Oliveto**
    - Gestione delle varietà di olive
    - Configurazione delle tecniche di coltivazione
    - Impostazione delle percentuali di mix varietale

2. **Simulazione Ambientale**
    - Simulazione degli effetti delle condizioni meteorologiche
    - Analisi dell'impatto sulla produzione
    - Visualizzazione KPI

3. **Analisi Economica**
    - Gestione dei costi di produzione
    - Analisi dei margini
    - Proiezioni finanziarie

4. **Produzione**
    - Monitoraggio della produzione
    - Analisi del fabbisogno idrico
    - Dettagli per varietà

## Risoluzione Problemi Comuni

### ModuleNotFoundError

Se riscontri errori del tipo "ModuleNotFoundError", verifica di:

1. Aver attivato l'ambiente virtuale corretto
2. Aver installato tutte le dipendenze:
   ```bash
   pip install -r requirements.txt  # se usi pip
   # oppure
   conda env update -f environment.yml  # se usi conda
   ```

### Errori di Importazione dei Modelli

Se riscontri errori nell'importazione dei modelli:

1. Verifica che `DEV_MODE=True` nel file `.env`
2. Controlla che tutti i file necessari siano presenti nella directory `sources/`

## Supporto

Per problemi o domande, aprire una issue nel repository del progetto.
