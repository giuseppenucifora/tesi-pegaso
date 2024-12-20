# Olive Oil Production Training Dataset Generator

## Overview
This Python script generates a synthetic training dataset for olive oil production simulation. It simulates various factors affecting olive production including weather conditions, olive varieties, cultivation techniques, and geographical zones.

## Features
- Parallel processing for efficient dataset generation
- Batch processing to manage memory usage
- Configurable simulation parameters
- Weather effect calculations
- Multiple olive varieties and cultivation techniques support
- Zone-based production simulation

## Prerequisites
Required Python packages:
- pandas
- numpy
- psutil
- tqdm
- concurrent.futures (part of Python standard library)
- multiprocessing (part of Python standard library)

Required input data files:
- `./sources/weather_data_solarenergy.parquet`: Weather data including solar energy measurements
- `./sources/olive_varieties.parquet`: Olive varieties and their characteristics

## Usage

### Command Line Arguments
```bash
python olive_oil_train_dataset.create_train_dataset [options]
```

Options:
- `--random-seed`: Seed for reproducible results (optional)
- `--num-simulations`: Total number of simulations to run (default: 100000)
- `--num-zones`: Number of zones per simulation (default: same as num-simulations)
- `--batch-size`: Size of each simulation batch (default: 10000)
- `--output-path`: Output file path (default: './sources/olive_training_dataset.parquet')
- `--max-workers`: Number of parallel workers (default: automatically optimized)

### Example
```bash
python olive_oil_train_dataset.create_train_dataset --num-simulations 50 --num-zones 10 --batch-size 50 --output-path "./output/olive_dataset.parquet"
```

## Output
The script generates a Parquet file containing simulated olive production data with the following key features:
- Simulation and zone identifiers
- Weather conditions (temperature, precipitation, solar energy)
- Production metrics per olive variety
- Oil yield calculations
- Water requirements
- Cultivation techniques

## Technical Details

### Simulation Parameters
The simulation takes into account:
- Temperature effects on production
- Water availability and drought resistance
- Solar radiation impact
- Variety-specific characteristics
- Cultivation technique influence
- Zone-specific variations