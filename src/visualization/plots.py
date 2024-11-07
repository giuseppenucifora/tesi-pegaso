import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Optional


def save_plot(plt: plt, title: str, output_dir: str = './kaggle/working/plots'):
    """
    Salva il plot corrente con un nome formattato.

    Parameters
    ----------
    plt : matplotlib.pyplot
        Riferimento a pyplot
    title : str
        Titolo del plot
    output_dir : str
        Directory di output per i plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pulisci il nome del file
    filename = "".join(x for x in title if x.isalnum() or x in [' ', '-', '_']).rstrip()
    filename = filename.replace(' ', '_').lower()

    filepath = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot salvato come: {filepath}")


def plot_variety_comparison(comparison_data: pd.DataFrame, metric: str):
    """
    Crea un grafico a barre per confrontare le varietà di olive su una metrica specifica.

    Parameters
    ----------
    comparison_data : pd.DataFrame
        DataFrame contenente i dati di confronto
    metric : str
        Nome della metrica da visualizzare
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(comparison_data['Variety'], comparison_data[metric])
    plt.title(f'Confronto di {metric} tra Varietà di Olive')
    plt.xlabel('Varietà')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')

    # Aggiungi etichette sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Salva il plot
    save_plot(plt,
              f'variety_comparison_{metric.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")}')
    plt.close()


def plot_efficiency_vs_production(comparison_data: pd.DataFrame):
    """
    Crea uno scatter plot dell'efficienza vs produzione.

    Parameters
    ----------
    comparison_data : pd.DataFrame
        DataFrame contenente i dati di confronto
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(comparison_data['Avg Olive Production (kg/ha)'],
                comparison_data['Oil Efficiency (L/kg)'],
                s=100)

    # Aggiungi etichette per ogni punto
    for i, row in comparison_data.iterrows():
        plt.annotate(row['Variety'],
                     (row['Avg Olive Production (kg/ha)'], row['Oil Efficiency (L/kg)']),
                     xytext=(5, 5), textcoords='offset points')

    plt.title('Efficienza Olio vs Produzione Olive per Varietà')
    plt.xlabel('Produzione Media Olive (kg/ha)')
    plt.ylabel('Efficienza Olio (L olio / kg olive)')
    plt.tight_layout()

    # Salva il plot
    save_plot(plt, 'efficiency_vs_production')
    plt.close()


def plot_water_efficiency_vs_production(comparison_data: pd.DataFrame):
    """
    Crea uno scatter plot dell'efficienza idrica vs produzione.

    Parameters
    ----------
    comparison_data : pd.DataFrame
        DataFrame contenente i dati di confronto
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(comparison_data['Avg Olive Production (kg/ha)'],
                comparison_data['Water Efficiency (L oil/m³ water)'],
                s=100)

    # Aggiungi etichette per ogni punto
    for i, row in comparison_data.iterrows():
        plt.annotate(row['Variety'],
                     (row['Avg Olive Production (kg/ha)'],
                      row['Water Efficiency (L oil/m³ water)']),
                     xytext=(5, 5), textcoords='offset points')

    plt.title('Efficienza Idrica vs Produzione Olive per Varietà')
    plt.xlabel('Produzione Media Olive (kg/ha)')
    plt.ylabel('Efficienza Idrica (L olio / m³ acqua)')
    plt.tight_layout()
    plt.show()

    # Salva il plot
    save_plot(plt, 'water_efficiency_vs_production')
    plt.close()


def plot_water_need_vs_oil_production(comparison_data: pd.DataFrame):
    """
    Crea uno scatter plot del fabbisogno idrico vs produzione di olio.

    Parameters
    ----------
    comparison_data : pd.DataFrame
        DataFrame contenente i dati di confronto
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(comparison_data['Avg Water Need (m³/ha)'],
                comparison_data['Avg Oil Production (L/ha)'],
                s=100)

    # Aggiungi etichette per ogni punto
    for i, row in comparison_data.iterrows():
        plt.annotate(row['Variety'],
                     (row['Avg Water Need (m³/ha)'],
                      row['Avg Oil Production (L/ha)']),
                     xytext=(5, 5), textcoords='offset points')

    plt.title('Produzione Olio vs Fabbisogno Idrico per Varietà')
    plt.xlabel('Fabbisogno Idrico Medio (m³/ha)')
    plt.ylabel('Produzione Media Olio (L/ha)')
    plt.tight_layout()
    plt.show()

    # Salva il plot
    save_plot(plt, 'water_need_vs_oil_production')
    plt.close()


def plot_production_trends(data: pd.DataFrame,
                           variety: Optional[str] = None,
                           metrics: Optional[list] = None):
    """
    Crea grafici di trend per le metriche di produzione.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con i dati di produzione
    variety : str, optional
        Varietà specifica da visualizzare
    metrics : list, optional
        Lista delle metriche da visualizzare
    """
    if metrics is None:
        metrics = ['olive_prod', 'oil_prod', 'water_need']

    # Filtra per varietà se specificata
    if variety:
        data = data[data['variety'] == variety]

    # Crea subplot per ogni metrica
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.lineplot(data=data, x='year', y=metric, ax=ax)
        if variety:
            ax.set_title(f'{metric} per {variety}')
        else:
            ax.set_title(f'{metric} - Tutte le varietà')
        ax.set_xlabel('Anno')

    plt.tight_layout()

    # Salva il plot
    title = f'production_trends{"_" + variety if variety else ""}'
    save_plot(plt, title)
    plt.close()


def plot_correlation_matrix(data: pd.DataFrame,
                            variables: Optional[list] = None,
                            title: str = "Matrice di Correlazione"):
    """
    Crea una matrice di correlazione con heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con i dati
    variables : list, optional
        Lista delle variabili da includere
    title : str
        Titolo del plot
    """
    if variables:
        corr_matrix = data[variables].corr()
    else:
        corr_matrix = data.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f')

    plt.title(title)
    plt.tight_layout()

    # Salva il plot
    save_plot(plt, 'correlation_matrix')
    plt.close()


def setup_plotting_style():
    """
    Configura lo stile dei plot per uniformità.
    """
    plt.style.use('seaborn')
    sns.set_palette("husl")

    # Impostazioni personalizzate
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10