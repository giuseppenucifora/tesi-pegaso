import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy import stats


def calculate_real_error(
        model,
        test_data: Dict,
        test_targets: np.ndarray,
        scaler_y,
        target_names: Optional[List[str]] = None
) -> Tuple[List[float], List[float]]:
    """
    Calcola l'errore reale denormalizzando le predizioni.

    Parameters
    ----------
    model : tf.keras.Model
        Modello addestrato
    test_data : dict
        Dati di test
    test_targets : np.ndarray
        Target di test
    scaler_y : scaler
        Scaler utilizzato per normalizzare i target
    target_names : list, optional
        Nomi dei target

    Returns
    -------
    tuple
        (percentage_errors, absolute_errors)
    """
    # Predizioni
    predictions = model.predict(test_data)

    # Denormalizza predizioni e target
    predictions_real = scaler_y.inverse_transform(predictions)
    targets_real = scaler_y.inverse_transform(test_targets)

    # Calcola errori percentuali e assoluti
    percentage_errors = []
    absolute_errors = []

    if target_names is None:
        target_names = [f'target_{i}' for i in range(predictions_real.shape[1])]

    # Calcola errori per ogni target
    for i in range(predictions_real.shape[1]):
        mae = np.mean(np.abs(predictions_real[:, i] - targets_real[:, i]))
        mape = np.mean(np.abs((predictions_real[:, i] - targets_real[:, i]) / targets_real[:, i])) * 100
        percentage_errors.append(mape)
        absolute_errors.append(mae)

        print(f"\n{target_names[i]}:")
        print(f"MAE assoluto: {mae:.2f}")
        print(f"Errore percentuale medio: {mape:.2f}%")
        print(f"Precisione: {100 - mape:.2f}%")
        print("-" * 50)

    return percentage_errors, absolute_errors


def evaluate_model_performance(
        model,
        data: Dict,
        targets: np.ndarray,
        set_name: str = "",
        threshold: Optional[float] = None
) -> Dict:
    """
    Valuta le performance del modello su un set di dati.

    Parameters
    ----------
    model : tf.keras.Model
        Modello da valutare
    data : dict
        Dati di input
    targets : np.ndarray
        Target reali
    set_name : str
        Nome del set di dati
    threshold : float, optional
        Soglia per calcolare accuracy binaria

    Returns
    -------
    dict
        Dizionario con le metriche calcolate
    """
    predictions = model.predict(data, verbose=0)
    metrics = {}

    target_names = ['olive_prod', 'min_oil_prod', 'max_oil_prod', 'avg_oil_prod', 'total_water_need']

    for i, name in enumerate(target_names):
        # Metriche di base
        mae = np.mean(np.abs(targets[:, i] - predictions[:, i]))
        mse = np.mean(np.square(targets[:, i] - predictions[:, i]))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets[:, i] - predictions[:, i]) / (targets[:, i] + 1e-7))) * 100

        # R2 score
        ss_res = np.sum(np.square(targets[:, i] - predictions[:, i]))
        ss_tot = np.sum(np.square(targets[:, i] - np.mean(targets[:, i])))
        r2 = 1 - (ss_res / (ss_tot + 1e-7))

        # Salva le metriche
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_mape"] = mape
        metrics[f"{name}_r2"] = r2

        # Calcola accuracy binaria se fornita una soglia
        if threshold is not None:
            binary_acc = np.mean(
                (predictions[:, i] > threshold) == (targets[:, i] > threshold)
            )
            metrics[f"{name}_binary_acc"] = binary_acc

    if set_name:
        print(f"\nPerformance sul set {set_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    return metrics


def calculate_efficiency_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        resource_usage: np.ndarray
) -> Dict:
    """
    Calcola metriche di efficienza basate sull'utilizzo delle risorse.

    Parameters
    ----------
    predictions : np.ndarray
        Predizioni del modello
    targets : np.ndarray
        Target reali
    resource_usage : np.ndarray
        Dati sull'utilizzo delle risorse

    Returns
    -------
    dict
        Metriche di efficienza
    """
    metrics = {}

    # Efficienza di produzione
    production_efficiency = predictions / (resource_usage + 1e-7)
    target_efficiency = targets / (resource_usage + 1e-7)

    # Calcola metriche
    metrics['mean_efficiency'] = np.mean(production_efficiency)
    metrics['efficiency_error'] = np.mean(np.abs(production_efficiency - target_efficiency))
    metrics['efficiency_std'] = np.std(production_efficiency)

    # ROI stimato
    estimated_roi = (predictions - resource_usage) / (resource_usage + 1e-7)
    actual_roi = (targets - resource_usage) / (resource_usage + 1e-7)
    metrics['roi_error'] = np.mean(np.abs(estimated_roi - actual_roi))

    # Sostenibilità
    metrics['resource_utilization'] = np.mean(predictions / resource_usage)
    metrics['efficiency_improvement'] = (
                                                np.mean(production_efficiency) - np.mean(target_efficiency)
                                        ) / np.mean(target_efficiency) * 100

    return metrics


def calculate_forecast_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray,
        horizons: List[int]
) -> Dict:
    """
    Calcola l'accuratezza delle previsioni per diversi orizzonti temporali.

    Parameters
    ----------
    predictions : np.ndarray
        Predizioni del modello
    targets : np.ndarray
        Target reali
    horizons : list
        Lista degli orizzonti temporali da valutare

    Returns
    -------
    dict
        Accuratezza per ogni orizzonte
    """
    accuracy_metrics = {}

    for horizon in horizons:
        # Seleziona dati per l'orizzonte corrente
        pred_horizon = predictions[:-horizon]
        target_horizon = targets[horizon:]

        # Calcola metriche
        mae = np.mean(np.abs(pred_horizon - target_horizon))
        mape = np.mean(np.abs((pred_horizon - target_horizon) / (target_horizon + 1e-7))) * 100
        rmse = np.sqrt(np.mean(np.square(pred_horizon - target_horizon)))

        # Calcola il coefficiente di correlazione
        corr = np.corrcoef(pred_horizon.flatten(), target_horizon.flatten())[0, 1]

        # Salva le metriche
        accuracy_metrics[f'horizon_{horizon}'] = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'correlation': corr
        }

        print(f"\nMetriche per orizzonte {horizon}:")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"Correlazione: {corr:.4f}")

    return accuracy_metrics


def compute_confidence_intervals(
        predictions: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola intervalli di confidenza usando bootstrap.

    Parameters
    ----------
    predictions : np.ndarray
        Predizioni del modello
    alpha : float
        Livello di significatività
    n_bootstrap : int
        Numero di campioni bootstrap

    Returns
    -------
    tuple
        (lower_bound, upper_bound, mean_predictions)
    """
    n_samples, n_targets = predictions.shape
    bootstrap_means = np.zeros((n_bootstrap, n_targets))

    # Bootstrap sampling
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, size=n_samples)
        bootstrap_sample = predictions[indices]
        bootstrap_means[i] = np.mean(bootstrap_sample, axis=0)

    # Calcola intervalli di confidenza
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_means, upper_percentile, axis=0)
    mean_predictions = np.mean(predictions, axis=0)

    # Calcola intervalli usando t-distribution
    std_error = np.std(bootstrap_means, axis=0)
    t_value = stats.t.ppf(1 - alpha / 2, df=n_samples - 1)
    margin_error = t_value * std_error

    print("\nIntervalli di Confidenza:")
    for i in range(n_targets):
        print(f"\nTarget {i + 1}:")
        print(f"Media: {mean_predictions[i]:.4f}")
        print(f"Intervallo: [{lower_bound[i]:.4f}, {upper_bound[i]:.4f}]")
        print(f"Margine di errore: ±{margin_error[i]:.4f}")

    return lower_bound, upper_bound, mean_predictions