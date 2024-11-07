import tensorflow as tf
import numpy as np
from typing import Dict, Optional, List
import os
import json
from datetime import datetime


@tf.keras.saving.register_keras_serializable()
class CustomCallback(tf.keras.callbacks.Callback):
    """
    Callback personalizzato per monitorare la non-negatività delle predizioni
    e altre metriche durante il training.
    """

    def __init__(self, validation_data: Optional[tuple] = None):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        try:
            if hasattr(self.model, 'validation_data'):
                val_x = self.model.validation_data[0]
                if isinstance(val_x, list):  # Per il modello della radiazione
                    val_pred = self.model.predict(val_x, verbose=0)
                else:
                    val_pred = self.model.predict(val_x, verbose=0)

                # Verifica non-negatività
                if np.any(val_pred < 0):
                    print("\nWarning: Rilevati valori negativi nelle predizioni")
                    print(f"Min value: {np.min(val_pred)}")

                # Statistiche predizioni
                print(f"\nStatistiche predizioni epoca {epoch}:")
                print(f"Min: {np.min(val_pred):.4f}")
                print(f"Max: {np.max(val_pred):.4f}")
                print(f"Media: {np.mean(val_pred):.4f}")

                # Aggiunge le metriche ai logs
                if logs is not None:
                    logs['val_pred_min'] = np.min(val_pred)
                    logs['val_pred_max'] = np.max(val_pred)
                    logs['val_pred_mean'] = np.mean(val_pred)
        except Exception as e:
            print(f"\nWarning nel CustomCallback: {str(e)}")


@tf.keras.saving.register_keras_serializable()
class WarmUpLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Schedule del learning rate con warm-up lineare e decay esponenziale.
    """

    def __init__(self, initial_learning_rate: float = 1e-3,
                 warmup_steps: int = 500,
                 decay_steps: int = 5000):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        warmup_pct = tf.cast(step, tf.float32) / self.warmup_steps
        warmup_lr = self.initial_learning_rate * warmup_pct
        decay_factor = tf.pow(0.1, tf.cast(step, tf.float32) / self.decay_steps)
        decayed_lr = self.initial_learning_rate * decay_factor
        return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps
        }


class MetricLogger(tf.keras.callbacks.Callback):
    """
    Logger avanzato per metriche di training che salva i risultati in JSON
    e crea grafici di progresso.
    """

    def __init__(self, log_dir: str = './logs',
                 metric_list: Optional[List[str]] = None,
                 save_freq: int = 1):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metric_list = metric_list or ['loss', 'val_loss', 'mae', 'val_mae']
        self.save_freq = save_freq
        self.history = {metric: [] for metric in self.metric_list}

        # Timestamp per il nome del file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'metrics_{self.timestamp}.json')

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        # Aggiorna lo storico
        for metric in self.metric_list:
            if metric in logs:
                self.history[metric].append(float(logs[metric]))

        # Salva i log periodicamente
        if (epoch + 1) % self.save_freq == 0:
            self._save_logs()
            self._create_plots()

    def _save_logs(self):
        """Salva i log in formato JSON."""
        with open(self.log_file, 'w') as f:
            json.dump({
                'history': self.history,
                'epochs': len(next(iter(self.history.values())))
            }, f, indent=4)

    def _create_plots(self):
        """Crea grafici delle metriche."""
        import matplotlib.pyplot as plt

        # Plot per ogni metrica
        for metric in self.metric_list:
            if metric in self.history and len(self.history[metric]) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.history[metric])
                plt.title(f'Model {metric}')
                plt.ylabel(metric)
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(self.log_dir, f'{metric}_{self.timestamp}.png'))
                plt.close()


class EarlyStoppingWithBest(tf.keras.callbacks.EarlyStopping):
    """
    Early stopping avanzato che salva il miglior modello e fornisce
    analisi dettagliate sulla convergenza.
    """

    def __init__(self,
                 monitor: str = 'val_loss',
                 min_delta: float = 0,
                 patience: int = 0,
                 verbose: int = 0,
                 mode: str = 'auto',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = True,
                 start_from_epoch: int = 0):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            start_from_epoch=start_from_epoch
        )
        self.best_epoch = 0
        self.convergence_history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        # Aggiungi il valore corrente alla storia
        self.convergence_history.append(float(current))

        # Calcola statistiche di convergenza
        if len(self.convergence_history) > 1:
            improvement = self.convergence_history[-2] - current
            pct_improvement = (improvement / self.convergence_history[-2]) * 100
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved by {pct_improvement:.2f}%")

        # Aggiorna best_epoch se necessario
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print(f"\nRestoring model weights from epoch {self.best_epoch + 1}")
                    self.model.set_weights(self.best_weights)

    def get_convergence_stats(self) -> Dict:
        """
        Restituisce statistiche dettagliate sulla convergenza.
        """
        if len(self.convergence_history) < 2:
            return {}

        improvements = np.diff(self.convergence_history)
        return {
            'best_epoch': self.best_epoch + 1,
            'best_value': float(self.best),
            'avg_improvement': float(np.mean(improvements)),
            'total_improvement': float(self.convergence_history[0] - self.best),
            'convergence_rate': float(np.mean(np.abs(improvements[1:] / improvements[:-1]))),
            'final_value': float(self.convergence_history[-1])
        }