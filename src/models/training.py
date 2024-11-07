import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List
import os
import keras
from src.models.transformer import create_olive_oil_transformer
from src.models.callbacks import CustomCallback, WarmUpLearningRateSchedule


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Compila il modello con le impostazioni ottimizzate.

    Parameters
    ----------
    model : tf.keras.Model
        Modello da compilare
    learning_rate : float
        Learning rate iniziale

    Returns
    -------
    tf.keras.Model
        Modello compilato
    """
    lr_schedule = WarmUpLearningRateSchedule(
        initial_learning_rate=learning_rate,
        warmup_steps=500,
        decay_steps=5000
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01
        ),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )

    return model


def create_callbacks(target_names: List[str],
                     val_data: Dict,
                     val_targets: np.ndarray) -> List[tf.keras.callbacks.Callback]:
    """
    Crea i callbacks per il training del modello.

    Parameters
    ----------
    target_names : list
        Lista dei nomi dei target
    val_data : dict
        Dati di validazione
    val_targets : np.ndarray
        Target di validazione

    Returns
    -------
    list
        Lista dei callbacks configurati
    """

    class TargetSpecificMetric(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, target_names):
            super().__init__()
            self.validation_data = validation_data
            self.target_names = target_names

        def on_epoch_end(self, epoch, logs={}):
            x_val, y_val = self.validation_data
            y_pred = self.model.predict(x_val, verbose=0)

            for i, name in enumerate(self.target_names):
                mae = np.mean(np.abs(y_val[:, i] - y_pred[:, i]))
                logs[f'val_{name}_mae'] = mae

    # Crea le cartelle per i checkpoint e i log
    os.makedirs('./kaggle/working/models/oil_transformer/checkpoints', exist_ok=True)
    os.makedirs('./kaggle/working/models/oil_transformer/logs', exist_ok=True)

    callbacks = [
        # Early Stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.0005,
            mode='min'
        ),

        # Model Checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./kaggle/working/models/oil_transformer/checkpoints/model_{epoch:02d}_{val_loss:.4f}.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=True
        ),

        # Target specific metrics
        TargetSpecificMetric(
            validation_data=(val_data, val_targets),
            target_names=target_names
        ),

        # Reduce LR on Plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),

        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./kaggle/working/models/oil_transformer/logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    return callbacks


def setup_transformer_training(train_data: Dict,
                               train_targets: np.ndarray,
                               val_data: Dict,
                               val_targets: np.ndarray) -> Tuple[tf.keras.Model, List, List[str]]:
    """
    Configura e prepara il transformer con dimensioni dinamiche.

    Parameters
    ----------
    train_data : dict
        Dati di training
    train_targets : np.ndarray
        Target di training
    val_data : dict
        Dati di validazione
    val_targets : np.ndarray
        Target di validazione

    Returns
    -------
    tuple
        (model, callbacks, target_names)
    """
    # Estrai le shape dai dati
    temporal_shape = (train_data['temporal'].shape[1], train_data['temporal'].shape[2])
    static_shape = (train_data['static'].shape[1],)
    num_outputs = train_targets.shape[1]

    print(f"Shape rilevate:")
    print(f"- Temporal shape: {temporal_shape}")
    print(f"- Static shape: {static_shape}")
    print(f"- Numero di output: {num_outputs}")

    # Target names
    target_names = ['olive_prod', 'min_oil_prod', 'max_oil_prod', 'avg_oil_prod', 'total_water_need']

    assert len(target_names) == num_outputs, \
        f"Il numero di target names ({len(target_names)}) non corrisponde al numero di output ({num_outputs})"

    # Crea il modello
    model = create_olive_oil_transformer(
        temporal_shape=temporal_shape,
        static_shape=static_shape,
        num_outputs=num_outputs
    )

    # Compila il modello
    model = compile_model(model)

    # Crea i callbacks
    callbacks = create_callbacks(target_names, val_data, val_targets)

    return model, callbacks, target_names


def train_transformer(train_data: Dict,
                      train_targets: np.ndarray,
                      val_data: Dict,
                      val_targets: np.ndarray,
                      epochs: int = 150,
                      batch_size: int = 64,
                      save_name: str = 'final_model') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Funzione principale per l'addestramento del transformer.

    Parameters
    ----------
    train_data : dict
        Dati di training
    train_targets : np.ndarray
        Target di training
    val_data : dict
        Dati di validazione
    val_targets : np.ndarray
        Target di validazione
    epochs : int
        Numero di epoche
    batch_size : int
        Dimensione del batch
    save_name : str
        Nome per salvare il modello

    Returns
    -------
    tuple
        (model, history)
    """
    # Setup del modello
    model, callbacks, target_names = setup_transformer_training(
        train_data, train_targets, val_data, val_targets
    )

    # Mostra il summary del modello
    model.summary()
    os.makedirs(f"./kaggle/working/models/oil_transformer/", exist_ok=True)
    keras.utils.plot_model(model, f"./kaggle/working/models/oil_transformer/{save_name}.png", show_shapes=True)

    # Training
    history = model.fit(
        x=train_data,
        y=train_targets,
        validation_data=(val_data, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Salva il modello
    save_path = f'./kaggle/working/models/oil_transformer/{save_name}.keras'
    model.save(save_path, save_format='keras')

    os.makedirs(f'./kaggle/working/models/oil_transformer/weights/', exist_ok=True)
    model.save_weights(f'./kaggle/working/models/oil_transformer/weights')
    print(f"\nModello salvato in: {save_path}")

    return model, history


def retrain_model(base_model: tf.keras.Model,
                  train_data: Dict,
                  train_targets: np.ndarray,
                  val_data: Dict,
                  val_targets: np.ndarray,
                  test_data: Dict,
                  test_targets: np.ndarray,
                  epochs: int = 50,
                  batch_size: int = 128) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict]:
    """
    Implementa il retraining del modello con i dati combinati.

    Parameters
    ----------
    base_model : tf.keras.Model
        Modello base da riaddestrate
    train_data : dict
        Dati di training
    train_targets : np.ndarray
        Target di training
    val_data : dict
        Dati di validazione
    val_targets : np.ndarray
        Target di validazione
    test_data : dict
        Dati di test
    test_targets : np.ndarray
        Target di test
    epochs : int
        Numero di epoche
    batch_size : int
        Dimensione del batch

    Returns
    -------
    tuple
        (model, history, final_metrics)
    """
    print("Valutazione performance iniziali del modello...")
    initial_metrics = {
        'train': evaluate_model_performance(base_model, train_data, train_targets, "training"),
        'val': evaluate_model_performance(base_model, val_data, val_targets, "validazione"),
        'test': evaluate_model_performance(base_model, test_data, test_targets, "test")
    }

    # Combina i dati
    combined_data = {
        'temporal': np.concatenate([
            train_data['temporal'],
            val_data['temporal'],
            test_data['temporal']
        ]),
        'static': np.concatenate([
            train_data['static'],
            val_data['static'],
            test_data['static']
        ])
    }
    combined_targets = np.concatenate([train_targets, val_targets, test_targets])

    # Nuova suddivisione
    indices = np.arange(len(combined_targets))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * 0.9)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    # Prepara i dati per il retraining
    retrain_data = {k: v[train_idx] for k, v in combined_data.items()}
    retrain_targets = combined_targets[train_idx]
    retrain_val_data = {k: v[val_idx] for k, v in combined_data.items()}
    retrain_val_targets = combined_targets[val_idx]

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./kaggle/working/models/oil_transformer/retrain_checkpoints/model_{epoch:02d}_{val_loss:.4f}.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=True
        )
    ]

    # Ricompila il modello
    base_model = compile_model(
        base_model,
        learning_rate=1e-4  # Learning rate più basso per il fine-tuning
    )

    print("\nAvvio retraining...")
    history = base_model.fit(
        retrain_data,
        retrain_targets,
        validation_data=(retrain_val_data, retrain_val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("\nValutazione performance finali...")
    final_metrics = {
        'train': evaluate_model_performance(base_model, train_data, train_targets, "training"),
        'val': evaluate_model_performance(base_model, val_data, val_targets, "validazione"),
        'test': evaluate_model_performance(base_model, test_data, test_targets, "test")
    }

    # Salva il modello
    save_path = './kaggle/working/models/oil_transformer/retrained_model.keras'
    base_model.save(save_path, save_format='keras')
    print(f"\nModello riaddestrato salvato in: {save_path}")

    # Report miglioramenti
    print("\nMiglioramenti delle performance:")
    for dataset in ['train', 'val', 'test']:
        print(f"\nSet {dataset}:")
        for metric in initial_metrics[dataset].keys():
            initial = initial_metrics[dataset][metric]
            final = final_metrics[dataset][metric]
            improvement = ((initial - final) / initial) * 100
            print(f"{metric}: {improvement:.2f}% di miglioramento")

    return base_model, history, final_metrics
