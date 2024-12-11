import os
import json
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.models import load_model


@keras.saving.register_keras_serializable()
class DataAugmentation(tf.keras.layers.Layer):
    """Custom layer per l'augmentation dei dati"""

    def __init__(self, noise_stddev=0.03, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev

    def call(self, inputs, training=None):
        if training:
            return inputs + tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.noise_stddev
            )
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"noise_stddev": self.noise_stddev})
        return config


@keras.saving.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """Custom layer per l'encoding posizionale"""

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        _, seq_length, _ = input_shape

        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32) *
            (-tf.math.log(10000.0) / self.d_model)
        )

        pos_encoding = tf.zeros((1, seq_length, self.d_model))
        pos_encoding_even = tf.sin(position * div_term)
        pos_encoding_odd = tf.cos(position * div_term)

        pos_encoding = tf.concat(
            [tf.expand_dims(pos_encoding_even, -1),
             tf.expand_dims(pos_encoding_odd, -1)],
            axis=-1
        )
        pos_encoding = tf.reshape(pos_encoding, (1, seq_length, -1))
        pos_encoding = pos_encoding[:, :, :self.d_model]

        self.pos_encoding = self.add_weight(
            shape=(1, seq_length, self.d_model),
            initializer=tf.keras.initializers.Constant(pos_encoding),
            trainable=False,
            name='positional_encoding'
        )

        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        pos_encoding_tiled = tf.tile(self.pos_encoding, [batch_size, 1, 1])
        return inputs + pos_encoding_tiled

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


@keras.saving.register_keras_serializable()
class WarmUpLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate=1e-3, warmup_steps=500, decay_steps=5000):
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


@keras.saving.register_keras_serializable()
def weighted_huber_loss(y_true, y_pred):
    weights = tf.constant([1.0, 0.8, 0.8, 1.0, 0.6], dtype=tf.float32)
    huber = tf.keras.losses.Huber(delta=1.0)
    loss = huber(y_true, y_pred)
    weighted_loss = tf.reduce_mean(loss * weights)
    return weighted_loss


def model_fn(model_dir):
    """Carica il modello salvato."""
    custom_objects = {
        'DataAugmentation': DataAugmentation,
        'PositionalEncoding': PositionalEncoding,
        'WarmUpLearningRateSchedule': WarmUpLearningRateSchedule,
        'weighted_huber_loss': weighted_huber_loss
    }

    model = load_model(os.path.join(model_dir, 'model.keras'), custom_objects=custom_objects)
    return model


def input_fn(request_body, request_content_type):
    """Converte l'input in un formato utilizzabile dal modello."""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)

        # Estrai i dati temporali e statici
        temporal_data = np.array(input_data['temporal']).reshape(1, 1, -1)
        static_data = np.array(input_data['static']).reshape(1, -1)

        return {
            'temporal': temporal_data,
            'static': static_data
        }
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Esegue la predizione usando il modello."""
    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, accept):
    """Converte l'output del modello nel formato richiesto."""
    if accept == 'application/json':
        prediction_list = prediction.tolist()
        return json.dumps({
            'prediction': prediction_list
        })
    raise ValueError(f"Unsupported accept type: {accept}")