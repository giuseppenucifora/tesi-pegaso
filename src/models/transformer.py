import tensorflow as tf
from tf.keras import layers
from typing import Tuple, Optional, List


@tf.keras.saving.register_keras_serializable()
class DataAugmentation(layers.Layer):
    """
    Layer personalizzato per l'augmentation dei dati temporali.

    Attributes
    ----------
    noise_stddev : float
        Deviazione standard del rumore gaussiano
    """

    def __init__(self, noise_stddev: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Applica l'augmentation durante il training.

        Parameters
        ----------
        inputs : tf.Tensor
            Dati di input
        training : bool, optional
            Flag che indica se siamo in fase di training

        Returns
        -------
        tf.Tensor
            Dati aumentati se in training, altrimenti dati originali
        """
        if training:
            return inputs + tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.noise_stddev
            )
        return inputs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"noise_stddev": self.noise_stddev})
        return config


@tf.keras.saving.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    """
    Layer per l'encoding posizionale nel transformer.

    Attributes
    ----------
    d_model : int
        Dimensionalità del modello
    """

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape: tf.TensorShape):
        """
        Costruisce la matrice di encoding posizionale.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape dell'input
        """
        _, seq_length, _ = input_shape

        # Crea la matrice di encoding posizionale
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32) *
            (-tf.math.log(10000.0) / self.d_model)
        )

        # Calcola sin e cos
        pos_encoding = tf.zeros((1, seq_length, self.d_model))
        pos_encoding_even = tf.sin(position * div_term)
        pos_encoding_odd = tf.cos(position * div_term)

        # Assegna i valori alle posizioni pari e dispari
        pos_encoding = tf.concat(
            [tf.expand_dims(pos_encoding_even, -1),
             tf.expand_dims(pos_encoding_odd, -1)],
            axis=-1
        )
        pos_encoding = tf.reshape(pos_encoding, (1, seq_length, -1))
        pos_encoding = pos_encoding[:, :, :self.d_model]

        # Salva l'encoding come peso non trainabile
        self.pos_encoding = self.add_weight(
            shape=(1, seq_length, self.d_model),
            initializer=tf.keras.initializers.Constant(pos_encoding),
            trainable=False,
            name='positional_encoding'
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applica l'encoding posizionale.

        Parameters
        ----------
        inputs : tf.Tensor
            Dati di input

        Returns
        -------
        tf.Tensor
            Dati con encoding posizionale aggiunto
        """
        batch_size = tf.shape(inputs)[0]
        return inputs + tf.tile(self.pos_encoding, [batch_size, 1, 1])

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


@tf.keras.saving.register_keras_serializable()
class OliveTransformerBlock(layers.Layer):
    """
    Blocco transformer personalizzato per dati di produzione olive.

    Attributes
    ----------
    num_heads : int
        Numero di teste di attenzione
    key_dim : int
        Dimensione delle chiavi
    ff_dim : int
        Dimensione del feed-forward network
    dropout : float
        Tasso di dropout
    """

    def __init__(self, num_heads: int, key_dim: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(key_dim)
        ])

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass del blocco transformer.

        Parameters
        ----------
        inputs : tf.Tensor
            Dati di input
        training : bool, optional
            Flag di training

        Returns
        -------
        tf.Tensor
            Output del blocco transformer
        """
        # Multi-head attention
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout
        })
        return config


def create_olive_oil_transformer(
        temporal_shape: Tuple[int, int],
        static_shape: Tuple[int],
        num_outputs: int,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        num_transformer_blocks: int = 4,
        mlp_units: List[int] = [256, 128, 64],
        dropout: float = 0.2
) -> tf.keras.Model:
    """
    Crea un transformer per la predizione della produzione di olio d'oliva.

    Parameters
    ----------
    temporal_shape : tuple
        Shape dei dati temporali (timesteps, features)
    static_shape : tuple
        Shape dei dati statici (features,)
    num_outputs : int
        Numero di output del modello
    d_model : int
        Dimensionalità del modello
    num_heads : int
        Numero di teste di attenzione
    ff_dim : int
        Dimensione del feed-forward network
    num_transformer_blocks : int
        Numero di blocchi transformer
    mlp_units : list
        Unità nei layer MLP
    dropout : float
        Tasso di dropout

    Returns
    -------
    tf.keras.Model
        Modello transformer configurato
    """
    # Input layers
    temporal_input = layers.Input(shape=temporal_shape, name='temporal')
    static_input = layers.Input(shape=static_shape, name='static')

    # === TEMPORAL PATH ===
    x = layers.LayerNormalization(epsilon=1e-6)(temporal_input)
    x = DataAugmentation()(x)

    # Temporal projection
    x = layers.Dense(d_model // 2, activation='gelu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(d_model, activation='gelu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)

    # Positional encoding
    x = PositionalEncoding(d_model)(x)

    # Transformer blocks
    skip_connection = x
    for _ in range(num_transformer_blocks):
        x = OliveTransformerBlock(num_heads, d_model, ff_dim, dropout)(x)

    # Add final skip connection
    x = layers.Add()([x, skip_connection])

    # Temporal pooling
    attention_pooled = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // 4
    )(x, x)
    attention_pooled = layers.GlobalAveragePooling1D()(attention_pooled)

    # Additional pooling operations
    avg_pooled = layers.GlobalAveragePooling1D()(x)
    max_pooled = layers.GlobalMaxPooling1D()(x)

    # Combine pooling results
    temporal_features = layers.Concatenate()([attention_pooled, avg_pooled, max_pooled])

    # === STATIC PATH ===
    static_features = layers.LayerNormalization(epsilon=1e-6)(static_input)
    for units in [256, 128, 64]:
        static_features = layers.Dense(
            units,
            activation='gelu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(static_features)
        static_features = layers.Dropout(dropout)(static_features)

    # === FEATURE FUSION ===
    combined = layers.Concatenate()([temporal_features, static_features])

    # === MLP HEAD ===
    x = combined
    for units in mlp_units:
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            units,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)
        x = layers.Dropout(dropout)(x)

    # Output layer
    outputs = layers.Dense(
        num_outputs,
        activation='linear',
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(x)

    # Create model
    model = tf.keras.Model(
        inputs={'temporal': temporal_input, 'static': static_input},
        outputs=outputs,
        name='OliveOilTransformer'
    )

    return model