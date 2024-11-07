import tensorflow as tf
from tf.keras import layers
from typing import List, Optional


@tf.keras.saving.register_keras_serializable()
class MultiScaleAttention(layers.Layer):
    """
    Layer di attenzione multi-scala per catturare pattern temporali a diverse granularità.

    Attributes
    ----------
    num_heads : int
        Numero di teste di attenzione
    head_dim : int
        Dimensionalità per ogni testa
    scales : List[int]
        Lista delle scale temporali da considerare
    """

    def __init__(
            self,
            num_heads: int = 8,
            head_dim: int = 64,
            scales: List[int] = [1, 2, 4],
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scales = scales
        self.dropout = dropout

        # Creiamo un'attention layer per ogni scala
        self.attention_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_dim,
                dropout=dropout,
                name=f'attention_scale_{scale}'
            ) for scale in scales
        ]

        # Layer per combinare le diverse scale
        self.combine = layers.Dense(
            head_dim * num_heads,
            activation='gelu',
            name='scale_combination'
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Lista per salvare gli output delle diverse scale
        scale_outputs = []

        for scale, attention in zip(self.scales, self.attention_layers):
            # Applica max pooling per ridurre la sequenza alla scala corrente
            if scale > 1:
                pooled = tf.keras.layers.MaxPool1D(
                    pool_size=scale,
                    strides=scale
                )(inputs)
            else:
                pooled = inputs

            # Applica attenzione alla sequenza ridotta
            attended = attention(pooled, pooled)

            # Se necessario, riporta alla dimensione originale
            if scale > 1:
                attended = tf.keras.layers.UpSampling1D(size=scale)(attended)
                # Taglia eventuali timestep in eccesso
                attended = attended[:, :tf.shape(inputs)[1], :]

            scale_outputs.append(attended)

        # Concatena e combina gli output delle diverse scale
        concatenated = tf.concat(scale_outputs, axis=-1)
        output = self.combine(concatenated)

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "scales": self.scales,
            "dropout": self.dropout
        })
        return config


@tf.keras.saving.register_keras_serializable()
class TemporalConvBlock(layers.Layer):
    """
    Blocco di convoluzione temporale con residual connection.

    Attributes
    ----------
    filters : int
        Numero di filtri convoluzionali
    kernel_sizes : List[int]
        Lista delle dimensioni dei kernel da utilizzare
    dilation_rates : List[int]
        Lista dei tassi di dilatazione
    """

    def __init__(
            self,
            filters: int = 64,
            kernel_sizes: List[int] = [3, 5, 7],
            dilation_rates: List[int] = [1, 2, 4],
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        self.dropout = dropout

        # Crea i layer convoluzionali
        self.conv_layers = []
        for k_size in kernel_sizes:
            for d_rate in dilation_rates:
                self.conv_layers.append(
                    layers.Conv1D(
                        filters=filters // (len(kernel_sizes) * len(dilation_rates)),
                        kernel_size=k_size,
                        dilation_rate=d_rate,
                        padding='same',
                        activation='gelu'
                    )
                )

        # Layer per il processing finale
        self.combine = layers.Conv1D(filters, 1)
        self.layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Lista per gli output di ogni convoluzione
        conv_outputs = []

        # Applica ogni combinazione di kernel size e dilation rate
        for conv in self.conv_layers:
            conv_outputs.append(conv(inputs))

        # Concatena tutti gli output
        concatenated = tf.concat(conv_outputs, axis=-1)

        # Combinazione finale
        x = self.combine(concatenated)
        x = self.layernorm(x)
        x = self.dropout(x, training=training)

        # Residual connection
        return x + inputs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_sizes": self.kernel_sizes,
            "dilation_rates": self.dilation_rates,
            "dropout": self.dropout
        })
        return config


@tf.keras.saving.register_keras_serializable()
class WeatherEmbedding(layers.Layer):
    """
    Layer per l'embedding di feature meteorologiche.
    Combina embedding categorici e numerici.

    Attributes
    ----------
    embedding_dim : int
        Dimensionalità dell'embedding
    num_numerical : int
        Numero di feature numeriche
    categorical_features : dict
        Dizionario con feature categoriche e loro cardinalità
    """

    def __init__(
            self,
            embedding_dim: int = 32,
            num_numerical: int = 8,
            categorical_features: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_numerical = num_numerical
        self.categorical_features = categorical_features or {
            'season': 4,
            'time_period': 4,
            'weather_condition': 10
        }

        # Layer per feature numeriche
        self.numerical_projection = layers.Dense(
            embedding_dim,
            activation='gelu'
        )

        # Layer per feature categoriche
        self.categorical_embeddings = {
            name: layers.Embedding(
                input_dim=num_categories,
                output_dim=embedding_dim
            )
            for name, num_categories in self.categorical_features.items()
        }

        # Layer di combinazione finale
        self.combine = layers.Dense(embedding_dim, activation='gelu')

    def call(self, inputs: dict) -> tf.Tensor:
        # Processa feature numeriche
        numerical = self.numerical_projection(inputs['numerical'])

        # Lista per gli embedding categorici
        categorical_outputs = []

        # Processa ogni feature categorica
        for name, embedding_layer in self.categorical_embeddings.items():
            if name in inputs['categorical']:
                embedded = embedding_layer(inputs['categorical'][name])
                categorical_outputs.append(embedded)

        # Combina tutti gli embedding
        if categorical_outputs:
            categorical = tf.reduce_mean(tf.stack(categorical_outputs, axis=1), axis=1)
            combined = tf.concat([numerical, categorical], axis=-1)
        else:
            combined = numerical

        return self.combine(combined)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_numerical": self.num_numerical,
            "categorical_features": self.categorical_features
        })
        return config


@tf.keras.saving.register_keras_serializable()
class OliveVarietyEmbedding(layers.Layer):
    """
    Layer per l'embedding delle varietà di olive e delle loro caratteristiche.

    Attributes
    ----------
    embedding_dim : int
        Dimensionalità dell'embedding
    num_varieties : int
        Numero di varietà di olive
    num_techniques : int
        Numero di tecniche di coltivazione
    """

    def __init__(
            self,
            embedding_dim: int = 32,
            num_varieties: int = 11,
            num_techniques: int = 3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_varieties = num_varieties
        self.num_techniques = num_techniques

        # Embedding per varietà e tecniche
        self.variety_embedding = layers.Embedding(
            input_dim=num_varieties,
            output_dim=embedding_dim
        )

        self.technique_embedding = layers.Embedding(
            input_dim=num_techniques,
            output_dim=embedding_dim
        )

        # Layer per feature continue
        self.continuous_projection = layers.Dense(
            embedding_dim,
            activation='gelu'
        )

        # Layer di combinazione
        self.combine = layers.Dense(embedding_dim, activation='gelu')

    def call(self, inputs: dict) -> tf.Tensor:
        # Embedding varietà
        variety_embedded = self.variety_embedding(inputs['variety'])

        # Embedding tecniche
        technique_embedded = self.technique_embedding(inputs['technique'])

        # Proiezione feature continue
        continuous_projected = self.continuous_projection(inputs['continuous'])

        # Combinazione
        combined = tf.concat([
            variety_embedded,
            technique_embedded,
            continuous_projected
        ], axis=-1)

        return self.combine(combined)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_varieties": self.num_varieties,
            "num_techniques": self.num_techniques
        })
        return config