import tensorflow as tf
import tf.keras.layers as layers

def create_radiation_model(input_shape, solar_params_shape=(3,)):
    """
    Modello per la radiazione solare con vincoli di non-negatività.
    """
    # Input layers
    main_input = layers.Input(shape=input_shape, name='main_input')
    solar_input = layers.Input(shape=solar_params_shape, name='solar_params')

    # Branch CNN
    x1 = layers.Conv1D(32, 3, padding='same')(main_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv1D(64, 3, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)

    # Branch LSTM
    x2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(main_input)
    x2 = layers.Bidirectional(layers.LSTM(32))(x2)
    x2 = layers.BatchNormalization()(x2)

    # Solar parameters processing
    x3 = layers.Dense(32)(solar_input)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    # Combine all branches
    x = layers.concatenate([x1, x2, x3])

    # Dense layers with non-negativity constraints
    x = layers.Dense(64, kernel_constraint=tf.keras.constraints.NonNeg())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, kernel_constraint=tf.keras.constraints.NonNeg())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Output layer con vincoli di non-negatività
    output = layers.Dense(1,
                   kernel_constraint=tf.keras.constraints.NonNeg(),
                   activation='relu')(x)

    model = layers.Model(inputs=[main_input, solar_input], outputs=output, name="SolarRadiation")
    return model


def create_energy_model(input_shape):
    """
    Modello migliorato per l'energia solare che sfrutta la relazione con la radiazione.
    Include vincoli di non-negatività e migliore gestione delle dipendenze temporali.
    """
    inputs = layers.Input(shape=input_shape)

    # Branch 1: Elaborazione temporale con attention
    # Multi-head attention per catturare relazioni temporali
    x1 = layers.MultiHeadAttention(num_heads=8, key_dim=32)(inputs, inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    # Temporal Convolution branch per catturare pattern locali
    x2 = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        kernel_constraint=tf.keras.constraints.NonNeg()
    )(inputs)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        kernel_constraint=tf.keras.constraints.NonNeg()
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    # LSTM branch per memoria a lungo termine
    x3 = layers.LSTM(64, return_sequences=True)(inputs)
    x3 = layers.LSTM(32, return_sequences=False)(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    # Global pooling per ogni branch
    x1 = layers.GlobalAveragePooling1D()(x1)
    x2 = layers.GlobalAveragePooling1D()(x2)

    # Concatena tutti i branch
    x = layers.concatenate([x1, x2, x3])

    # Dense layers con vincoli di non-negatività
    x = layers.Dense(
        128,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        kernel_regularizer=layers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        64,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        kernel_regularizer=layers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer con vincolo di non-negatività
    output = layers.Dense(
        1,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        activation='relu',  # Garantisce output non negativo
        kernel_regularizer=layers.l2(0.01)
    )(x)

    model = layers.Model(inputs=inputs, outputs=output, name="SolarEnergy")
    return model


def create_uv_model(input_shape):
    """
    Modello migliorato per l'indice UV che sfrutta sia radiazione che energia solare.
    Include vincoli di non-negatività e considera le relazioni non lineari tra le variabili.
    """
    inputs = layers.Input(shape=input_shape)

    # CNN branch per pattern locali
    x1 = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        kernel_constraint=tf.keras.constraints.NonNeg()
    )(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)

    x1 = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        kernel_constraint=tf.keras.constraints.NonNeg()
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)

    # Attention branch per relazioni complesse
    # Specialmente utile per le relazioni con radiazione ed energia
    x2 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)

    # Dense branch per le feature più recenti
    x3 = layers.GlobalAveragePooling1D()(inputs)
    x3 = layers.Dense(
        64,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        kernel_regularizer=layers.l2(0.01)
    )(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    # Fusion dei branch
    x = layers.concatenate([x1, x2, x3])

    # Dense layers con vincoli di non-negatività
    x = layers.Dense(
        128,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        kernel_regularizer=layers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        64,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        kernel_regularizer=layers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer con vincolo di non-negatività
    output = layers.Dense(
        1,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        activation='relu',  # Garantisce output non negativo
        kernel_regularizer=layers.l2(0.01)
    )(x)

    model = layers.Model(inputs=inputs, outputs=output, name="SolarUV")
    return model