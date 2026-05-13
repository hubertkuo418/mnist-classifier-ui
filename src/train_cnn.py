import tensorflow as tf
from tensorflow.keras import layers, models


def train_cnn(x_train_cnn, y_train):

    # Model
    model_cnn = models.Sequential([

        tf.keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile
    model_cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training
    history = model_cnn.fit(
        x_train_cnn, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2
    )

    return model_cnn, history