import tensorflow as tf
from tensorflow.keras import layers, models


def train_cnn(x_train_cnn, y_train):
    """
    Train a Convolutional Neural Network (CNN) for MNIST digit classification.

    Model architecture:
    - 2 Conv2D layers for feature extraction
    - MaxPooling for spatial downsampling
    - Fully connected layer for classification

    Returns:
    - model_cnn: trained CNN model
    - history: training history (loss/accuracy tracking)
    """

    # CNN is used because it can automatically learn spatial features from images
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

    # Compile model with Adam optimizer (stable for CNN training)
    model_cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model with validation split for overfitting monitoring
    history = model_cnn.fit(
        x_train_cnn,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2
    )
    
    model_cnn.save("saved_model/cnn_model.keras")
    
    return model_cnn, history