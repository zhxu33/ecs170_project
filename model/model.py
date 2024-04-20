# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

# Data preparation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    "emirhan_human_dataset/datasets/human_data/train_data",
    target_size=(128, 128),
    batch_size=128,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    "emirhan_human_dataset/datasets/human_data/test_data",
    target_size=(128, 128),
    batch_size=128,
    class_mode='categorical')

# Function to build and train a model using MobileNet for transfer learning
def train_model_with_mobile_net():
    print('#####~Model => MobileNet')
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(128, 128, 3), pooling='avg')
    base_model.trainable = False  # Freeze the base model to not train it
    inputs = base_model.input
    x = Dense(64, activation='relu')(base_model.output)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    history = model.fit(train_generator, validation_data=test_generator, epochs=5, callbacks=callbacks, verbose=1)

    # Save the trained model
    model.save('mobilenet_human_action_model.h5')
    print("Model saved as mobilenet_human_action_model.h5")

    # Save the class indices
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print("Class indices saved to class_indices.json")

    return model

# Build and train the model with MobileNet
model_mobilenet = train_model_with_mobile_net()
# 64.37% accuracy
