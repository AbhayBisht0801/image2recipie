import os
import urllib.request as request 
import gdown
from Image2Recipie import logger
import tensorflow as tf
from Image2Recipie.utils.common import save_object,get_maxlen,load_object
from tensorflow.keras.layers import Dense,Concatenate,LSTM,Bidirectional,Dropout,Input,GlobalAvgPool2D,Embedding,LayerNormalization,Bidirectional,BatchNormalization
import cv2 as cv
import pandas as pd
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Image2Recipie.config.configuration import Model_prepare_Config
from pathlib import Path
 

class PrepareModel:
    def __init__(self,config:Model_prepare_Config):
        self.config = config
        self.input_image=None 
    @staticmethod
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)
    
    def prepare_model(self):
        features=load_object(self.config.text_feature)
        image_input = Input(shape=(160, 160, 3))  # Example input shape for ResNet50
        text_input = Input(shape=(features['input_maxlen'],))  # Text input

        # Base model for image input (ResNet50)
        base_model = ResNet50(include_top=self.config.params_include_top, weights=self.config.params_weights, input_shape=self.config.params_image_size)
        base_model.trainable = False

        # Get the output from the base model
        x = base_model(image_input)  # Shape: (batch_size, height, width, channels)

        # Flatten the output from the base model
        x = GlobalAvgPool2D()(x)  # Shape: (batch_size, channels)
        x = Dense(128, activation='relu')(x)
        x=BatchNormalization()(x)
        x=Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x=BatchNormalization()(x)
        x=Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x=BatchNormalization()(x)
        # Text input processing
        masking = tf.keras.layers.Masking(mask_value=0)  # Set mask_value as scalar (commonly 0)
        encoder_inputs_masked = masking(text_input)

        # Embedding and LSTM layers for text input
        embed = Embedding(input_dim=features['input_maxlen']+1, output_dim=128, input_length=features['input_maxlen'])(encoder_inputs_masked)
        lstm2 = Bidirectional(LSTM(64, return_sequences=True))(embed)
        layer_norm=LayerNormalization()(lstm2)
        lstm3 = Bidirectional(LSTM(32, return_sequences=False))(layer_norm)
        layer_norm1=LayerNormalization()(lstm3)
        concatenate=Concatenate()([x,layer_norm1])
        # Concatenate image and text features

        encoder_state_h = Dense(32)(concatenate)
        encoder_state_c = Dense(32)(concatenate)
        encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder input
        decoder_input = Input(shape=(None,), name='decoder_inputs')  # Changed shape

        # Decoder embedding
        decoder_embedding = Embedding(input_dim=features['output_vocab_size'] + 1,  # Add +1 for padding token
            output_dim=32,
                                    mask_zero=True)(decoder_input)

        # Decoder LSTM
        decoder_lstm = LSTM(32, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # Final dense layer for decoder outputs
        decoder_dense = Dense(features['output_vocab_size']+1, activation='softmax', name='decoder_dense')  # Changed output dimension
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the final model
        model = Model(inputs=[image_input, text_input, decoder_input], outputs=decoder_outputs)
        adam=Adam(learning_rate=self.config.params_learning_rate)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.save_model(path=self.config.training_model,model=model)