import os
import urllib.request as request 
import gdown
from Image2Recipie import logger
import tensorflow as tf
from Image2Recipie.utils.common import load_object
import cv2 as cv
import pandas as pd
import numpy as np
from Image2Recipie.entity.config_entity import Model_Training_Config
class ModelTraining:
    def __init__(self,config:Model_Training_Config):
        self.config = config
    def get_model(self):
        self.model=tf.keras.models.load_model(
            self.config.training_model
        )
        
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)
    def model_training(self):
        input_images=load_object(self.config.image_input)
        
        input_text=load_object(self.config.text_input)
        
        y=load_object(os.path.join(self.config.output_preprocess,'y_tokenized.pkl'))
        y_encoder_input=load_object(os.path.join(self.config.input_preprocess,'y_input_encoder.pkl'))
        self.model.fit([input_images,input_text,y_encoder_input],y,epochs=self.config.params_epochs,batch_size=self.config.params_batch_size,validation_split=0.2)
        self.save_model(path=self.config.trained_model_path,model=self.model)
