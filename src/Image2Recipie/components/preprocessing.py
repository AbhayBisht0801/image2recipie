import os
import urllib.request as request 
import gdown
from Image2Recipie import logger
import tensorflow as tf
from Image2Recipie.utils.common import save_object,get_maxlen,clean_ingredients,CustomTokenizer
from Image2Recipie.config.configuration import DataPreprocessConfig
import cv2 as cv
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocess:
    def __init__(self,config:DataPreprocessConfig):
        self.config = config
        self.input_image=None 
    def input_preprocess(self):
        self.data=pd.read_csv(os.path.join(self.config.unzip_dir,'cuisine_updated.csv'))
        self.features={}
        X=[]
        for i in self.data['image_path']:
            
            img = cv.imread(os.path.join(self.config.unzip_dir, "data/data/" + i))
            img = cv.resize(img, (224, 224))
            img=cv.resize(img,(180,180))
            X.append(img)
        input_image=np.array(X)
        save_object(self.config.image_input,input_image)
        maxlen=get_maxlen(self.data['name'])
        token=Tokenizer()
        token.fit_on_texts(self.data['name'])
        
        Xtoken=token.texts_to_sequences(self.data['name'])
        Xtoken=pad_sequences(Xtoken,maxlen=maxlen)
        self.features['input_maxlen']=maxlen
        vocab_size=len(token.word_index)
        self.features['input_vocal_size']=vocab_size
        save_object(self.config.text_input,Xtoken)
        save_object(os.path.join(self.config.input_preprocess,'input_tokenizer.pkl'),token)
        logger.info(f'The Preprocessed input is saved in {self.config.input_preprocess}')
        

    def output_preprcess(self):
        self.data['cleaned_ingredient']=self.data['cleaned_ingredient'].apply(clean_ingredients)
        self.data['cleaned_ingredient']=self.data['cleaned_ingredient'].apply(lambda x:','.join([x.strip() for x in x.split(',')]))
        tokenizer=CustomTokenizer()
        y_tokenized=tokenizer.fit_transform(self.data['cleaned_ingredient'])
       
        max_output=max(len(x) for x in y_tokenized)
        ytoken=pad_sequences(ytoken,maxlen=max_output,padding='post')
        y_output=to_categorical(ytoken,num_classes=None)
        self.features['max_outputtoken']=max_output
        
        save_object(os.path.join(self.config.output_preprocess,'y_tokenized.pkl'),y_output)
        save_object(os.path.join(self.config.output_preprocess,'tokenizer.pkl'),tokenizer)
        save_object(os.path.join(self.config.root_dir,'features.pkl',self.features))
        logger.info(f'The Preprocessed output is saved in {self.config.output_preprocess}')



        
    

            
            



