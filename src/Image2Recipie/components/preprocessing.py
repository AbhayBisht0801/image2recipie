import os
import urllib.request as request 
import gdown
from Image2Recipie import logger
import tensorflow as tf
from Image2Recipie.utils.common import save_object,get_maxlen,clean_ingredients,CustomTokenizer,detect_english_text
from Image2Recipie.config.configuration import DataPreprocessConfig
import cv2 as cv
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocess:
    def __init__(self,config:DataPreprocessConfig):
        self.config = config
        self.input_image=None 
    
        

    def output_preprcess(self):
        self.data=pd.read_csv(os.path.join(self.config.unzip_dir,'cuisine_updated.csv'))
        self.features={}
        self.data['cleaned_ingredient']=self.data['cleaned_ingredient'].apply(clean_ingredients)
        self.data['cleaned_ingredient']=self.data['cleaned_ingredient'].apply(lambda x:','.join([x.strip() for x in x.split(',')]))
        self.data['cleaned_ingredient'] = self.data['cleaned_ingredient'].str.replace(r'\b(tablespoons|teaspoons|cups|cup|inch|cloves|whole|pieces|liters|liter|tablespoon|teaspoon|litters|litter|chopped|trimmed|tsp|fresh|pinch|drop|drops)\b', '', regex=True)
        self.data['cleaned_ingredient']=['cleaned_ingredient'].apply(lambda x:','.join([x.strip() for x in x.split(',')]))
        tokenizer=CustomTokenizer()
        y_tokenized=tokenizer.fit_transform(self.data['cleaned_ingredient'])
       
        max_output=max(len(x) for x in y_tokenized)
        ytoken=pad_sequences(y_tokenized,maxlen=max_output,padding='post')
        y_output=to_categorical(ytoken,num_classes=None)
        y_encoder_input=[]
        for i in range(0,ytoken.shape[0]):
            y_encoder_input.append((np.insert(ytoken[i][:-1],0,0)))
        y_encoder=np.array(y_encoder_input)
        
        
        self.features['max_outputtoken']=max_output
        self.features['output_vocab_size']=len(tokenizer.get_word_index())
        save_object(os.path.join(self.config.input_preprocess,'y_input_encoder.pkl'),y_encoder)
        save_object(os.path.join(self.config.output_preprocess,'y_tokenized.pkl'),y_output)
        save_object(os.path.join(self.config.output_preprocess,'tokenizer.pkl'),tokenizer)
        
        logger.info(f'The Preprocessed output is saved in {self.config.output_preprocess}')
    def input_preprocess(self):
        non_english=detect_english_text(df=self.data)
        non_english=detect_english_text(df=self.data)
        self.data.drop(index=non_english,inplace=True)
        self.data.reset_index(drop=True,inplace=True)
        
        token=Tokenizer()
        token.fit_on_texts(self.data['name'])
        
        Xtoken=token.texts_to_sequences(self.data['name'])
        maxlen=max(len(x) for x in Xtoken)
        Xtoken=pad_sequences(Xtoken,maxlen=maxlen)
        self.features['input_maxlen']=maxlen
        vocab_size=len(token.word_index)
        self.features['input_vocal_size']=vocab_size
        X=[]
        for i in self.data['image_path']:
            
            img = cv.imread(os.path.join(self.config.unzip_dir, "data/data/" + i))
            img=cv.resize(img,(160,160))
            X.append(img)
        input_image=np.array(X)
        save_object(self.config.image_input,input_image)
        save_object(self.config.text_input,Xtoken)
        save_object(os.path.join(self.config.input_preprocess,'input_tokenizer.pkl'),token)
        print(self.features)
        save_object(os.path.join(self.config.root_dir,'features.pkl'),self.features)
        logger.info(f'The Preprocessed input is saved in {self.config.input_preprocess}')




        
    

            
            



