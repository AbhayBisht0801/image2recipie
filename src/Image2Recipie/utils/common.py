import os
from box.exceptions import BoxValueError
import yaml
from Image2Recipie import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import cv2 as cv
import pickle
import numpy as np
from numpy.linalg import norm
import re
import re
from string import punctuation
from nltk.corpus import stopwords
import nltk
import re
from string import punctuation

nltk.download('stopwords')
stopwords=stopwords.words('english')
stopwords.append('ingredients')
stopwords.extend(['trimmed','chopped','gm','inch'])
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise e
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
def format_recipe_ingredients(text):
  result=text.lower().replace('\t','').replace('\n\n',',').replace('\n','').replace(',','',1).replace('\xa0',' ')
  return result
def detect_english_text(df):
    non_english=[]
    for i in range(0,df.shape[0]):
        if re.search(r"[A-Za-z]",df['name'][i][0]):
            None
        else:
            non_english.append(i)
    return non_english
def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen


def clean_ingredients(recipe):
    # Regular expression to match optional quantities, measurements, or just ingredients.
    pattern = r'(\d+\s*\/?\d*)\s*(tablespoons|teaspoons|cups|cup|inch|cloves|whole|grams|g|gm|cup|gram|kgs|kg|ml|pieces|liters|liter|tablespoon?|teaspoon|litters|litter|chopped|trimmed|tsp|fresh|)\s+([A-Za-z\s]+)|([^.,]*?)(?=\bfor\b)|(salt|oil|ghee|water|butter|almonds?|cashews?|raisins?|dates?|pistachios?|walnuts?|apricots?|prunes?|figs?|hazelnuts?|pecans?|macadamia|brazil\s?nuts?|cherries?|cranberries?|blueberries?|peanuts?|\w+ oil|\w+ butter)?'

# Compile the pattern
    regex = re.compile(pattern)
    ingredients = []
    # Find all matches in the text
    matches = regex.findall(recipe)
    for match in matches:
        if match[2]!='':
          if [item for item in match[2].split() if item in stopwords or any(char in item for char in punctuation) or any(char.isdigit() for char in item)]:
            None
          else:
            ingredients.append(match[2].split('for')[0])
        elif match[3]!='':
          if [item for item in match[3].split() if item in stopwords or any(char in item for char in punctuation) or any(char.isdigit() for char in item)]:
            None
          else:
            ingredients.append(match[3])
        elif match[4]!='':
          if [item for item in match[4].split() if item in stopwords or any(char in item for char in punctuation) or any(char.isdigit() for char in item)]:
            None
          else:
            ingredients.append(match[4])
    return ','.join([x for x in ingredients if x!=''])

class CustomTokenizer:
    def __init__(self):
        self.ytoken = None
        self.word_index = None
        self.word_frequency = None

    def fit_transform(self,column):
        y_token=[]
        word_frequency={}
        for i in column:
          for j in i.split(','):
            if j not in word_frequency.keys():
              word_frequency[j]=1
            else:
              word_frequency[j]+=1
        result=sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        word_index={}
        counter=1
        for i in result:
          word_index[i[0]]=counter
          counter+=1
        self.word_index=word_index
        self.word_frequency=word_frequency
        for i in column:
          each_token=[]
          for j in i.split(','):
            each_token.append(word_index[j])
          y_token.append(each_token)
        return y_token
    def inverse_transform(self,output):
      for index,i in enumerate(output):
        if type(i)==list:
          for index,value in enumerate(i):
            for key,values in self.word_index.items():
              if value==values:
                i[index]=key
        else:
          for key,values in self.word_index.items():
            if i==values:
              output[index]=key


      return output
    def get_word_index(self):
        """Returns the word-to-index mapping dictionary."""
        return self.word_index

    def get_word_frequency(self):
        """Returns the word frequency dictionary."""
        return self.word_frequency

