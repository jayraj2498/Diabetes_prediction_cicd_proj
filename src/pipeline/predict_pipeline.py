# here we load the model and we crete dataframe function 

import os 
import sys 
import pandas as pd 

from src.exception import CustomException 
from src.logger import logging 

from src.utils import load_object


class  predictpipeline :
    def __init__(self) :
        pass 
    
    def predict(self , features) :
        try :
            model_path = os.path.join('E:\\Bank_cicd\\artifacts\\model.pkl')
            preprocessor_path= os.path.join('E:\\Bank_cicd\\artifacts\\proprocessor.pkl') 
            
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading") 
            
            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            
            return pred  
        
        except Exception as e :
            raise CustomException(e,sys)
        
        
        
        
class CustomData: 
    def __init__(self, 
                 gender: str,
                 age: float,
                 hypertension: int,
                 heart_disease: int,
                 smoking_history: str,
                 bmi: float,
                 HbA1c_level: float,
                 blood_glucose_level: int
                 ):
        
        
        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.smoking_history = smoking_history
        self.bmi = bmi
        self.HbA1c_level = HbA1c_level
        self.blood_glucose_level = blood_glucose_level
        
        
    # here we make incoming data as dataframe to train our mdel     
        
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict ={
            "gender": [self.gender],
            "age": [self.age],
            "hypertension": [self.hypertension],
            "heart_disease": [self.heart_disease],
            "smoking_history": [self.smoking_history],
            "bmi": [self.bmi],
            "HbA1c_level": [self.HbA1c_level],
            "blood_glucose_level": [self.blood_glucose_level]
            
            }
            return pd.DataFrame(custom_data_input_dict) 
        
        except Exception as e:
            raise CustomException(e,sys)
   
    
       
    
    
# these funct return input in the form of dataframe 
# inshort from our  web application  whatever input is comming same input we are get mapp with the our abouve particaular val 