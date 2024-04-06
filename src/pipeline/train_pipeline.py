import os 
import sys

from src.exception import CustomException 
from src.logger import logging
import pandas as pd 

from src.components.data_ingestion import DataIngestionConfig 
from src.components.data_ingestion import DataIngestion  

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer




if __name__ == "__main__": 
    obj=DataIngestion()
    train_data ,test_data =obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data ,test_data)   # here we get preprocess data 
    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
    
    
# python -m src.pipeline.train_pipeline  <-- run in terminal 