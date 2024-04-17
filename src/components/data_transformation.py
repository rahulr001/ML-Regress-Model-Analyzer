import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation():

    def __init__(self):
        self.preprocessor_config = DataTransformationConfig()

    @staticmethod
    def get_preprocessor_obj(target_column):
        try:
            raw_df = pd.read_csv(os.path.join('artifacts', 'raw_data.csv'))

            num_features = [feature for feature in raw_df.columns if raw_df[feature].dtype != 'O']

            num_features = [_num for _num in num_features if _num not in target_column + ['contact_list_id']]

            num_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

            preprocessor = ColumnTransformer(
                [
                    ('num_pipline', num_pipline, num_features),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, target_column):
        # try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            return train_df, test_df
        # except Exception as e:
            # raise CustomException(e, sys)
