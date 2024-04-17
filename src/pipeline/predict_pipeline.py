import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import Helpers


class PredictPipeline(Helpers):

    def __init__(self):
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = self.load_object(model_path)
        self.preprocessor = self.load_object(preprocessor_path)

    def predict(self, features):
        try:
            scaled_data = self.preprocessor.transform(features)
            return self.model.predict(scaled_data)
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str,
                 test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
