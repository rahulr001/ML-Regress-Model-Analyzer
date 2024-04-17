import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from src.components.data_transformation import DataTransformation

class Helpers:

    @staticmethod
    def save_object(file_path, obj):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def load_object(file_path):
        try:
            with open(file_path, 'rb') as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def model_evaluation(x_train, x_test, y_train, y_test , models, params):
        # try:
            preprocessor = DataTransformation().get_preprocessor_obj(['is_opened', 'is_clicked'])
            report = dict()
            _models = dict()
            for model_name, model in models.items():
                print(x_train.shape, y_train.shape)
                grid_search = GridSearchCV(model, params.get(model_name), cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(x_train, y_train)
                # model.set_params(**grid_search.best_params_)
                print(grid_search.best_params_)

                multi_model = MultiOutputRegressor(model)

                model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', multi_model)])
                
                model.fit(x_train, y_train)

                test_prediction = model.predict(x_test)
                score = r2_score(y_test, test_prediction)
                print("model", model_name)
                print('score', score)
                mae = mean_absolute_error(y_test, test_prediction)
                mse = mean_squared_error(y_test, test_prediction)

                print(f'Mean Absolute Error: {mae}')
                print(f'Mean Squared Error: {mse}')

                report[model_name] = score
                _models[model_name] = model

            return report, _models
        # except Exception as e:
            # raise CustomException(e, sys)
