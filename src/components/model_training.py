import os
import sys
from src.utils import Helpers
from src.logger import logging
from xgboost import XGBRegressor
from dataclasses import dataclass
from sklearn.metrics import r2_score
from src.configs import model_params
from catboost import CatBoostRegressor
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, \
    OrthogonalMatchingPursuit, ARDRegression, RANSACRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer(Helpers):

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, x_data, y_data):
        # try:
            x_train, x_test, y_train, y_test =  train_test_split(x_data, y_data, test_size=0.2, random_state=42 )

            models = {
                # 'SVR': SVR(),
                # 'Ridge Regression': Ridge(),
                # 'Lasso Regression': Lasso(),
                # 'MLP Regressor': MLPRegressor(),
                # 'XGB Regressor': XGBRegressor(),
                # 'ARD Regression': ARDRegression(),
                # 'Huber Regressor': HuberRegressor(),
                # 'RANSAC Regressor': RANSACRegressor(),
                # 'ElasticNet Regression': ElasticNet(),
                # "Linear Regression": LinearRegression(),
                # 'TheilSen Regressor': TheilSenRegressor(),
                # 'Ada Boost Regressor': AdaBoostRegressor(),
                # 'Cat Boost Regressor': CatBoostRegressor(),
                # 'Extra Trees Regressor': ExtraTreesRegressor(),
                # 'Bayesian Ridge Regression': BayesianRidge(),
                # 'K Neighbors Regressor': KNeighborsRegressor(),
                # 'Decision Tree Regressor': DecisionTreeRegressor(),
                # 'Random Forest Regressor': RandomForestRegressor(),
                # 'Gaussian Process Regressor': GaussianProcessRegressor(),
                # 'Gradient Boosting Regressor': GradientBoostingRegressor(),
                # 'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
                # 'Passive Aggressive Regressor': PassiveAggressiveRegressor(),
            }

            report, _models = self.model_evaluation(x_train, x_test, y_train, y_test, models, model_params)

            best_score = max(report.values())
            if best_score < 0.6:
                print("No best model found")

            best_model_name = list(report.keys())[list(report.values()).index(best_score)]

            best_model = _models[best_model_name]

            self.save_object(self.model_trainer_config.trained_model_path, best_model)

            predicted = best_model.predict(x_test)

            return r2_score(y_test, predicted)
        # except Exception as e:
            # raise CustomException(e, sys)
