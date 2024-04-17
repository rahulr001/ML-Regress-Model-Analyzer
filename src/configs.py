model_params = {
    "SVR": {
        'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] ,
        # 'C': [0.1, 1, 10],
        # 'epsilon': [0.1, 0.2, 0.5],
        # 'gamma': ['scale', 'auto']
    },
    #     kernel: ((...) -> Any) | Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = "rbf",
    # degree: Int = 3,
    # gamma: float | Literal['scale', 'auto'] = "scale",
    # coef0: Float = 0,
    # tol: Float = 0.001,
    # C: Float = 1,
    # epsilon: Float = 0.1,
    # shrinking: bool = True,
    # cache_size: Float = 200,
    # verbose: bool = False,
    # max_iter: Int = ...

    "Ridge Regression": {
        'alpha': [0.1, 0.5, 1.0],
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False],
        'max_iter': [1, 3],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'random_state': [42, 2, 88]
    },

    "Lasso Regression": {
        'alpha': [0.1, 0.5, 1.0],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'selection': ['cyclic', 'random']
    },

    "MLP Regressor": {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200, 400, 600]
    },

    "XGB Regressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "ARD Regression": {
        'alpha_1': [1e-06, 1e-05, 1e-04],
        'alpha_2': [1e-06, 1e-05, 1e-04],
        'lambda_1': [1e-06, 1e-05, 1e-04],
        'lambda_2': [1e-06, 1e-05, 1e-04],
        'threshold_lambda': [1000.0, 10000.0, 100000.0],
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },

    "Huber Regressor": {
        'epsilon': [1.35, 1.5, 1.75],
        'alpha': [0.0001, 0.001, 0.01],
        'fit_intercept': [True, False],
        'max_iter': [100, 200, 300]
    },

    "RANSAC Regressor": {
        'min_samples': [None, 0.5, 1.0],
        'residual_threshold': [None, 0.5, 1.0],
        'max_trials': [100, 200, 300]
    },

    "ElasticNet Regression": {
        'alpha': [0.1, 0.5, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'selection': ['cyclic', 'random']
    },

    "Linear Regression": {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },

    "TheilSen Regressor": {
        'fit_intercept': [True, False],
        'max_subpopulation': [None, 100, 200, 300],
        'n_subsamples': [None, 100, 200, 300]
    },

    "Ada Boost Regressor": {
        'random_state': [42],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    },

    "Cat Boost Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },

    "Extra Trees Regressor": {
        'random_state': [42],
        'criterion': ['mse', 'mae'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [100, 200, 300]
    },

    "Bayesian Ridge Regression": {
        'alpha_1': [1e-06, 1e-05, 1e-04],
        'alpha_2': [1e-06, 1e-05, 1e-04],
        'lambda_1': [1e-06, 1e-05, 1e-04],
        'lambda_2': [1e-06, 1e-05, 1e-04],
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },

    "K Neighbors Regressor": {
        'n_neighbors': [5, 10, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },

    "Decision Tree Regressor": {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },

    "Random Forest Regressor": {
        'random_state': [42],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "Gaussian Process Regressor": {
        'normalize_y': [True, False],
        'kernel': [None]
    },

    "Gradient Boosting Regressor": {
        'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'criterion': ['squared_error', 'friedman_mse'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "Orthogonal Matching Pursuit": {
        'n_nonzero_coefs': [None, 5, 10, 20, 50],
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },

    "Passive Aggressive Regressor": {
        'C': [0.1, 1, 10],
        'fit_intercept': [True, False],
        'max_iter': [100, 200, 300],
        'tol': [1e-3, 1e-4, 1e-5]
    },

}
