import numpy as np

fine_fune_grids = {
    'RandomForest': {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_features': ['sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },

    'BalancedRandomForest': {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'sampling_strategy': ['auto', 'majority', 'not minority', 'all'] + [float(x) for x in np.linspace(0.1, 1.0, 10)]
    },

    'EasyEnsemble': {
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'sampling_strategy': ['auto', 'majority', 'not minority', 'all'] + [float(x) for x in
                                                                            np.linspace(0.1, 1.0, 10)],
        'replacement': [True, False],
    },

    'XGBoost': {
        'eta': [0.05, 0.1, 0.25, 0.5],
        'gamma': [0, 0.5, 2, 5, 10],
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'max_depth': [int(x) for x in np.linspace(0, 50, num=10)],
        # 'min_child_weight': [0, 1, 5, 10, 25],
        'subsample': [0.2, 0.5, 0.8, 1]
    },

    'SVC': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
        'class_weight': [None, 'balanced']
    },

    'LogisticRegression': {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'class_weight': [None, 'balanced']
    },

}