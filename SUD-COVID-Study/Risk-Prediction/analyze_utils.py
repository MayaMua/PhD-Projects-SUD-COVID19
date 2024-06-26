import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import re
import pickle
from processing_utils import *
from utils import *
from config import *

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, make_scorer, \
    roc_auc_score, roc_curve, auc, confusion_matrix, PrecisionRecallDisplay, precision_recall_curve

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from xgboost import XGBClassifier



def get_ml_save_path(target):
    model_path = os.path.join(ML_MODELS, target)
    result_path = os.path.join(ML_RESULTS, target)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return model_path, result_path

fine_tune_grids = {
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
        'sampling_strategy': ['auto', 'majority', 'not minority', 'all'] + [float(x) for x in np.linspace(0.1, 1.0, 10)],
        'replacement': [True, False],
    },

    'XGBoost': {
        'eta': [0.05, 0.1, 0.25, 0.5],
        'gamma': [0, 0.5, 2, 5, 10],
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'max_depth': [int(x) for x in np.linspace(0, 50, num = 10)],
        # 'min_child_weight': [0, 1, 5, 10, 25],
        'subsample': [0.2, 0.5, 0.8, 1]
    },
    'KNeighbors': {
        'n_neighbors': list(range(1, 21))
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


def get_data(target):
    if target == 'los':
        X_path = os.path.join(PROCESSED_ML_DATA_PATH, 'los_X.csv')
        y_path = os.path.join(PROCESSED_ML_DATA_PATH, 'los_y.csv')
        columns = load_pd_pickle('selected_features_Long Stay_0.05')

    elif target == 'readmission':
        X_path = os.path.join(PROCESSED_ML_DATA_PATH, 'read_X.csv')
        y_path = os.path.join(PROCESSED_ML_DATA_PATH, 'read_y.csv')
        columns = load_pd_pickle('selected_features_Readmission_0.05')

    elif target == 'death':
        X_path = os.path.join(PROCESSED_ML_DATA_PATH, 'death_X.csv')
        y_path = os.path.join(PROCESSED_ML_DATA_PATH, 'death_y.csv')
        columns = load_pd_pickle('selected_features_Death_0.05')

    X = pd.read_csv(X_path)
    X = X[columns]
    y = pd.read_csv(y_path)
    print(y.value_counts())
    return X, y


def get_train_test_split(target, test_size=0.2):
    X, y = get_data(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = X_train.reset_index().drop(['index'], axis=1)
    X_test = X_test.reset_index().drop(['index'], axis=1)

    y_train = y_train.reset_index().drop(['index'], axis=1)
    y_test = y_test.reset_index().drop(['index'], axis=1)
    # print the length of the training and test sets
    print("Length of Training Set: ", len(X_train))
    print("Length of Test Set: ", len(X_test))
    return X_train, X_test, y_train, y_test


# Define a list of models
def get_models():
    return [
        ('RandomForest', RandomForestClassifier(random_state=0)),
        ('SVC', SVC(kernel='linear', probability=True)),
        ('XGBoost', XGBClassifier()),
        ('KNeighbors', KNeighborsClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('BalancedRandomForest', BalancedRandomForestClassifier(random_state=0)),
        ('EasyEnsemble', EasyEnsembleClassifier(random_state=0))
    ]


# Define a list of data sampling techniques
def get_sampling_techniques():
    return [
        ('Original', None),  # No oversampling or undersampling
        ('SMOTE', SMOTE(sampling_strategy='auto', random_state=42)),
        ('SMOTEENN', SMOTEENN(enn=None, sampling_strategy='auto', random_state=42))
    ]


# Define a function to train and evaluate models
def train_and_evaluate_models(X_train, y_train, n_splits, sampling_techniques):
    models = get_models()
    results = {}

    for model_name, model in models:

        results[model_name] = {}

        for technique_name, technique in sampling_techniques:

            # Skip 'BalancedRandomForest' and 'EasyEnsemble' models after training with original data
            if model_name in ['BalancedRandomForest', 'EasyEnsemble'] and technique_name != 'Original':
                continue

            results[model_name][technique_name] = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
                'ROC AUC': [],
                'PR AUC': []
            }

            print(f"Training {model_name} with {technique_name}...")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            for i, (t, v) in enumerate(cv.split(X_train, y_train)):
                X_train_fold, y_train_fold = X_train.iloc[t], y_train.iloc[t]
                X_valid_fold, y_valid_fold = X_train.iloc[v], y_train.iloc[v]

                if technique is not None and technique_name != 'Original':
                    X_train_fold, y_train_fold = technique.fit_resample(X_train_fold, y_train_fold)

                clf_model = model
                clf_model.fit(X_train_fold, y_train_fold.values.ravel())

                # save_model(clf_model, save_path=model_path)
                y_pred = clf_model.predict(X_valid_fold)

                precision, recall, _ = precision_recall_curve(y_valid_fold, clf_model.predict_proba(X_valid_fold)[:, 1])

                results[model_name][technique_name]['Accuracy'].append(accuracy_score(y_valid_fold, y_pred))
                results[model_name][technique_name]['Precision'].append(precision_score(y_valid_fold, y_pred))
                results[model_name][technique_name]['Recall'].append(recall_score(y_valid_fold, y_pred))
                results[model_name][technique_name]['F1'].append(f1_score(y_valid_fold, y_pred))
                results[model_name][technique_name]['ROC AUC'].append(
                    roc_auc_score(y_valid_fold, clf_model.predict_proba(X_valid_fold)[:, 1]))
                results[model_name][technique_name]['PR AUC'].append(auc(recall, precision))

    print("Training Done!")
    return results


def format_evaluation_results(evaluation_results, target, result_path):
    cols = ['Sampling Technique', 'Model', 'Accuracy Mean', 'Accuracy SD',
            'Precision Mean', 'Precision SD',
            'Recall Mean', 'Recall SD',
            'F1 Mean', 'F1 SD',
            'ROC AUC Mean', 'ROC AUC SD',
            'PR AUC Mean', 'PR AUC SD'
            ]
    data = []
    for model_name, model_results in evaluation_results.items():
        for technique_name, metrics in model_results.items():
            print(model_name, technique_name, metrics)
            row = {'Model': model_name, 'Sampling Technique': technique_name}
            for metric_name, values in metrics.items():
                mean = np.mean(values)
                std = np.std(values)
                row[f'{metric_name} Mean'] = round(mean, 4)
                row[f'{metric_name} SD'] = round(std, 4)
            data.append(row)

    df = pd.DataFrame(data, columns=cols)

    result_df = df.groupby('Sampling Technique', group_keys=True).apply(lambda x: x).reset_index(drop=True)
    fn = f'{target} training results.csv'
    result_df.to_csv(os.path.join(result_path, fn), index=False)
    print(result_df)


def fine_tuning(target, model_name, model, param_grid, X_train, y_train,
                X_test, y_test, metrics, predictions, fine_tuned_models, model_path, samping_technique=None):

    # Check if the model has already been trained
    if os.path.exists(os.path.join(model_path, target + '_best_' + model_name + '.model')):
        print(f"{model_name} has already been trained!")
        best_model = load_model('los_best_' + model_name, save_path=model_path)

    else:
        if samping_technique is not None:
            X_train, y_train = samping_technique.fit_resample(X_train, y_train)

        print(f"Training {model_name}...")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring='recall',
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train.values.ravel())

        best_model = search.best_estimator_

        print("Best Parameters: ", search.best_params_)
        print("Best Score: ", search.best_score_)

        save_model(best_model, 'los_best_' + model_name, save_path=model_path)
        print(f"Best model for {model_name} saved!")

    # best_model.fit(X_train, y_train.values.ravel())

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Make predictions with probability scores
    probabilities = best_model.predict_proba(X_test)
    precision, recall, _ = precision_recall_curve(y_test, probabilities[:, 1])

    metrics[model_name] = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'Precision': precision_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, probabilities[:, 1]),
        'PR AUC': auc(recall, precision),
    }

    predictions[model_name] = y_pred
    fine_tuned_models[model_name] = best_model

def compute_confusion_matrix(y_test, predictions, normalize):

    if normalize is None:
        cm = confusion_matrix(y_test, predictions)
    else:
        cm = confusion_matrix(y_test, predictions, normalize='true')
        decimal_places = 2
        cm= np.round(cm, decimal_places)
    return cm


def plot_confusion_matrix(y_test, predictions, labels, file_name, save_path, normalize=None):

    cm = compute_confusion_matrix(y_test, predictions, normalize=normalize)
    ax = plt.subplot()
    sns.set(font_scale=1.5) # Adjust to fit

    print(cm)

    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g");

    # Labels, title and ticks
    label_font = {'size':'10'} # Adjust to fit
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)
    title_font = {'size': '10'}  # Adjust to fit

    if normalize is None:
        ax.set_title('Confusion Matrix (n)', fontdict=title_font)
    else:
        ax.set_title('Confusion Matrix (%)', fontdict=title_font)

    ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust to fit

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Save the figure
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')

    plt.show()

def get_reall_precision(model, X_test, y_test):
    # y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    return precision, recall

def plot_precision_recall_curve(*models, X_test, y_test, file_name, save_path):
    precision = []
    recall = []
    plt.figure(figsize=(8, 6))
    for model in models:
        p, r = get_reall_precision(model, X_test, y_test)
        precision.append(p)
        recall.append(r)

        plt.plot(r, p, label=type(model).__name__)

    baseline = len(y_test[y_test.values.ravel() == 1]) / len(y_test)
    plt.plot([0, 1], [baseline, baseline], label='Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    legend = plt.legend(loc="lower left", fontsize="small")
    legend.set_bbox_to_anchor((0.0, 0.05))
    plt.savefig(os.path.join(save_path, file_name))  # Save the plot to a file
    plt.show()  # Show the plot (optional)


def plot_feature_importance(fi, features, top, file_name, common_features=None, save_path=None):
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': fi})

    feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
    feature_importance_top = feature_importance.head(top)
    feature_importance_top.loc[:, 'Feature'] = feature_importance_top['Feature'].str.replace('lab_', '').replace('med_', '').replace('_', ' ')

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_top, color='steelblue')
    # Adjust the title font size
    # plt.title(f'The importance of top {top} features')
    # Adjust the label font size
    if common_features:
        for tick in plt.gca().get_yticklabels():
            if tick.get_text() in common_features:
                tick.set_color('red')

    plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust to fit
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
