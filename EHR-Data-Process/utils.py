import pandas as pd
import numpy as np
import pickle
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import *


# Load source data from csv files
def load_data(name):
    return pd.read_csv(name, low_memory=False)





def check_top_icd(df, n):
    top_n_problem = df['dx_code'].value_counts()[:n].index.to_list()
    val_count = df['dx_code'].value_counts()[:n]
    tmp = df.loc[df['dx_code'].isin(top_n_problem)]
    tmp = tmp.groupby('dx_code')['dx_name'].agg(list).reset_index(name='dx_code_description')
    tmp['dx_code_description'] = tmp['dx_code_description'].apply(lambda x: set(x))
    val_count_df = pd.DataFrame(val_count).reset_index().set_axis(['dx_code', 'count'], axis=1, copy=False)
    val_count_df = val_count_df.merge(tmp, how='left')

    return val_count_df




def load_processed_data(name, data_path=PROCESSED_DATA_PATH):
    if name in SUD_ICD_CODE.keys():
        return pd.read_csv(os.path.join(SUD_PATH[name], name + '.csv'), low_memory=False)
    return pd.read_csv(os.path.join(data_path, name + '.csv'), low_memory=False)


def save_processed_data(df, name, data_path=PROCESSED_DATA_PATH):
    # If the name is in SUD_ICD_CODE.values(), then save it in COVID_SUD_DIR
    if name in SUD_ICD_CODE.keys():
        df.to_csv(os.path.join(SUD_PATH[name], name + '.csv'), index=False)
    else:
        df.to_csv(os.path.join(data_path, name + '.csv'), index=False)


def save_analyze_result(df, name, result_path=RESULTS):
    # If the name is in SUD_ICD_CODE.values(), then save it in COVID_SUD_DIR
    df.to_csv(os.path.join(result_path, name + '.csv'), index=False)


def save_fig(fig, name, path=RESULTS):
    fig.savefig(os.path.join(path, name + '.png'), bbox_inches='tight')


###########################################
# Save and load pickle file for ML dataset

def save_as_pickle(f, fn):
    pth = os.path.join('Merged Data', 'Data for ML', fn + '.pk')
    with open(pth, 'wb') as p:
        pickle.dump(f, p)


def load_from_pickle(fn):
    pth = os.path.join('Merged Data', 'Data for ML', fn + '.pk')
    with open(pth, 'rb') as p:
        return pickle.load(p)


def load_pd_pickle(fn):
    pth = os.path.join('Merged Data', 'Data for ML', fn + '.pk')
    return pd.read_pickle(pth)


###########################################
# Save and load ML models
def save_model(model, name, save_path=ML_MODELS):
    # This will save the model itself and its weights
    joblib.dump(model, os.path.join(save_path, name + '.model'))


def load_model(name, save_path=ML_MODELS):
    return joblib.load(os.path.join(save_path, name + '.model'))


###########################################
# Plotting confusion matrix

def compute_confusion_matrix(y_test, predictions, labels, normalize=None):
    if normalize is None:
        cm = confusion_matrix(y_test, predictions, labels=labels)
    else:
        cm = confusion_matrix(y_test, predictions, normalize='true', labels=labels)
        decimal_places = 2
        cm = np.round(cm, decimal_places)
    return cm


def plot_confusion_matrix(y_test, predictions, labels, normalize=None, title='Confusion matrix (n)', fig_path=RESULTS):

    cm = compute_confusion_matrix(y_test, predictions, labels, normalize=normalize)
    ax = plt.subplot()
    sns.set(font_scale=1.5)  # Adjust to fit

    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g")

    # Labels, title and ticks
    label_font = {'size': '10'}  # Adjust to fit
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '10'}  # Adjust to fit
    ax.set_title('Confusion Matrix (n)', fontdict=title_font)

    ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust to fit

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()
    plt.savefig(os.path.join(fig_path, title + '.png'), bbox_inches='tight')


def convert_to_markdown(df, name, md_path=RESULTS):
    df.to_markdown(os.path.join(md_path, name + '.md'), index=False)


# Drop features with small number of values
def drop_features(flattened, n):
    count = 0
    lst = []
    for x in flattened.columns:
        # print(x,"\n",flattened[x].isna().all())
        y = flattened[x].count()
        # print(y)
        if (y <= n):
            # print(x,flattened[x].count())
            lst.append(x)
            count += 1
    print(f"Number of features with less than {n} values: {count}")
    return lst
