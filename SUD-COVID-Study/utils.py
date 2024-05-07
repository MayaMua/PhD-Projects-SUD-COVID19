import pandas as pd
from config import *
from data_config import *

def load_data(name):
    return pd.read_csv(name, low_memory=False)

def save_processed_data(df, name, data_path=PROCESSED_DATA_PATH):
    # If the name is in SUD_ICD_CODE.values(), then save it in COVID_SUD_DIR
    if name in SUD_ICD_CODE.keys():
        df.to_csv(os.path.join(PROCESSED_DATA_PATH[name], name + '.csv'), index=False)
    else:
        df.to_csv(os.path.join(data_path, name + '.csv'), index=False)

def load_processed_data(name, data_path=PROCESSED_DATA_PATH):
    if name in SUD_ICD_CODE.keys():
        return pd.read_csv(os.path.join(PROCESSED_DATA_PATH[name], name + '.csv'), low_memory=False)
    return pd.read_csv(os.path.join(data_path, name + '.csv'), low_memory=False)


