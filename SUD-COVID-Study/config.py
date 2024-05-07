import os
from sys import platform

home = os.environ['HOME']
# abs_path = os.path.dirname(os.path.abspath(__file__))

if platform == "win32":
    data_dir = 'D:\Data\Froedtert\opioid'
else:
    home = os.environ['HOME']
    data_dir = os.path.join(home, 'Data', 'Froedtert')

COVID_DATA_NAME_1 = 'covid-May2021'
COVID_DATA_NAME_2 = 'COVID+June2021-Jan2023'
COVID_SUD_DATA_NAME = 'SUD_SM_COVID_Jan2020_Jan2023'

COVID_DIR_1 = os.path.join(data_dir, COVID_DATA_NAME_1)
COVID_DIR_2 = os.path.join(data_dir, COVID_DATA_NAME_2)
COVID_SUD_DIR = os.path.join(data_dir, COVID_SUD_DATA_NAME)


# Processed Data Location
# MERGED = '../Merged Data'
# COVID_MERGED_DATA_PATH = '../Merged Data/COVID-2020-2023'
# PROCESSED_DATA_PATH = '../Merged Data/Processed Data'
# PROCESSED_ML_DATA_PATH = '../Merged Data/Data for ML'
#
# RESULTS = '../Results'
# STAT_RESULTS = '../Results/STAT Results'
# ML_RESULTS = '../Results/ML Results'
# ML_MODELS = '../Results/Models'
# STAT_RESULTS = '../Results/Stats Results'


MERGED = 'Merged Data'
COVID_MERGED_DATA_PATH = 'Merged Data/COVID-2020-2023'
PROCESSED_DATA_PATH = 'Merged Data/Processed Data'
PROCESSED_ML_DATA_PATH = 'Merged Data/Data for ML'

RESULTS = 'Results'
STAT_RESULTS = 'Results/STAT Results'
ML_RESULTS = 'Results/ML Results'
ML_MODELS = 'Results/Models'
STAT_RESULTS = 'Results/Stats Results'


def main():
    # Create directories if not exist
    if not os.path.exists(MERGED):
        os.mkdir(MERGED)

    if not os.path.exists(COVID_MERGED_DATA_PATH):
        os.mkdir(COVID_MERGED_DATA_PATH)

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.mkdir(PROCESSED_DATA_PATH)

    if not os.path.exists(PROCESSED_ML_DATA_PATH):
        os.mkdir(PROCESSED_ML_DATA_PATH)

    if not os.path.exists(RESULTS):
        os.mkdir(RESULTS)

    if not os.path.exists(STAT_RESULTS):
        os.mkdir(STAT_RESULTS)

    if not os.path.exists(ML_RESULTS):
        os.mkdir(ML_RESULTS)

    if not os.path.exists(ML_MODELS):
        os.mkdir(ML_MODELS)
    
if __name__ == "__main__":
    main()