import os
from sys import platform

home = os.environ['HOME']

# Data Location
if platform == "win32":
    data_dir = 'D:\Data\Froedtert\opioid'
else:
    home = os.environ['HOME']
    data_dir = os.path.join(home, 'Data', 'Froedtert')

# Different data sources
COVID_DATA_NAME_1 = 'covid-May2021'
COVID_DATA_NAME_2 = 'COVID+June2021-Jan2023'

# COVID_SUD_DATA_NAME = 'SUD_SM_COVID_Jan2020_Jan2023'
DATA_CHECK = 'data_check'

COVID_DIR_1 = os.path.join(data_dir, COVID_DATA_NAME_1)
COVID_DIR_2 = os.path.join(data_dir, COVID_DATA_NAME_2)
# COVID_SUD_DIR = os.path.join(data_dir, COVID_SUD_DATA_NAME)

# Processed Data Location
MERGED = 'Merged Data'
COVID_MERGED_DATA_PATH = 'Merged Data/COVID-2020-2023'
PROCESSED_DATA_PATH = 'Merged Data/Processed Data'
PROCESSED_ML_DATA_PATH = 'Merged Data/Data for ML'

RESULTS = 'Results'
ML_RESULTS = 'Results/ML Results'
ML_MODELS = 'Results/Models'
STAT_RESULTS = 'Results/Stats Results'

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

if not os.path.exists(ML_RESULTS):
    os.mkdir(ML_RESULTS)

if not os.path.exists(ML_MODELS):
    os.mkdir(ML_MODELS)

# Define the ICD codes for each SUD
SUD_ICD_CODE = {
    'Alcohol Use Disorder': ['F10'],
    'Cannabis Use Disorder': ['F12'],
    'Cocaine Use Disorder': ['F14'],
    'Opioid Use Disorder': ['F11', 'T40'],
    'Tobacco Use Disorder': ['F17']
}

SUD_PATH = {}
# Create SUD path directory
for key, value in SUD_ICD_CODE.items():
    path = os.path.join(PROCESSED_DATA_PATH, key)
    if os.path.exists(path) == False:
        os.mkdir(path)
    SUD_PATH[key] = path

# Define the Immunization name
immunization_name = ['PFIZER', 'MODERNA', 'JANSSEN', 'COVID']

# %% Define the columns for each table

enc_type_list = ['IP', 'ED', 'UN', 'OS', 'AV', 'EI', 'IS']
admission_enc_type_list = ['IP', 'ED', 'UN', 'OS']

diagnosis_columns = ['patient_num', 'encounter_num', 'enc_type', 'dx_date_shifted', 'dx_code']

encounters_columns = ['patient_num', 'encounter_num', 'enc_type', 'age_at_visit_years', 'discharge_status_c',
                      'payor_type_primary_name', 'admit_date_shifted', 'discharge_date_shifted', 'length_of_stay_days']

diagnostic_results_columns = ['patient_num', 'component_name', 'abnormal_ind', 'i2b2_date_shifted',
                              'result_num', 'result_modifier']

medication_orders_columns = ['patient_num', 'order_date_shifted', 'order_status', 'order_med_id',
                             'medication_name', 'pharm_class', 'ingredient_rxcui', 'specific_rxcui']

demographics_columns = ['patient_num', 'sex', 'marital_status', 'employment_status', 'race',
                        'ethnicity', 'zip3']

social_history_lifestyle_columns = ['patient_num', 'contact_date_shifted', 'edu_level_name', 'alcohol_use_name',
                                    'ill_drug_user_name', 'tobacco_user_name']

problem_list_columns = ['patient_num', 'encounter_num', 'dx_code', 'epic_dx_description', 'date_for_filter_shifted'
    , 'noted_date_shifted', 'resolved_date_shifted', 'date_of_entry_shifted', 'chronic_yn'
    , 'principal_pl_yn', 'hospital_pl_yn']

immunization_columns = ['patient_num', 'encounter_num', 'immune_date_shifted', 'immunization_name', 'immnztn_status_name']

vitals_columns = ['patient_num', 'encounter_num', 'bmi']

table_column_dict = {'diagnosis': diagnosis_columns,
                     'encounters': encounters_columns,
                     'diagnostic_results': diagnostic_results_columns,
                     'medication_orders': medication_orders_columns,
                     'demographics': demographics_columns,
                     'social_history_lifestyle': social_history_lifestyle_columns,
                     'problem_list': problem_list_columns,
                     'immunization': immunization_columns,
                     'vitals': vitals_columns
                     }

# Define the features
INFO = {'Age': [
    '<45',
    '45-64',
    '>=65'],

    'Sex': [
        'Male',
        'Female'],

    'Fully Vaccinated': ['Yes', 'No'],

    'Race': [
        'White or Caucasian',
        'Black or African American',
        'American Indian or Alaska Native',
        'Asian',
        'Other/not specified races'],

    'Ethnicity': [
        'Non Hispanic',
        'Hispanic',
        'Other/not specified ethnicities'],

    'Primary Pay': [
        'Private',
        'Medicare',
        'Government Plan',
        'Medicaid',
        'Other Methods']}

INFO_OUD = {'Age': ['<45', '45-64', '>=65'],

            'Sex': [
                'Male',
                'Female'],

            'Fully Vaccinated': ['Yes', 'No'],

            'Race': [
                'White or Caucasian',
                'Black or African American',
                'American Indian or Alaska Native',
                'Asian',
                'Other/not specified races'],

            'Primary Pay': [
                'Private',
                'Medicare',
                'Government Plan',
                'Medicaid',
                'Other Methods']}

INFO_VAC = {'Age': ['<45', '45-64', '>=65'],

            'Sex': [
                'Male',
                'Female'],

            'Race': [
                'White or Caucasian',
                'Black or African American',
                'American Indian or Alaska Native',
                'Asian',
                'Other/not specified races'],

            'Primary Pay': [
                'Private',
                'Medicare',
                'Government Plan',
                'Medicaid',
                'Other Methods']

            }

DESIRED_FEATURES = ['Total', 'Male',
                    'Female', 'White or Caucasian',
                    'Black or African American',
                    'American Indian or Alaska Native',
                    'Asian',
                    'Other/not specified races', 'Non Hispanic',
                    'Hispanic',
                    'Other/not specified ethnicities', 'Private',
                    'Medicare',
                    'Government Plan',
                    'Medicaid',
                    'Other Methods']

# Adjust the order and map to more readable names
ORDERED_VAR_MAP = [
    ('OUD', 'OUD'),
    ('vaccinated', 'Vaccination'),
    (None, 'Age'),
    ('C(Age, Treatment(\'<45\'))[T.45-64]', 'Age 45-64'),
    ('C(Age, Treatment(\'<45\'))[T.>=65]', 'Age >=65'),
    ('Sex[T.Male]', 'Male'),
    (None, 'Race'),
    ('C(Race, Treatment(\'White or Caucasian\'))[T.Black or African American]', 'Black or African American'),
    ('C(Race, Treatment(\'White or Caucasian\'))[T.American Indian or Alaska Native]',
     'American Indian or Alaska Native'),
    ('C(Race, Treatment(\'White or Caucasian\'))[T.Asian]', 'Asian'),
    ('C(Race, Treatment(\'White or Caucasian\'))[T.Other/not specified races]', 'Other/not specified races'),
    (None, 'Primary Pay'),
    ('C(Q(\'Primary Pay\'), Treatment(\'Private\'))[T.Medicare]', 'Medicare'),
    ('C(Q(\'Primary Pay\'), Treatment(\'Private\'))[T.Government Plan]', 'Government Plan'),
    ('C(Q(\'Primary Pay\'), Treatment(\'Private\'))[T.Medicaid]', 'Medicaid'),
    ('C(Q(\'Primary Pay\'), Treatment(\'Private\'))[T.Other Methods]', 'Other Methods')
]
