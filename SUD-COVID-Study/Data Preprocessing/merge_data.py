import os
import datetime

import sys
sys.path.append('.')

from utils import *
from processing_utils import *
from config import *
from data_config import *

def merge_databases(table_interested):

    # Merge 2 databases: covid-May2021 and COVID-June2021-Jan2023

    for table in table_interested:
        target_path = os.path.join(COVID_MERGED_DATA_PATH, table + '.csv')
        if os.path.exists(target_path):
            print(f'{table}.csv already exists, skip...')
            print('=' * 20)
        else:
            if table == 'demographics':
                table_path_1 = os.path.join(COVID_DIR_1, 'patient_' + table + '.csv')
                table_path_2 = os.path.join(COVID_DIR_2, table + '.csv')
            else:
                table_path_1 = os.path.join(COVID_DIR_1, table + '.csv')
                table_path_2 = os.path.join(COVID_DIR_2, table + '.csv')

            print(f'Processing {table}.csv ...')
            table_1 = pd.read_csv(table_path_1, low_memory=False)
            table_2 = pd.read_csv(table_path_2, low_memory=False)

            table_merge = pd.concat([table_1, table_2], axis=0)
            table_merge = table_merge[table_column_dict[table]]

            table_merge.to_csv(os.path.join(COVID_MERGED_DATA_PATH, table + '.csv'), index=False)

            print(f'Processed {table}.csv ...')
            print('=' * 20)

    print('Merging data completed ...')


def merge_diganosis_encounter():
    
    print('Loading diagnosis...')
    diagnosis_path = os.path.join(COVID_MERGED_DATA_PATH, 'diagnosis.csv')
    diagnosis = load_data(diagnosis_path)
    diagnosis = diagnosis[diagnosis['enc_type'].isin(admission_enc_type_list)]
    diag_filtered = diagnosis[diagnosis_columns].astype(str)
    unique_patients = diag_filtered['patient_num'].unique()
    # This is also including the patients who have no diagnosis code of covid
    print('There are {} unique patients in the COVID-19 data set (2020-2023).'.format(len(unique_patients)))
    print()
    
    print('Loading encounters...')
    encounters_path = os.path.join(COVID_MERGED_DATA_PATH, 'encounters.csv')
    encounters = load_data(encounters_path)
    encounters = encounters[encounters['enc_type'].isin(admission_enc_type_list)]
    encounters.drop(columns=['enc_type'], inplace=True)

    print('Merging diagnosis and encounters...')
    diag_enc_merge = diagnosis.merge(encounters, on=['patient_num', 'encounter_num'], how='left')
    # Drop rows where 'discharge_status_c' and 'admit_date_shifted' are null
    diag_enc_merge = diag_enc_merge[diag_enc_merge['admit_date_shifted'].notnull()]
    diag_enc_merge = diag_enc_merge[diag_enc_merge['discharge_status_c'].notnull()]
    diag_enc_merge = diag_enc_merge.sort_values(by=['patient_num', 'dx_date_shifted'], ascending=True)
    
    # Create a new column 'dx_code_list' which contains a list of unique diagnosis codes for each patient
    if not os.path.exists(os.path.join(PROCESSED_DATA_PATH, 'dx_code_list.csv')):
        print(f'Not exists.')
        # group by patient_num and encounter_num amd create a new column named 'dx_code_list'
        # drop duplicates of patient_num and encounter_num
        diag_enc_merge['dx_code_list'] = diag_enc_merge.groupby(['patient_num', 'encounter_num'])['dx_code'].transform(lambda x: ', '.join(x))
        diag_enc_merge['dx_code_list'] = diag_enc_merge['dx_code_list'].apply(lambda x: list(set(x.split(', '))))
        diag_enc_merge = diag_enc_merge.drop_duplicates(subset=['patient_num', 'encounter_num'])
        
        # # save dx_code_list to csv
        save_processed_data(diag_enc_merge[['patient_num', 'encounter_num', 'dx_code_list']], 'dx_code_list', data_path=PROCESSED_DATA_PATH)
    else:
        print('Start merging')
        dx_code_list_df = load_processed_data('dx_code_list')
        diag_enc_merge = diag_enc_merge.drop_duplicates(subset=['patient_num', 'encounter_num'])
        diag_enc_merge = diag_enc_merge.merge(dx_code_list_df, on=['patient_num', 'encounter_num'])
        
    diag_enc_merge = diag_enc_merge.drop(['dx_code'], axis=1)
    
    
    # Change the data type of 'xxx_date_shift' to datetime
    diag_enc_merge['dx_date_shifted'] = pd.to_datetime(diag_enc_merge['dx_date_shifted'])
    diag_enc_merge['admit_date_shifted'] = pd.to_datetime(diag_enc_merge['admit_date_shifted'])
    diag_enc_merge['discharge_date_shifted'] = pd.to_datetime(diag_enc_merge['discharge_date_shifted'])

    # Drop the rows where 'dx_date_shifted' before date 2020-03-01
    diag_enc_merge = diag_enc_merge[diag_enc_merge['dx_date_shifted'] > datetime(2020, 3, 1)]
    # Change the data type of 'xxx_date_shift' to datetime
    diag_enc_merge['dx_date_shifted'] = pd.to_datetime(diag_enc_merge['dx_date_shifted']).dt.date
    diag_enc_merge['admit_date_shifted'] = pd.to_datetime(diag_enc_merge['admit_date_shifted']).dt.date
    diag_enc_merge['discharge_date_shifted'] = pd.to_datetime(diag_enc_merge['discharge_date_shifted']).dt.date
    
    