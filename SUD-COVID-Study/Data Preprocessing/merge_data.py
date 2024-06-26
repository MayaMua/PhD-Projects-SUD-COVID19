import os
import datetime

import sys

sys.path.append('.')

from utils import *
from process_utils import *
from config import *
from data_config import *

processed_data_path = os.path.join('..', PROCESSED_DATA_PATH)
covid_merged_data_path = os.path.join('..', COVID_MERGED_DATA_PATH)


def merge_databases(table_interested):
    # Merge 2 databases: covid-May2021 and COVID-June2021-Jan2023

    for table in table_interested:
        target_path = os.path.join(covid_merged_data_path, table + '.csv')
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

            table_merge.to_csv(os.path.join(covid_merged_data_path, table + '.csv'), index=False)

            print(f'Processed {table}.csv ...')
            print('=' * 20)

    print('Merging data completed ...', end='\n\n')


def merge_diagnosis_encounter():
    print('Loading diagnosis...')
    diagnosis_path = os.path.join(covid_merged_data_path, 'diagnosis.csv')
    diagnosis = load_data(diagnosis_path)
    diagnosis = diagnosis[diagnosis['enc_type'].isin(admission_enc_type_list)]
    diag_filtered = diagnosis[diagnosis_columns].astype(str)
    unique_patients = diag_filtered['patient_num'].unique()
    # This is also including the patients who have no diagnosis code of covid
    print('There are {} unique patients in the COVID-19 data set (2020-2023).'.format(len(unique_patients)))
    print()

    print('Loading encounters...')
    encounters_path = os.path.join(covid_merged_data_path, 'encounters.csv')
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
    print('Creating dx_code_list...')
    dx_code_list_path = os.path.join(processed_data_path, 'dx_code_list.csv')
    if not os.path.exists(dx_code_list_path):
        print(f'Not exists. Creating a csv of dx_code_list...')
        # group by patient_num and encounter_num amd create a new column named 'dx_code_list'
        # drop duplicates of patient_num and encounter_num
        diag_enc_merge['dx_code_list'] = diag_enc_merge.groupby(['patient_num', 'encounter_num']
                                                                )['dx_code'].transform(lambda x: ', '.join(x))
        diag_enc_merge['dx_code_list'] = diag_enc_merge['dx_code_list'].apply(lambda x: list(set(x.split(', '))))
        diag_enc_merge = diag_enc_merge.drop_duplicates(subset=['patient_num', 'encounter_num'])

        # # save dx_code_list to csv
        save_processed_data(diag_enc_merge[['patient_num', 'encounter_num', 'dx_code_list']],
                            'dx_code_list', data_path=processed_data_path)
    else:
        print(f'Existing. Start merging...')
        dx_code_list_df = load_processed_data('dx_code_list', data_path=processed_data_path)
        diag_enc_merge = diag_enc_merge.drop_duplicates(subset=['patient_num', 'encounter_num'])
        diag_enc_merge = diag_enc_merge.merge(dx_code_list_df, on=['patient_num', 'encounter_num'])

    diag_enc_merge = diag_enc_merge.drop(['dx_code'], axis=1)

    # Change the data type of 'xxx_date_shift' to datetime
    diag_enc_merge['dx_date_shifted'] = pd.to_datetime(diag_enc_merge['dx_date_shifted'])
    diag_enc_merge['admit_date_shifted'] = pd.to_datetime(diag_enc_merge['admit_date_shifted'])
    diag_enc_merge['discharge_date_shifted'] = pd.to_datetime(diag_enc_merge['discharge_date_shifted'])

    # Drop the rows where 'dx_date_shifted' before date 2020-03-01 when COVID-19 pandemic was broken
    diag_enc_merge = diag_enc_merge[diag_enc_merge['dx_date_shifted'] > datetime(2020, 3, 1)]

    # Change the data type of 'xxx_date_shift' to datetime
    # save_processed_data(diag_enc_merge, 'diag_enc_merge_with_dx_code',
    #                     data_path=processed_data_path)

    # diag_enc_merge_los = load_processed_data('diag_enc_merge_original', data_path=processed_data_path)

    diag_enc_merge.dropna(subset=['admit_date_shifted'], inplace=True)
    diag_enc_merge.dropna(subset=['discharge_date_shifted'], inplace=True)
    diag_enc_merge.sort_values(["patient_num", "admit_date_shifted"], inplace=True)

    diag_enc_merge['dx_date_shifted'] = pd.to_datetime(diag_enc_merge['dx_date_shifted'])
    diag_enc_merge['admit_date_shifted'] = pd.to_datetime(diag_enc_merge['admit_date_shifted'])
    diag_enc_merge['discharge_date_shifted'] = pd.to_datetime(diag_enc_merge['discharge_date_shifted'])

    print('Counting the length of stay ...')
    diag_enc_merge_recalculate = count_los(diag_enc_merge)
    diag_enc_merge_recalculate = diag_enc_merge_recalculate.dropna(columns=['length_of_stay_days'])
    save_processed_data(diag_enc_merge_recalculate, 'diag_enc_merge_covid_los_recalculate',
                        data_path=processed_data_path)
    print('Data saved successfully!')


def merge_immune_data():

    def set_mfg(x: str):
        if 'PFIZER' in x:
            return 'PFIZER'
        elif 'MODERNA' in x:
            return 'MODERNA'
        elif 'JANSSEN' in x:
            return 'JANSSEN'
        else:
            return 'Others'

    def get_vaccine_fulfillment_date(group):
        group = group.sort_values('immune_date_shifted')
        if any(group['immu_mfg'] == 'JANSSEN'):
            return group.loc[group['immu_mfg'] == 'JANSSEN', 'immune_date_shifted'].iloc[0]
        else:
            if group['immu_mfg'].isin(['MODERNA', 'PFIZER']).sum() >= 2:
                return group.loc[group['immu_mfg'].isin(['MODERNA', 'PFIZER']), 'immune_date_shifted'].iloc[1]
        return pd.NaT

    # Read immunization.csv
    print(f'Processing immune...')
    immunization_path = os.path.join(covid_merged_data_path, 'immunization.csv')
    immu_df = load_data(immunization_path)
    # Select patients with vaccination
    immu_df = immu_df[immu_df['immnztn_status_name'] == 'Given']
    immu_df = immu_df[immu_df['immunization_name'].str.contains('|'.join(immunization_name), na=False, case=False)]
    immu_df = immu_df[immu_df['immune_date_shifted'].notnull()]
    immu_df = immu_df.drop_duplicates(subset=['patient_num', 'encounter_num'], keep='first')
    immu_df = immu_df[immunization_columns].astype(str)
    immu_df['immune_date_shifted'] = pd.to_datetime(immu_df['immune_date_shifted'], format='%Y-%m-%d %H:%M:%S').dt.date
    print(f'Set manufactures')
    immu_df['immu_mfg'] = immu_df['immunization_name'].apply(set_mfg)
    immu_df = immu_df.drop(columns=['immunization_name']).sort_values(by=['patient_num', 'immune_date_shifted'])

    immu_df['immune_date_shifted'] = pd.to_datetime(immu_df['immune_date_shifted'])
    df_sorted = immu_df.sort_values(['patient_num', 'immune_date_shifted'])
    fulfillment_dates = df_sorted.groupby('patient_num').apply(get_vaccine_fulfillment_date).dropna().reset_index()
    fulfillment_dates.columns = ['patient_num', 'date_of_full_vaccination']

    print(f'Merge immnue with diag_encounter_table')
    fulfillment_dates["patient_num"] = fulfillment_dates["patient_num"].astype(str)
    all_patient = load_processed_data('diag_enc_merge_covid_los_recalculate', data_path=processed_data_path)
    all_patient["patient_num"] = all_patient["patient_num"].astype(str)
    fully_vaccinated_patients, not_fully_vaccinated_patients = filter_vaccinated_patients(fulfillment_dates,
                                                                                          all_patient)
    patients_with_vaccine_info = pd.concat([fully_vaccinated_patients, not_fully_vaccinated_patients])
    neglect_columns = ['date_of_full_vaccination']
    all_patients = patients_with_vaccine_info.drop(columns=neglect_columns)
    save_processed_data(all_patients, 'all_patients_with_dxcode_vaccine_info', data_path=processed_data_path)
 
    print(f'All patients with vaccine info!')


def merge_patients_with_vaccine_info():
    pass