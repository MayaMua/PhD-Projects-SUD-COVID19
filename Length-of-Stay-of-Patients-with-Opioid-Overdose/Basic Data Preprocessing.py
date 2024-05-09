import os
import pandas as pd
import numpy as np
import datetime
import config
from utils import load_data, save_clean_data, remove_duplicates
from config import data_path

#================================================================================
# Preprocess diagnosis
print("Read Diagnosis")
diagnosis = load_data('diagnosis')
# Extract patients of opioid
l=['IP','ED','UN','OS']
diagnosis = diagnosis[diagnosis['dx_code'].str.contains("T40")]
diagnosis = diagnosis[diagnosis['enc_type'].isin(l)]
diag_filtered = diagnosis[config.diagnosis_columns].astype(str)

# Preprocess encounter
print("Read Encounters")
encounters = load_data('encounters')
l=['IP','ED','UN','OS']
encounters = encounters[encounters['enc_type'].isin(l)]
enc_filtered = encounters[config.encounter_columns].astype(str)

# Drop null and remove duplicates
diag_enc_merge = diag_filtered.merge(enc_filtered,on=['patient_num','encounter_num'], how='left')
diag_enc_merge = diag_enc_merge[diag_enc_merge.admit_date_shifted.notnull()]
diag_enc_dedup = diag_enc_merge.drop_duplicates(subset=['patient_num','encounter_num','admit_date_shifted','discharge_date_shifted'], keep='first')

#
diag_enc_dedup['admit_date_shifted'] = pd.to_datetime(diag_enc_dedup['admit_date_shifted'], format='%Y-%m-%d %H:%M:%S')
diag_enc_dedup['discharge_date_shifted'] = pd.to_datetime(diag_enc_dedup['discharge_date_shifted'], format='%Y-%m-%d %H:%M:%S')
diag_enc_dedup['dx_date_shifted'] = pd.to_datetime(diag_enc_dedup['dx_date_shifted'], format='%Y-%m-%d %H:%M:%S')

# merge discharge_date_shifted and admit_date_shifted to calculate real LOS
diag_final = remove_duplicates(diag_enc_dedup.sort_values(by=['patient_num', 'dx_date_shifted'], ascending=True))
diag_final['admit_date_shifted'] = pd.to_datetime(diag_final['admit_date_shifted'], format='%Y-%m-%d')
diag_final['discharge_date_shifted'] = pd.to_datetime(diag_final['discharge_date_shifted'], format='%Y-%m-%d')
diag_final['admit_date_shifted'] = diag_final['admit_date_shifted'].dt.date
diag_final['discharge_date_shifted'] = diag_final['discharge_date_shifted'].dt.date
diag_final['length_of_stay_days'] = (diag_final['discharge_date_shifted'] - diag_final['admit_date_shifted']).dt.days
diag_final.drop(['discharge_status_c', 'department_name'], axis=1, inplace=True)
# diag_final.dropna(axis = 0, how = 'any', inplace = True)
print(f'Shape of merged Dia and Enc: {diag_final.shape}')
save_clean_data(diag_final, 'overdose_preprocessing')