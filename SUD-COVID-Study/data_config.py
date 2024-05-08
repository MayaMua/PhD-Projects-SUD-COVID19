    
admission_enc_type_list = ['IP', 'ED', 'UN', 'OS']

diagnosis_columns = ['patient_num', 'encounter_num', 'dx_date_shifted', 'dx_code', 'enc_type']

encounters_columns = ['patient_num', 'encounter_num', 'enc_type', 'age_at_visit_years', 'discharge_status_c',
                    'payor_type_primary_name', 'admit_date_shifted', 'discharge_date_shifted', 'length_of_stay_days']

diagnostic_results_columns = ['patient_num', 'component_name', 'abnormal_ind', 'i2b2_date_shifted',
                            'result_num',  'result_modifier']

medication_orders_columns = ['patient_num', 'order_date_shifted', 'order_status', 'order_med_id',
                            'medication_name', 'pharm_class', 'ingredient_rxcui','specific_rxcui']

patient_demographics_columns = ['patient_num', 'sex', 'marital_status', 'employment_status', 'race',
                                'ethnicity', 'zip3', 'state_name', 'language']

social_history_lifestyle_columns = ['patient_num', 'contact_date_shifted', 'edu_level_name','alcohol_use_name',
                                    'ill_drug_user_name', 'tobacco_user_name']

problem_list_columns = ['patient_num','encounter_num','dx_code','epic_dx_description','date_for_filter_shifted'
                        ,'noted_date_shifted','resolved_date_shifted','date_of_entry_shifted','chronic_yn'
                        ,'principal_pl_yn','hospital_pl_yn']

immunization_columns = ['patient_num', 'encounter_num', 'immune_date_shifted', 'immunization_name', 'immnztn_status_name']

vitals_columns = ['patient_num', 'encounter_num', 'bmi']

table_column_dict = {'diagnosis': diagnosis_columns,
                    'encounters': encounters_columns,
                    'diagnostic_results': diagnostic_results_columns,
                    'medication_orders': medication_orders_columns,
                    'demographics': patient_demographics_columns,
                    'social_history_lifestyle': social_history_lifestyle_columns,
                    'problem_list': problem_list_columns,
                    'immunization': immunization_columns,
                    'vitals': vitals_columns
                    }


# Define the ICD codes for each SUD
SUD_ICD_CODE = {
    'Alcohol Use Disorder': ['F10'],
    'Cannabis Use Disorder': ['F12'],
    'Cocaine Use Disorder': ['F14'],
    'Opioid Use Disorder': ['F11', 'T40'],
    'Tobacco Use Disorder': ['F17']
}


# SUD_PATH = {}
# # Create SUD path directory
# for key, value in SUD_ICD_CODE.items():
#     path = os.path.join(PROCESSED_DATA_PATH, key)
#     if os.path.exists(path) == False:
#         os.mkdir(path)
#     SUD_PATH[key] = path
    

# Define the Immunization name
immunization_name = ['PFIZER', 'MODERNA', 'JANSSEN', 'COVID']
