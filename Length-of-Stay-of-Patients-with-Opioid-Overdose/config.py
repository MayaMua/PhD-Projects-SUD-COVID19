
import os

work_space = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = os.path.join(work_space, "Data")
save_data_path = os.path.join(work_space, "Cleaned Data")

# keep important columns to make a subset of each table 
diagnosis_columns = ['patient_num', 'encounter_num', 'dx_date_shifted']
encounter_columns = ['patient_num', 'encounter_num', 'age_at_visit_years', 'discharge_status_c', 'department_name',
                    'payer_type_primary_name', 'admit_date_shifted',
                    'discharge_date_shifted', 'length_of_stay_days']
diagnosis_result_columns = ['patient_num', 'component_name', 'abnormal_ind', 'i2b2_date_shifted',
                            'result_num',  'result_modifier']
medication_orders_columns = ['patient_num', 'order_date_shifted', 'order_status', 'order_med_id',
                            'medication_name', 'pharm_class', 'ingredient_rxcui','specific_rxcui']
patient_demographics_columns = ['patient_num', 'sex', 'marital_status', 'employment_status', 'race',
                                'ethnicity', 'zip3', 'language']
social_history_lifestyle_columns = ['patient_num', 'contact_date_shifted', 'edu_level_name','alcohol_use_name',
                                    'ill_drug_user_name', 'tobacco_user_name']
problem_list_columns = ['patient_num','encounter_num','dx_code','epic_dx_description','date_for_filter_shifted'
                        ,'noted_date_shifted','resolved_date_shifted','date_of_entry_shifted','chronic_yn'
                        ,'principal_pl_yn','hospital_pl_yn']
