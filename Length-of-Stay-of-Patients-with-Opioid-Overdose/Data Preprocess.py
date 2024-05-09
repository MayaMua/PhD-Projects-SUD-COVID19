import os
import pandas as pd
import numpy as np
import datetime
import config
from utils import load_data, save_clean_data, remove_duplicates, read_preprocess_file, drop_features
from config import data_path
diag = read_preprocess_file('overdose_preprocessing')

# #================================================================================
# #Preprocess diagnostic results
def process_diag_result():
    print('start to process diagnostic results')
    diagnostic_results = load_data('diagnostic_results')
    # # drop PERFORMED BY in component_name
    diagnostic_results.drop(diagnostic_results.loc[diagnostic_results['component_name']=='PERFORMED BY'].index, inplace=True)
    # # Only keep quantitive results
    diagnostic_results = diagnostic_results[diagnostic_results.result_num.notnull()]
    diagnostic_results = diagnostic_results[config.diagnosis_result_columns]
    diag_results_merge = diag.merge(diagnostic_results, how='left', on=['patient_num'])


    # Select patients with valid dates 
    diag_results_merge['admit_date_shifted'] = pd.to_datetime(diag_results_merge['admit_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    diag_results_merge['discharge_date_shifted'] = pd.to_datetime(diag_results_merge['discharge_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    diag_results_merge['i2b2_date_shifted'] = pd.to_datetime(diag_results_merge['i2b2_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    diag_results_admit = diag_results_merge[(diag_results_merge['admit_date_shifted'] <= diag_results_merge['i2b2_date_shifted']) &  
                                            (diag_results_merge['i2b2_date_shifted'] <= diag_results_merge['admit_date_shifted'] + datetime.timedelta(days=10))]

    diag_results_dedup = diag_results_admit.sort_values(by=['patient_num','admit_date_shifted','i2b2_date_shifted'
                                                            ]).drop_duplicates(subset=['patient_num','encounter_num','component_name'
                                                                                        ], keep='first')
    # Create a pivot table based on component_name                                 
    diag_results_subset = diag_results_dedup[['patient_num','encounter_num','component_name','result_num']]
    df_pivot = diag_results_subset.pivot_table(index=['patient_num','encounter_num'],
                        columns='component_name', aggfunc='first').reset_index()
    flattened = pd.DataFrame(df_pivot.to_records())
    flattened.columns = [hdr.replace(", ''","").replace("('", "").replace("')", "").replace("result_num', '","") \
                        for hdr in flattened.columns]

    #Drop features with small number of values
    lst = drop_features(flattened, 5)
    flattened.drop(columns=lst, inplace=True)
    flattened.drop(['index'], axis=1, inplace=True)
    flattened.fillna(0, inplace=True)                                               
    save_clean_data(flattened, 'Overdose_LOS_diag')
    print(f'The shape of the final table is {flattened.shape}')
    print('Finished')

# #Preprocess medication orders
def process_med_order():
    print('start to process medication orders')
    medication_orders = load_data('medication_orders')
    medication_orders = medication_orders[config.medication_orders_columns]
    diag_med = diag.merge(medication_orders, on='patient_num', how='left')

    # Select patients with valid dates 
    diag_med['admit_date_shifted'] = pd.to_datetime(diag_med['admit_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    diag_med['discharge_date_shifted'] = pd.to_datetime(diag_med['discharge_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    diag_med['order_date_shifted'] = pd.to_datetime(diag_med['order_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    med_results_history = diag_med[(diag_med['admit_date_shifted'] > diag_med['order_date_shifted']) &  (diag_med['order_date_shifted']>=diag_med['admit_date_shifted']-datetime.timedelta(days=3*365))]

    #Filter discontinued, suspend order statuses
    medications_completed = med_results_history[med_results_history.order_status.isin([x for x in ['Discontinued', 'Suspend', 'Sent', 'Completed', 'Verified','Dispensed'
                                                                                            ] if x not in ['Discontinued','Suspend']])]
    med_orders_dedupe = medications_completed.sort_values(by=['patient_num','encounter_num','order_med_id','pharm_class','medication_name','ingredient_rxcui','specific_rxcui'
                                                            ]).drop_duplicates(['patient_num','encounter_num','order_med_id','pharm_class','medication_name','ingredient_rxcui','specific_rxcui'
                                                                                ],keep='first')
    med_orders_dedupe2 = med_orders_dedupe.sort_values(by=['patient_num','encounter_num','order_date_shifted','medication_name'
                                                            ]).drop_duplicates(['patient_num','encounter_num','order_date_shifted','medication_name'
                                                                                ],keep='first')
    # Create a pivot table based on component_name                                 
    med_pharm_grp = med_orders_dedupe2.groupby(['patient_num','encounter_num','pharm_class']).count()
    med_pharm_reset = med_pharm_grp[['order_med_id']]
    med_pharm_reset.reset_index(inplace=True)
    df_pivot = med_pharm_reset.pivot_table(index=['patient_num','encounter_num'],
                        columns='pharm_class', aggfunc='first').reset_index()
    flattened = pd.DataFrame(df_pivot.to_records())
    flattened.columns = [hdr.replace(", ''","").replace("('", "").replace("')", "").replace("order_med_id', '","") \
                        for hdr in flattened.columns]
    flattened.drop(['index'], axis=1, inplace=True)
    flattened.fillna(0, inplace=True)
    save_clean_data(flattened, 'Overdose_LOS_med')
    print(f'The shape of the final table is {flattened.shape}')
    print('Finished')

# process demographics and lifestyle()
def process_demo_ls():
    print('start to process demographics and lifestyle')

    patient_demographics  = load_data('patient_demographics')
    patient_demographics = patient_demographics[config.patient_demographics_columns]
    demo_merge = diag.merge(patient_demographics, how='left', on=['patient_num'])
    demo_only = demo_merge[['patient_num','admit_date_shifted','sex','marital_status','employment_status','race','ethnicity','zip3']]
    social_history_lifestyle  = load_data('social_history_lifestyle')
    sh_filtered = social_history_lifestyle[config.social_history_lifestyle_columns]

    sh_filtered['contact_date_shifted'] = pd.to_datetime(sh_filtered['contact_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    demo_only['admit_date_shifted'] = pd.to_datetime(demo_only['admit_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    # sh_filtered['year']=sh_filtered['contact_date_shifted'].dt.year
    sh_merge = demo_only.merge(sh_filtered ,how='left', on=['patient_num'])
    sh_merge_filt = sh_merge[(sh_merge['admit_date_shifted']+datetime.timedelta(days=30) > sh_merge['contact_date_shifted'])]
    sh_merge_dedup = sh_merge_filt.sort_values(by=['patient_num','contact_date_shifted']).drop_duplicates(['patient_num'],keep='last')
    
#     map_ = {'Never':'No', 'Not Currently':'Yes', 'Passive':'No'}
#     sh_merge_dedup.loc[:,'ill_drug_user_name'] = sh_merge_dedup.loc[:,'ill_drug_user_name'].map(map_).fillna('Not Asked')
#     sh_merge_dedup.loc[:,'alcohol_use_name'] = sh_merge_dedup.loc[:,'alcohol_use_name'].map(map_).fillna('Not Asked')
#     map_ = {'Never':'No', 'Quit':'Yes', 'Passive':'No'}
#     sh_merge_dedup.loc[:,'tobacco_user_name'] = sh_merge_dedup.loc[:,'tobacco_user_name'].map(map_).fillna('Not Asked')

    save_clean_data(sh_merge_dedup, 'Overdose_LOS_Demo_Life')
    print(f'The shape of the final table is {sh_merge_dedup.shape}')
    print('Finished')
    

def process_problem_list():
    problem_list  = load_data('problem_list')
    
    pl_dedup = problem_list.sort_values(by=['patient_num','encounter_num','dx_code','epic_dx_description','date_for_filter_shifted'
                                ,'noted_date_shifted','resolved_date_shifted','date_of_entry_shifted','chronic_yn'
                                ,'principal_pl_yn','hospital_pl_yn']).drop_duplicates(subset=['patient_num','encounter_num','dx_code','epic_dx_description','date_for_filter_shifted'
                                                                                            ,'noted_date_shifted','resolved_date_shifted','date_of_entry_shifted','chronic_yn'
                                                                                            ,'principal_pl_yn','hospital_pl_yn'], keep='first') 
    
    pl_dedup_c = pl_dedup.sort_values(by=['patient_num','encounter_num','dx_code','epic_dx_description','chronic_yn']).drop_duplicates(subset=['patient_num','encounter_num','dx_code','epic_dx_id'],keep='last')
    pl_dedup_p = pl_dedup.sort_values(by=['patient_num','encounter_num','dx_code','epic_dx_description','principal_pl_yn']).drop_duplicates(subset=['patient_num','encounter_num','dx_code','epic_dx_id'],keep='last')
    pl_dedup_d = pl_dedup.sort_values(by=['patient_num','encounter_num','dx_code','epic_dx_description','noted_date_shifted']).drop_duplicates(subset=['patient_num','encounter_num','dx_code','epic_dx_id'],keep='last')

    pl_dedup_merge = pl_dedup_d.merge(pl_dedup_c, how='left', on=['patient_num','encounter_num','dx_code','epic_dx_description'])
    pl_dedup_merge = pl_dedup_merge.merge(pl_dedup_p, how='left', on=['patient_num','encounter_num','dx_code','epic_dx_description'])
    
    pl_diag_merge = diag.merge(pl_dedup_merge, how='left', on=['patient_num'])
    
    pl_diag_merge['admit_date_shifted'] = pd.to_datetime(pl_diag_merge['admit_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    pl_diag_merge['discharge_date_shifted'] = pd.to_datetime(pl_diag_merge['discharge_date_shifted'],format='%Y-%m-%d %H:%M:%S')
    pl_diag_merge['date_for_filter_shifted'] = pd.to_datetime(pl_diag_merge['date_for_filter_shifted'],format='%Y-%m-%d %H:%M:%S')
    
    pl_history = pl_diag_merge[(pl_diag_merge['admit_date_shifted'] >= pl_diag_merge['date_for_filter_shifted']) &  (pl_diag_merge['date_for_filter_shifted']>=pl_diag_merge['admit_date_shifted']-datetime.timedelta(days=3*365))]
    pl_diag_subset = pl_history[["patient_num","encounter_num_x","epic_dx_description"]]
    pl_diag_subset["flag"]= 1
    
    pl_dedup_pivot = pl_diag_subset.pivot_table(index=['patient_num','encounter_num_x'],
                        columns='epic_dx_description', aggfunc='first').reset_index()
    flattened = pd.DataFrame(pl_dedup_pivot.to_records())
    #Drop features with small number of values
    lst = drop_features(flattened, 5)
    flattened.drop(columns=lst, inplace=True)
    flattened.columns = [hdr.replace(", ''","").replace("(*", "").replace("flag', ","").replace("'", "").replace(".0","").replace("('", "").replace("(", "").replace(")", "") \
                     for hdr in flattened.columns]
    flattened.drop(['index'], axis=1, inplace=True)
    flattened.fillna(0, inplace=True)                                               
    save_clean_data(flattened, 'Overdose_LOS_pl')
    print(f'The shape of the final table is {flattened.shape}')
    print('Finished')
    
# process_diag_result()
# process_med_order()    
process_demo_ls()
# process_problem_list()