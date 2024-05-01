from config import data_path, save_data_path
import datetime
import pandas as pd
import os

# Read files and return a datafrane
def load_data(file_name):
    file1 = os.path.join(data_path, file_name + '1.csv')
    file2 = os.path.join(data_path, file_name + '2.csv')
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return pd.concat([df1,df2, ], ignore_index=True)


# remove duplicates in Diagnosis and Encounters
def remove_duplicates(data):
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    column_names = data.columns
    column_names = column_names.tolist()
    patient_num = 1
    prior_admission=[]
    process_prioradmission=0
    
    for index, row in data.iterrows():
        if(patient_num != row['patient_num']):
            if prior_admission:
                b_series = pd.Series(prior_admission,index=column_names)
                df2 = df2.append(b_series,ignore_index=True)
                df2.loc[(df2['patient_num']==patient_num) & (df2['encounter_num'] == encounter_num),'discharge_date_shifted']=discharge_date
            prior_admission = row.tolist()
            admit_date = row['admit_date_shifted']; discharge_date = row['discharge_date_shifted']; encounter_num=row['encounter_num']
            patient_num = row['patient_num']; discharge_status = row['discharge_status_c']; department_name = row['department_name']
            
        else:
            #print(patient_num,row['admit_date_shifted'].date() , discharge_date.date()+datetime.timedelta(days=1))  if(patient_num==16838993)
            process_prioradmission=0
            if(discharge_date is pd.NaT):
                #print(department_name,row['department_name'],row['admit_date_shifted'],admit_date.date())            
                if(row['admit_date_shifted'].date() <= admit_date.date() + datetime.timedelta(days=2)):
                    if(row['discharge_date_shifted'] is not pd.NaT):
                        discharge_date = row['discharge_date_shifted']
                        #prior_admission['discharge_date_shifted']=discharge_date
                    continue
                else:
                    process_prioradmission=1
                        
            elif(row['admit_date_shifted'].date() <= discharge_date.date()+datetime.timedelta(days=1)):
                
                if(row['discharge_date_shifted'] is not pd.NaT and row['discharge_date_shifted'].date() > discharge_date.date()):
                    discharge_date = row['discharge_date_shifted']
                    #prior_admission['discharge_date_shifted']=discharge_date
                continue
            else:
                process_prioradmission=1
                    
        if (len(prior_admission) !=0 & process_prioradmission ==1) :
            b_series = pd.Series(prior_admission,index=column_names)
            df2 = df2.append(b_series,ignore_index=True)
            df2.loc[(df2['patient_num']==patient_num) & (df2['encounter_num'] == encounter_num),'discharge_date_shifted']=discharge_date
            prior_admission = row.tolist()
    b_series = pd.Series(prior_admission,index=column_names)
    df2 = df2.append(b_series,ignore_index=True)
    df2.loc[(df2['patient_num']==patient_num) & (df2['encounter_num'] == encounter_num),'discharge_date_shifted']=discharge_date
    return df2

# save cleaned data
def save_clean_data(df, file_name):
    df.to_csv(os.path.join(save_data_path, file_name + '.csv'), index=False)
    
def read_preprocess_file(file_name):
    return pd.read_csv(os.path.join(save_data_path, file_name + '.csv'))

def drop_features(flattened, n):
    count=0
    lst=[]
    for x in flattened.columns:
        #print(x,"\n",flattened[x].isna().all())
        y=flattened[x].count()
        # print(y)
        if(y <= n):
            #print(x,flattened[x].count())
            lst.append(x)
            count+=1
    print(count)
    return lst

import pickle


def save_as_pickle(f, fn):
    pth = os.path.join(os.getcwd(), 'ML0-7', 'Dataset', fn+'.pk')
    with open(pth, 'wb') as p:
        pickle.dump(f, p)


def load_from_pickle(fn):
    pth = os.path.join(os.getcwd(), 'ML0-7', 'Dataset', fn + '.pk')
    with open(pth, 'rb') as p:
        return pickle.load(p)


def load_pd_pickle(fn):
    pth = os.path.join(os.getcwd(), 'ML0-7', 'Dataset', fn + '.pk')
    return pd.read_pickle(pth)
