from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re

from config import *
from utils import *



# Re-calculate the length of stay of patients
# Input: patient_df
# Output: patient_df with updated length of stay
def count_los(df):
    # Initialize output data
    output_data = []

    # Iterate over unique patients
    for patient in df['patient_num'].unique():
        # Get all records for this patient
        patient_data = df[df['patient_num'] == patient]

        # Initialize admission_data variable to hold the ongoing admission data
        admission_data = None

        # Iterate over records for this patient
        for idx, row in patient_data.iterrows():
            if admission_data is None:
                # If this is the first row, start a new admission
                admission_data = row.to_dict()
                admission_data['encounter_list'] = [admission_data['encounter_num']]
                continue

            # Check if this row's admit_date is before or at the previous discharge_date
            if row['admit_date_shifted'] <= admission_data['discharge_date_shifted']:
                # If yes, update the discharge_date and add the encounter to the list
                admission_data['discharge_date_shifted'] = max(admission_data['discharge_date_shifted'],
                                                               row['discharge_date_shifted'])
                admission_data['encounter_list'].append(row['encounter_num'])
            else:
                # If no, this is a new admission
                # First, calculate the length of stay of the previous admission and add it to the output
                admission_data['total_length_of_stay'] = (admission_data['discharge_date_shifted'] - admission_data[
                    'admit_date_shifted']).days + 1
                output_data.append(admission_data)

                # Then, start a new admission with this row
                admission_data = row.to_dict()
                admission_data['encounter_list'] = [admission_data['encounter_num']]

        # At the end of the patient's records, add the last admission
        admission_data['total_length_of_stay'] = (admission_data['discharge_date_shifted'] - admission_data[
            'admit_date_shifted']).days + 1
        output_data.append(admission_data)

    # Convert the output data to a DataFrame
    patient_df = pd.DataFrame(output_data)
    return patient_df


# Get patients with or without full vaccination
def filter_vaccinated_patients(fulfillment_dates, patient_df):
    # Convert the dates to datetime format
    fulfillment_dates["date_of_full_vaccination"] = pd.to_datetime(fulfillment_dates["date_of_full_vaccination"])
    patient_df["admit_date_shifted"] = pd.to_datetime(patient_df["admit_date_shifted"])
    fulfillment_dates["patient_num"] = fulfillment_dates["patient_num"].astype(str)
    patient_df["patient_num"] = patient_df["patient_num"].astype(str)

    # Merge the two data frames on patient id
    df = pd.merge(patient_df, fulfillment_dates, left_on='patient_num', right_on='patient_num', how='left')

    # Create a column "admission_after_vaccine" to track if the admission date is after the vaccine date
    df['admission_after_vaccine'] = df['admit_date_shifted'] >= df['date_of_full_vaccination'] + timedelta(days=14)

    # Define a function to filter admissions based on vaccine status
    def filter_admissions(group):
        if group['admission_after_vaccine'].any():
            return group[group['admission_after_vaccine']]
        else:
            return group

    # Apply the function to each patient group
    df_grouped = df.groupby('patient_num').apply(filter_admissions).reset_index(drop=True)

    # Split the dataframe into fully vaccinated patients and not fully vaccinated patients
    fully_vaccinated_patients = df_grouped[df_grouped['admission_after_vaccine'] == True]
    not_fully_vaccinated_patients = df_grouped[df_grouped['admission_after_vaccine'] == False]

    # Change 'True' to 'Yes' and 'False' to 'No' for the column 'admission_after_vaccine'
    fully_vaccinated_patients.loc[
        fully_vaccinated_patients['admission_after_vaccine'] == True, 'admission_after_vaccine'] = 'Yes'
    not_fully_vaccinated_patients.loc[
        not_fully_vaccinated_patients['admission_after_vaccine'] == False, 'admission_after_vaccine'] = 'No'

    return fully_vaccinated_patients, not_fully_vaccinated_patients


def process_demo(df):
    df['Marital Status'] = np.where((df['marital_status'] == 'Married') |
                                        (df['marital_status'] == 'Single') |
                                        (df['marital_status'] == 'Divorced') |
                                        (df['marital_status'] == 'Legally Separated') |
                                        (df['marital_status'] == 'Widowed'), df['marital_status'],
                                        "Other/not specified status")

    df['Race'] = np.where((df['race'].str.contains('White')) |
                              (df['race'].str.contains('Black')) |
                              (df['race'].str.contains('Asian')) |
                              (df['race'].str.contains('Indian')), df['race'], "Other/not specified races")

    df['Ethnicity'] = np.where((df['ethnicity'].str.contains('Non')) |
                                   (df['ethnicity'] == "Hispanic"), df['ethnicity'], "Other/not specified ethnicities")

    df['Employment Status'] = np.where((df['employment_status'] == 'Retired') |
                                           (df['employment_status'] == "Disabled") |
                                           (df['employment_status'] == "Not Employed") |
                                           (df['employment_status'] == "Full Time") |
                                           (df['employment_status'] == "Part Time") |
                                           (df['employment_status'] == "Self Employed")
                                           , df['employment_status'], "Other/not specified status")



    def categorize_zip(zip_code):
        mapping = {
            '530': 'Southeast',
            '531': 'Southeast',
            '532': 'Southeast',
            '534': 'Southeast',
            '535': 'South Central',
            '537': 'South Central',
            '541': 'Northeast',
            '542': 'Northeast',
            '543': 'Northeast',
            '544': 'North Central',
            '545': 'North Central',
            '546': 'Western',
            '547': 'Western',
            '548': 'Northwest',
            '549': 'Fox Valley',
        }
        return mapping.get(str(zip_code)[:3], 'Others')

    df['Region in Wisconsin'] = df['zip3'].apply(categorize_zip)
    

    map_ = {'Civilian Health and Medical Program for the VA (CHAMPVA)': 'Government Plan',
            'DEPARTMENTS OF CORRECTIONS': 'Other Methods',
            'Department of Veterans Affairs': 'Government Plan',
            'Local Government': 'Government Plan',
            'MEDICAID': 'Medicaid',
            'MEDICARE': 'Medicare',
            'Managed Care (Private)': 'Private',
            'Managed Care (private) or private health insurance (indemnity), not otherwise specified': 'Private',
            'Medicaid - Out of State': 'Medicaid',
            'Medicaid HMO': 'Medicaid',
            'Medicare HMO': 'Medicare',
            'Medicare Managed Care Other': 'Medicare',
            'Medicare Other': 'Medicare',
            'OTHER GOVERNMENT (Federal/State/Local) (excluding Department of Corrections)': 'Government Plan',
            'Other': 'Other Methods',
            'Other specified but not otherwise classifiable (includes Hospice - Unspecified plan)': 'Other Methods',
            'Private health insurance, other commercial Indemnity': 'Private',
            'Sharing Agreements': 'Other Methods',
            'TRICARE (CHAMPUS)': 'Government Plan',
            'Worker\'s Compensation': 'Other Methods'
            }
    df.loc[:, 'Primary Pay'] = df.loc[:, 'payor_type_primary_name'].map(map_).fillna('Other Methods')
    _drops = ['marital_status', 'race', 'ethnicity', 'ethnicity', 
              'employment_status', 'zip3', 'payor_type_primary_name']
    df.drop(columns=_drops, inplace=True)
    return df


def process_payment(df):
    map_ = {'Civilian Health and Medical Program for the VA (CHAMPVA)': 'Government Plan',
            'DEPARTMENTS OF CORRECTIONS': 'Other Methods',
            'Department of Veterans Affairs': 'Government Plan',
            'Local Government': 'Government Plan',
            'MEDICAID': 'Medicaid',
            'MEDICARE': 'Medicare',
            'Managed Care (Private)': 'Private',
            'Managed Care (private) or private health insurance (indemnity), not otherwise specified': 'Private',
            'Medicaid - Out of State': 'Medicaid',
            'Medicaid HMO': 'Medicaid',
            'Medicare HMO': 'Medicare',
            'Medicare Managed Care Other': 'Medicare',
            'Medicare Other': 'Medicare',
            'OTHER GOVERNMENT (Federal/State/Local) (excluding Department of Corrections)': 'Government Plan',
            'Other': 'Other Methods',
            'Other specified but not otherwise classifiable (includes Hospice - Unspecified plan)': 'Other Methods',
            'Private health insurance, other commercial Indemnity': 'Private',
            'Sharing Agreements': 'Other Methods',
            'TRICARE (CHAMPUS)': 'Government Plan',
            'Worker\'s Compensation': 'Other Methods'
            }
    df.loc[:, 'Primary Pay'] = df['payor_type_primary_name'].map(map_).fillna('Other Methods')
    
    df.drop(columns=['payor_type_primary_name'], inplace=True)
    return df


def get_population_results():
    demo_info = load_processed_data('statistics_result_of_all_demographics_to_table')

    # Define a function to extract numbers
    def extract_numbers(s):
        matches = re.findall(r'(\d+)', s)
        if matches:
            return int(matches[0])
        else:
            return None

    # Create the dictionary
    population_result = {}

    for index, row in demo_info.iterrows():
        feature = row['Feature']
        no_oud = extract_numbers(row['No OUD, n(%)'])
        oud = extract_numbers(row['OUD, n(%)'])

        # Check if feature belongs to the desired features list
        if feature in DESIRED_FEATURES:
            if no_oud is not None and oud is not None:
                population_result[feature] = [oud, no_oud]

    return population_result


def update_population_results(df):
    population_result = get_population_results()

    age_group = ['<45', '45-64', '>=65']
    vaccination = ['Yes', 'No']

    oud_group = df[df['OUD'] == 1]
    non_oud_group = df[df['OUD'] == 0]

    oud_age_dict = oud_group['Age'].value_counts().to_dict()
    non_oud_age_dict = non_oud_group['Age'].value_counts().to_dict()

    oud_vacc_dict = oud_group['Fully Vaccinated'].value_counts()
    non_oud_vacc_dict = non_oud_group['Fully Vaccinated'].value_counts()

    for group in age_group:
        oud_ = oud_age_dict[group]
        non_oud_ = non_oud_age_dict[group]
        population_result[group] = []
        population_result[group].append(oud_)
        population_result[group].append(non_oud_)

    for group in vaccination:
        oud_ = oud_vacc_dict[group]
        non_oud_ = non_oud_vacc_dict[group]
        population_result[group] = []
        population_result[group].append(oud_)
        population_result[group].append(non_oud_)

    population_result['Overall'] = population_result['Total']

    return population_result


#####################################


# Functions for processing patient data
# Create age groups <45 45-64 >=65
def create_age_groups(age):
    # df['age_group'] = pd.cut(df['age'], bins=[0, 45, 65, 120], labels=['<45', '45-64', '>=65'])
    if age < 45:
        return '<45'
    if age < 65:
        return '45-64'
    else:
        return '>=65'


def get_longest_stay(df):
    data_sorted = df.sort_values(['patient_num', 'total_length_of_stay'], ascending=[True, False])
    return data_sorted.drop_duplicates(subset='patient_num', keep='first')


def get_readmissioin_mortality(patients_all, read_days=30):
    patient_alive = patients_all[patients_all['death'] == 0]
    patient_dead = patients_all[patients_all['death'] == 1]
    data = patient_alive.copy()

    data['admit_date_shifted'] = pd.to_datetime(data['admit_date_shifted'])
    data['discharge_date_shifted'] = pd.to_datetime(data['discharge_date_shifted'])

    # Sort the dataframe by patient_num and admit_date_shifted
    data.sort_values(['patient_num', 'admit_date_shifted'], ascending=[True, True], inplace=True)
    # Calculate the number of days to next admission
    data['days_to_next_admission'] = data.groupby('patient_num')['admit_date_shifted'].shift(-1) - data[
        'discharge_date_shifted']
    # Create a new column '30_day_readmission' and
    # set its value to True where the next admission is within 30 days, False otherwise
    data['30_day_readmission'] = data['days_to_next_admission'].dt.days <= read_days

    # For patients with no readmission
    mask_no_readmission = ~data['patient_num'].isin(data.loc[data['30_day_readmission']]['patient_num'])
    data_no_readmission = data.loc[mask_no_readmission]

    # For each patient, find the record with the maximum total_length_of_stay
    idx = data_no_readmission.groupby('patient_num')['total_length_of_stay'].idxmax()
    data_no_readmission = data_no_readmission.loc[idx]

    # Combine dataframes
    data_30_readmit = pd.concat([data.loc[data['30_day_readmission']], data_no_readmission])

    data_30_readmit['total_length_of_stay'] = data_30_readmit['total_length_of_stay'].astype(int)
    data_readmit = get_longest_stay(data_30_readmit)
    data_readmit = process_payment(data_readmit)
    data_readmit_death = pd.concat([data_readmit, process_payment(patient_dead)])

    return data_readmit_death, data_readmit



def get_readmissioin(patients_all, read_days=30):

    data = patients_all

    data['admit_date_shifted'] = pd.to_datetime(data['admit_date_shifted'])
    data['discharge_date_shifted'] = pd.to_datetime(data['discharge_date_shifted'])

    # Sort the dataframe by patient_num and admit_date_shifted
    data.sort_values(['patient_num', 'admit_date_shifted'], ascending=[True, True], inplace=True)
    # Calculate the number of days to next admission
    data['days_to_next_admission'] = data.groupby('patient_num')['admit_date_shifted'].shift(-1) - data[
        'discharge_date_shifted']
    # Create a new column '30_day_readmission' and
    # set its value to True where the next admission is within 30 days, False otherwise
    days = str(read_days) + '_' + 'day_readmission'
    data[days] = data['days_to_next_admission'].dt.days <= read_days

    # For patients with no readmission
    mask_no_readmission = ~data['patient_num'].isin(data.loc[data[days]]['patient_num'])
    data_no_readmission = data.loc[mask_no_readmission]

    # Combine dataframes
    data_readmit = pd.concat([data.loc[data[days]], data_no_readmission])
    data_readmit.drop(columns=['days_to_next_admission'], inplace=True)
    data_readmit.sort_values(['patient_num', 'admit_date_shifted'], ascending=[True, True], inplace=True)

    return data_readmit