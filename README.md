# EHR-Data-Process

Informatics for Integrating Biology and the Bedside (i2b2) is an open-source platform that enables the integration and analysis of clinical and research data. It provides a flexible and scalable framework for managing and querying large volumes of healthcare data. This document outlines the process for clinical data processing using the CTSI Honest Broker database as the data source for i2b2.

The CTSI Honest Broker database serves as a centralized repository for clinical data from various sources, including:

- Epic Electronic Health Record (EHR) system

- Froedtert & [Medical College of Wisconsin]((https://www.mcw.edu/)) legacy systems

- GE/IDX Physician Billing System

- Genetic sequencing result data

- NAACCR Tumor Registry

- OnCore Clinical Trials Management System

  

There are a series of tables in the entire healthcare database. To use basic clinical information, the key tables, such as diagnoses, encounters, diagnostic results, demographics, social history/lifestyle, and immunization, are considered.

The tables of diagnoses and encounters contain the basic hospitalization information of patients such as admission and discharge date, departments, length of stay, ICD-10 codes, etc. Diagnostic results, medication orders, and problem lists contain the patient's lab test, vital measurement, the medication used, and other diagnoses.

## Data Processing Workflow
![Table Merging](/Intro/images/Table%20Merging.png)

Firstly, merge the diagnosis and encounter table by the key "patient_num" and "encounter_num". Then, the new table (concatenation of tables of encounter and diagnosis) merges the remaining tables separately. Suppose the merged table is too sparse, like diagnostic results, medication orders and problem lists. In that case, pivot tables are created based on the columns of lab test, medication used, and diagnoses respectively, and then drop the columns with too many missing values. Lastly, all tables are merged into one table by the key "patient_num" or with "encounter_num".

## Data Processing Steps

![Data Path](/Intro/images/Data%20Path.png)

1. `config.py` and `data_config.py` are used to store the basic information:
   - Data location (The data path should follow the structure shown in the image: `.../Data/Froedtert/[data]`.)
   - Some locations to store your results
   - `enc_type` definition (For admitted patients, the types are 'IP', 'ED', 'UN', 'OS'. For general patients, you can include almost all types: 'IP', 'ED', 'UN', 'OS', 'AV', 'EI', 'IS')
   - ICD-code 10 definition
   - Columns needed.
2. Merge the diagnosis and encounter table to get basic patient information. And then, we filter admitted patients and recalculate the length of stay. Next, we also check the immunization status.Check the `Data Preprocessing/merge_data.py` for more details.







# Appendices

Appendix A: CTSI Honest Broker Data Dictionary [1](https://ctsi.mcw.edu/ctri/resources/bmi-links/), [2](https://ctsi.mcw.edu/images/sites/37/CTSI-Honest-Broker-Data-Dictionary.pdf)

Appendix B: Code Scripts (`config.py`, )
