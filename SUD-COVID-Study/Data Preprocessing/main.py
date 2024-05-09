# import sys
# sys.path.append('..')
from merge_data import *


def main():
    merge_databases(['diagnosis', 'encounters', 'immunization', 'vitals', 'demographics', 'diagnostic_results'])
    # merge_diagnosis_encounter()
    merge_immune_data()

    # Extract dead patient and alive patients.





main()