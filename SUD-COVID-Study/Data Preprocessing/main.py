# import sys
# sys.path.append('..')
from merge_data import merge_databases, merge_diagnosis_encounter


def main():
    merge_databases(['diagnosis', 'encounters', 'immunization', 'vitals', 'demographics', 'diagnostic_results'])
    merge_diagnosis_encounter()

main()