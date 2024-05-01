import os
from sys import platform
home = os.environ['HOME']

if platform == "win32":
    data_dir = 'D:\Data\Froedtert\opioid'
else:
    home = os.environ['HOME']
    data_dir = os.path.join(home, 'Data', 'Froedtert')

COVID_DATA_NAME_1 = 'covid-May2021'
COVID_DATA_NAME_2 = 'COVID+June2021-Jan2023'
COVID_SUD_DATA_NAME = 'SUD_SM_COVID_Jan2020_Jan2023'

COVID_DIR_1 = os.path.join(data_dir, COVID_DATA_NAME_1)
COVID_DIR_2 = os.path.join(data_dir, COVID_DATA_NAME_2)
COVID_SUD_DIR = os.path.join(data_dir, COVID_SUD_DATA_NAME)


def main():
    pass

if __name__ == "__main__":
    main()