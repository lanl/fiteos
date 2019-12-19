import os
import pandas as pd
import glob
import numpy as np


def get_files(directory):
    '''
    Input: directory path for .PVE files
    Output: list of sorted .PVE files
    '''
    
    files = glob.glob(os.path.join(directory,'*.PVE'))
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    return files

def get_data(f_path):
    '''
    Input: file path to .PVE file
    Output: Dictionary of values in .PVE file
    
    Create DataFrame from .PVE file. Then convert that into dictionary with the
    appropriate shape.
    '''
    
    data_df = pd.read_fwf(f_path)
    column_labels = ['Parameter name', 'value', 'esd']
    data_df = data_df[column_labels].replace({' ':''}, regex=True)
    data_df = data_df[column_labels].replace({'"':''}, regex=True)
        
    parameters_dict = data_df.set_index('Parameter name').T.to_dict('list')
    
    return parameters_dict

def create_param_dict(files):
    '''
    Input: list of .PVE file paths.
    Output: Nested dictionary of all sample parameters from all .PVE files.

    First parameter is value, second value is esd.
    
    Example output:
    {0: {'1C': [12.994755, 0.000154], '1VOL': [254.97099999999998, 0.003], '1A': [4.759882, 3.4e-05]},\
     1: {'1C': [13.994755, 0.000154],'1VOL': [254.97099999999998, 0.003],'1A': [5.759882, 3.4e-05]}}
    '''

    parameters_dict = {}
    for i in range(len(files)):
        print('%s : %s' % (i,files[i]))
        parameters = get_data(files[i])
        parameters_dict[i] = parameters
        
    return parameters_dict
    

if __name__ == "__main__":
    '''
    Retrieve parameters (value, esd) from .PVE GSAS output file.
    Put parameters into nested dictionary.

    John Lazarz
    190708
    '''
    
    # replace directory string with appropriate path to .PVE files
    directory = '/Users/johnlazarz/python/scripts/mesoscale-eos'

    files = get_files(directory)
    parameters_dict = create_param_dict(files)
    print(parameters_dict)