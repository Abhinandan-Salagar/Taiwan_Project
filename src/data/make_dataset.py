import sys
#import logging
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
#from src.logger import create_log_path, CustomLogger

'''log_file_path = create_log_path('make_dataset')
# create the custom logger object
dataset_logger = CustomLogger(logger_name='make_dataset',
                              log_filename=log_file_path)
# set the level of logging to INFO
dataset_logger.set_log_level(level=logging.INFO)'''


def load_raw_data(input_path: Path) ->  pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows, columns = raw_data.shape
    '''dataset_logger.save_logs(msg=f'{input_path.stem} data read having {rows} rows and {columns} columns',
                             log_level='info')'''
    return raw_data


def train_val_split(data: pd.DataFrame,
                    test_size: float,
                    random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_data, val_data = train_test_split(data,
                                            test_size= test_size,
                                            random_state= random_state)
   
    return train_data, val_data




def save_data(data: pd.DataFrame,output_path: Path):
    data.to_csv(output_path,index=False)
    
    

def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
            
    except FileNotFoundError as e:
       
        default_dict = {'test_size': 0.25,
                        'random_state': None}
        # read the default_dictionary
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return test_size, random_state
        
    else:
       
        # read the parameters from the parameters file
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return test_size, random_state

def main():
    # read the input file name from command
    
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # interim data directory path
    interim_data_path = root_path / 'data' / 'interim'
    # make directory for the interim path
    interim_data_path.mkdir(exist_ok= True)
    # raw train file path
    raw_df_path = root_path / 'data' / 'raw' / 'rawdata' / 'train.csv'
    # load the training file
    raw_df = load_raw_data(input_path= raw_df_path)
    # parameters from params file
    #test_size, random_state = read_params('params.yaml')
    # split the file to train and validation data
    train_df, val_df = train_val_split(data= raw_df,
                                       test_size= 0.2,
                                       random_state= 42)
    # save the train data to the output path
    save_data(data= train_df, output_path= interim_data_path / 'train.csv')
    # save the val data to the output path
    save_data(data= val_df, output_path= interim_data_path / 'val.csv')
    
    
if __name__ == '__main__':
    main()
