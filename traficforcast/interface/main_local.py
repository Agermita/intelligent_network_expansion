import numpy as np
import pandas as pd
import os
from colorama import Fore, Style
from traficforcast.dl_logic.data_processing import data_preprocessing
from traficforcast.dl_logic.preparing_x_y import generate_X_y

#from traficforcast.utilities.new_functions import get_X_y_all, plot_history, get_cells_data, train_test_val_split, add_column_names

from traficforcast.dl_logic.preparing_x_y import data_hist_predicted
from traficforcast.dl_logic.model import initialize_model, compile_model, train_model
from tensorflow.keras import models

"""
def generate_X_y(
    file_path = os.path.join('raw_data','data.csv'))-> tuple [pd.DataFrame]:

    processed_data = pd.read_csv(file_path, sep=',')

    processed_data.drop('Unnamed: 0', axis=1, inplace=True)
    processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%Y-%m-%d')
    processed_data.loc[processed_data['Trafic LTE.float'] ==-10, ('Cell FDD TDD Indication_CELL_FDD', 'Cell FDD TDD Indication_CELL_TDD',
       'Downlink EARFCN_Band_1', 'Downlink EARFCN_Band_2',
       'Downlink EARFCN_Band_3', 'Downlink EARFCN_Band_4',
       'Downlink EARFCN_Band_5', 'Downlink bandwidth_CELL_BW_N100',
       'Downlink bandwidth_CELL_BW_N50', 'LTECell Tx and Rx Mode_1T1R',
       'LTECell Tx and Rx Mode_2T2R', 'LTECell Tx and Rx Mode_2T4R',
       'LTECell Tx and Rx Mode_4T4R', 'LTECell Tx and Rx Mode_8T8R',
       'City_City_1', 'City_City_3', 'City_City_4', 'City_City_5',
       'City Type_Rural', 'City Type_Urbain', 'City Type_Urbain dense')] = -10
    cells, cells_data, cells_data_trafic=get_cells_data(processed_data)
    columns=list(processed_data.columns)
    columns.remove('Date')
    columns.remove('eNodeB identity')
    columns.remove('Cell ID')
    columns.remove('eNodeB_identifier_int')
    X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict=get_X_y_all(cells_data, columns, 30)
    end_date=processed_data['Date'].max()
    start_date=processed_data['Date'].min()
    return X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict, cells, cells_data_trafic, start_date, end_date
"""

def train_model_function(X_train: pd.DataFrame = None, y_train: pd.DataFrame = None,
                X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> models:
    # sitting main parameters for the model
    input_shape =(X_train.shape[1],X_train.shape[2])
    output_length = y_train.shape[1]
    # initialize and comile the model
    model=initialize_model(input_shape, output_length)
    model=compile_model(model)

    # train the model on all cells data
    model, history=train_model(
            model,
            X_train,
            y_train,
            20, # try 50
            (X_val, y_val), # don't use validation data, use validation split rate
            0.3
        )

    return model, history


def predict_trafic_forcast(model:models=None, X_to_predict:np.array=None)-> np.array :
    y_pred=model.predict(X_to_predict)
    return y_pred

if __name__ == '__main__':
    try:
        print(Fore.BLUE + f"\nLoad original data and start preprocessing..." + Style.RESET_ALL)
        preproc_file=data_preprocessing() # data pre-processing

        print("✅ Data preprocessing finished")
        print(Fore.BLUE + f"\nLoad preprocessed data then split data to Train, val and Test..." + Style.RESET_ALL)

        X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict, cells, cells_data_trafic, start_date, end_date = generate_X_y(None, preproc_file)
        print("✅ Data split done successfully")
        #print(Fore.BLUE + f"\n Training the model..." + Style.RESET_ALL)

        model, history = train_model_function(X_train, y_train, X_val, y_val)
        print("✅ Model trained")

        print(Fore.BLUE + f"\n predict one month trafic..." + Style.RESET_ALL)
        y_pred=predict_trafic_forcast(model, X_to_predict)
        df_final=data_hist_predicted(cells_data_trafic,y_pred, cells,start_date, end_date)
        print("✅ Prediction done")

        print(Fore.BLUE + f"\n Save the real trafic and predicted trafic dataframe to csv file..." + Style.RESET_ALL)

        # Save the real trafic and predicted trafic dataframe to csv file
        file_final = "raw_data/trafic_real_predict.csv"
        #print (file_final)
        #file_final="~/code/Agermita/intelligent_network_expansion/raw_data/trafic_real_predict.csv"
        df_final.to_csv(file_final)

    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
