import numpy as np
import pandas as pd
from utilities.new_functions import replace_missing_dates, replace_missing_dates_all
from utilities.new_functions import get_cells_data, train_test_val_split, add_column_names
from utilities.new_functions import get_X_y_all, plot_history, data_hist_predicted
from dl_logic.model import initialize_model, compile_model, train_model


def preprocess_data(
    file_path:str="~/code/Agermita/intelligent_network_expansion/raw_data/data_finale_V4.csv")-> tuple [pd.DataFrame]:

    processed_data = pd.read_csv(file_path, sep=',')

    processed_data.drop('Unnamed: 0', axis=1, inplace=True)
    processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%Y-%m-%d')
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


def train_model(X_train: pd.DataFrame = None, y_train: pd.DataFrame = None,
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
        X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict, cells, cells_data_trafic, start_date, end_date = preprocess_data("~/code/Agermita/intelligent_network_expansion/raw_data/data_finale_V4.csv")
        model, history = train_model(X_train, y_train, X_val, y_val)
        y_pred=predict_trafic_forcast(model, X_to_predict)
        df_final=data_hist_predicted(cells_data_trafic,y_pred, cells,start_date, end_date)
        # Save the real trafic and predicted trafic dataframe to csv file
        file_final="~/code/Agermita/intelligent_network_expansion/trafic_real_predict.csv"
        df_final.to_csv(file_final)
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
