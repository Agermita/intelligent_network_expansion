import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

'''function to add missing dates on the Dataframe timeseries for one cell'''
def replace_missing_dates(df, start_date, end_date) -> pd.DataFrame :
    # ====================================================================
    # function to add missing dates on the Dataframe timeseries for one cell
    # df : dataframe containing the data of one cell
    # start_date and end_date : are used to define the range of dates that should existe on the datframe
    # ====================================================================
    missing_date=pd.date_range(start = start_date, end = end_date ).difference(df["Date"])

    df_new=df.copy()
    if(len(missing_date)>0):
        for i in range(0,len(missing_date)):
            data={'Date': missing_date[i],
                    'eNodeB identity': df['eNodeB identity'][0],
                    'Cell ID' : df['Cell ID'][0],
                    'Cell FDD TDD Indication' :df['Cell FDD TDD Indication'][0],
                    'Downlink EARFCN' :df['Downlink EARFCN'][0],
                    'Downlink bandwidth' : df['Downlink bandwidth'][0],
                    'LTECell Tx and Rx Mode': df['LTECell Tx and Rx Mode'][0],
                    'Trafic LTE': 0,
                    'L.Traffic.ActiveUser.Avg': 0,
                    'DL throughput_GRP': 0,
                    'DL PRB Usage(%)' : 0
                    }
            print (data)
            new_row=pd.DataFrame([data])
            df_new=pd.concat([df_new,new_row])
            df_new.sort_values('Date')
    return df_new

def replace_missing_dates_all(df) -> pd.DataFrame :
    # ====================================================================
    # function raning a loop on the replace_missing_dates to add missing dates for all cells
    # df : dataframe containing dataframes of all cells
    # ====================================================================
    cells=df[["eNodeB identity",'Cell ID']]
    cells=cells.drop_duplicates()
    start_date=df['Date'].min()
    end_date=df['Date'].max()

    for index, row in cells.iterrows():
        df_cell=df[(df["eNodeB identity"]==row[0]) & (df["Cell ID"]==row[1])]
        df_cell=df_cell.reset_index(drop=True)

        df_cell=replace_missing_dates(df_cell, start_date, end_date)
        if index==0:
            df_new=df_cell
        else:
            df_new=pd.concat([df_new,df_cell])

    return df_new


def get_cells_data(df) ->tuple [np.array] :
    # ====================================================================
    # split initial data_processed to dataframes of all cells
    #   (each dataframe contains the data of one cell)
    #       the concatenat them to one 3d array

    # Args :
    #       df : initial dataframe containing data processed

    # returns :
    #   tuple of :
    #       (
    #       dataframe of cells identifiers shape=num_cells,
    #       3d array of cells_data shape=(num_cells, num_days, columns),
    #       2d array of the trafic per day per cell shape=(num_cells, num_days)
    #       )
    # ====================================================================

    """create Dataframe of cells"""
    cells=df[["eNodeB identity",'Cell ID','eNodeB_identifier_int']].sort_values(by='eNodeB_identifier_int')
    cells=cells.drop_duplicates()
    #cells_2=cells.iloc[:20]  # to limit to first 500 CELLS

    data=[]
    data_trafic=[]

    """ loop on Dataframe of cells to filter data of each cell"""
    for index, row in cells.iterrows():
        df_cell=df[(df["eNodeB identity"]==row[0]) & (df["Cell ID"]==row[1])].sort_values(by='Date')

        df_cell=df_cell.reset_index(drop=True)

        #df_cell_1=df_cell.copy()
        df_cell.drop(['Date','eNodeB identity', 'Cell ID', 'eNodeB_identifier_int'], axis=1, inplace=True)

        df_cell_trafic=df_cell["Trafic LTE.float"]


        data.append(df_cell)
        data_trafic.append(df_cell_trafic)
    """
          cells_data=np.array(data)
          cells_data_trafic=np.array(data_trafic)
    """
    return cells, np.array(data), np.array(data_trafic)

""" Split data of each cell into train val and test """
def train_test_val_split(data_cell:pd.DataFrame,
                     output_length: int) -> tuple[pd.DataFrame]:
    # ==========================================================================
    #From a cell data (dataframe), create a train dataframe, validation dataframe
    # and test dataframe then split them into X and y using a fixed output_length

    # Args:
    #    data_cell (pd.DataFrame): chronological data of one cell
    #    output_length (int): How long y will be = the number of days to be predicted : one month 30 days

    #Returns:
    #    Tuple[pd.DataFrame]: A tuple of 7 dataframes
    #       (X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict )
    # ==========================================================================


    # TRAIN SET
    # ======================
    X_train_index_last = round(len(data_cell)-3*output_length)

    X_train=pd.DataFrame(data_cell.iloc[0:X_train_index_last, :])
    X_train.drop('Trafic LTE.float', axis=1, inplace=True) # delete original trafic column
    y_train=pd.DataFrame(
        data_cell.iloc[X_train_index_last:X_train_index_last+output_length][['Trafic LTE.float']])

    # Validation SET
    # ======================
    X_val=pd.DataFrame(data_cell.iloc[X_train_index_last+output_length-len(X_train):X_train_index_last+output_length , :])
    X_val.drop('Trafic LTE.float', axis=1, inplace=True) # delete original trafic column
    y_val=pd.DataFrame(data_cell.iloc[X_train_index_last+output_length:X_train_index_last+2*output_length][['Trafic LTE.float']])

    # Test SET
    # ======================
    X_test=pd.DataFrame(data_cell.iloc[X_train_index_last+2*output_length-len(X_train):X_train_index_last+2*output_length , :])
    X_test.drop('Trafic LTE.float', axis=1, inplace=True) # delete original trafic column
    y_test=pd.DataFrame(data_cell.iloc[X_train_index_last+2*output_length:X_train_index_last+3*output_length][['Trafic LTE.float']])

    # Dataframe to be used as input for prediction
    # ======================
    X_to_predict=pd.DataFrame(data_cell.iloc[len(data_cell)-len(X_train):len(data_cell) , :])
    X_to_predict.drop('Trafic LTE.float', axis=1, inplace=True) # delete original trafic column

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict)

"""
rename Dataframe columns
"""
def add_column_names(cell_data:np.array, columns:list):
    # ==========================================================================
    # rename dataframe columns

    # Args:
    #    cell_data (pd.DataFrame): chronological data of one cell
    #    columns (list): column names

    #Returns:
    #    pd.DataFrame: cell_data with columns renamed
    # ==========================================================================
    cell_i_data=pd.DataFrame(cell_data)
    cell_i_data.columns=columns
    return cell_i_data
"""
concatenate X_i for all cells in one array, and y_i for all cells in one array
"""
def get_X_y_all(cells_data:pd.DataFrame, columns: list,
                     output_length: int) -> tuple[pd.DataFrame]:
    # ====================================================================
    # generate X and y for all cells (whith a loop of train_test_val_split function on each cell)
    # of the given input_length and output_length

    # Args:
    #    cells_data (pd.DataFrame): dataframe of all cells data
    #     columns : columns names
    #     output_length (int): Length of each y_i, number of days to be predicted 30 days

    # Returns:
    #     Tuple[np.array]: A tuple of numpy arrays (X, y)
    # ====================================================================
    cell_index=0

    for cell in cells_data:
      # ====================================================================
      # generate X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict
      # for each cell
      # ====================================================================
      X_train, y_train, X_val, y_val, X_test, y_test, X_to_predict=train_test_val_split(add_column_names(cell, columns), output_length)

      if cell_index==0:
        X_train_all=X_train
        y_train_all=y_train
        X_val_all=X_val
        y_val_all=y_val
        X_test_all=X_test
        y_test_all=y_test
        X_to_predict_all=X_to_predict
      else :
        # ====================================================================
        # concatenate DataFrame of each cell to the globale dataframe
        # ====================================================================
        X_train_all= pd.concat([X_train_all,X_train])
        y_train_all=pd.concat([y_train_all , y_train])
        X_val_all=pd.concat([X_val_all , X_val])
        y_val_all=pd.concat([y_val_all , y_val])
        X_test_all=pd.concat([X_test_all , X_test])
        y_test_all=pd.concat([y_test_all ,y_test])
        X_to_predict_all=pd.concat([X_to_predict_all ,X_to_predict])
        print (len(X_train_all), len(y_train_all), len(X_test_all), len(y_test_all))
      cell_index=cell_index+1

    # ====================================================================
    # reshape X to (number of cells , number of days= , number of features=26)
    # reshape y to (number of cells , outputh_lenth=30 , number of features to be predicted = 1)
    # ====================================================================
    X_train_all_array=np.reshape(np.array(X_train_all),(len(cells_data),X_train.shape[0], X_train.shape[1]))
    y_train_all_array=np.reshape(np.array(y_train_all),(len(cells_data),y_train.shape[0], y_train.shape[1]))
    X_val_all_array=np.reshape(np.array(X_val_all),(len(cells_data),X_train.shape[0], X_train.shape[1]))
    y_val_all_array=np.reshape(np.array(y_val_all),(len(cells_data),y_train.shape[0], y_train.shape[1]))
    X_test_all_array=np.reshape(np.array(X_test_all),(len(cells_data),X_train.shape[0], X_train.shape[1]))
    y_test_all_array=np.reshape(np.array(y_test_all),(len(cells_data),y_train.shape[0], y_train.shape[1]))
    X_to_predict_all=np.reshape(np.array(X_to_predict_all),(len(cells_data),X_train.shape[0], X_train.shape[1]))


    return (X_train_all_array, y_train_all_array,X_val_all_array,
            y_val_all_array, X_test_all_array , y_test_all_array, X_to_predict_all)


"""
Function to plot visualize the training of your RNN over epochs.
This function shows both the evolution of the loss function (MSE) and metrics (MAE)
"""
def plot_history(history):

    fig, ax = plt.subplots(1,2, figsize=(20,7))
    # --- LOSS: MSE ---
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('MSE')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- METRICS:MAE ---

    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    return ax

""" Concatenate real trafic to predicted one per cell """
def data_hist_predicted(y, y_pred, cells, start_date, end_date):
    # ====================================================================
    # generate X and y for all cells (whith a loop of train_test_val_split function on each cell)
    # of the given input_length and output_length

    # Args:
    #   y (np.array): 3d array of cronological trafic per cell shape=(num_cells, num_real_days, 1)
    #   y_pred (np.array): 3d array of cronological predicted trafic per cell shape=(num_cells, num_days=30, 1)
    #   cells (pd.DataFrame): dataframe of cells identifiers
    #   start_day (date) : first date of the historycal data (initial dataframe)
    #   end_date (date) : last date of the historycal data (initial dataframe)

    # Returns:
    #     pd.DataFrame: a dataframe containing the global trafic (history + prediction) per cell per date
    # ====================================================================
    end_date_2=end_date+datetime.timedelta(days=y_pred.shape[1])
    dates=pd.date_range(start = start_date, end = end_date_2)
    print (y.shape , y_pred.shape)
    # reshape the y to the 2d
    y_reshaped_2d=y
    y_reshaped_2d=y_reshaped_2d.reshape(-1, y.shape[1])
    print (y_reshaped_2d.shape)
    print (f"y shape 2d : {y_reshaped_2d.shape}")
    # convert y and y_pred to Dataframe
    list_y_pred=pd.DataFrame(y_pred)

    list_y=pd.DataFrame(y_reshaped_2d)
    cells=cells.reset_index(drop=True) # to be deleted after
    #concatenate y and y_pred and cell ids
    cell_data=pd.concat([list_y,list_y_pred],axis=1, ignore_index=True, sort=False)
    print (f"concatenation of y and y_pred : {cell_data.shape}")

    cell_data_final=pd.concat([cells,cell_data],axis=1, sort=False)
    # rename columns
    columns=['eNodeB identity','Cell ID','eNodeB_identifier_int']
    comunms_all=columns+(list(dates))
    cell_data_final.columns=comunms_all

    cell_data_final.set_index(['eNodeB identity','Cell ID','eNodeB_identifier_int'])

    # format the  dataframe to have dates as one column
    cell_data_final2=cell_data_final.melt(id_vars=['eNodeB identity','Cell ID','eNodeB_identifier_int'],
        var_name="Date",
        value_name="Trafic")
    cell_data_final2['Trafic'] = pd.to_numeric(cell_data_final2['Trafic'])
    # specify what are predicted and real dates

    cell_data_final2['Flag '] = ['Real' if x <= end_date else 'Predicted' for x in cell_data_final2['Date']]

    return cell_data_final2
