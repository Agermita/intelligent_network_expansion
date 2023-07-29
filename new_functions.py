import pandas as pd
import numpy as np

'''function to add missing dates on the Dataframe timeseries for one cell'''
def replace_missing_dates(df, start_date, end_date) -> pd.DataFrame :
    missing_date=pd.date_range(start = start_date, end = end_date ).difference(df["Date"])
    date={}
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

""" this function will format the X_initial dataframe global to get X with 3 dimensions X.shape=(number of cell, number of days, number of features) """
def create_X(df) ->np.array :
    cells=df[["eNodeB identity",'Cell ID']]
    cells=cells.drop_duplicates()
    start_date=df['Date'].min()
    end_date=df['Date'].max()
    print (cells.shape)
    data=[]
    for index, row in cells.iterrows():
        df_cell=df[(df["eNodeB identity"]==row[0]) & (df["Cell ID"]==row[1])]
        df_cell=df_cell.reset_index(drop=True)
        print (f'for cell {index} here is the shape before : {df_cell.shape}')
        df_cell=replace_missing_dates(df_cell, start_date, end_date)
        print (f'for cell {index} here is the shape After : {df_cell.shape}')
        data.append(df_cell)

    print (len(data))


    X=np.array(data)
    return X

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
