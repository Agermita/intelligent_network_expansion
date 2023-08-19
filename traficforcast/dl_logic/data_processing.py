import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

#import matplotlib
#import matplotlib.pyplot as plt
#import seaborn as sns

#Path raw data
csv_path = os.path.join(os.getcwd(),  'raw_data')

#files to concatenate
files=['LTE KPIs Part 1.csv','LTE KPIs Part 2.csv','LTE KPIs Part 3.csv','LTE KPIs Part 4.csv']

#file with city information
file_city='site_city_type.csv'

#Date column that will be transformed to date
date_column='Date'

#columns to be scaled that will be added after running function replace_str_to_float
columns_to_scal=['Trafic LTE.float','L.Traffic.ActiveUser.Avg.float','L.Traffic.User.Avg.float','DL throughput_GRP.float','DL PRB Usage.float']

#data pre-processed file name
preproc_file=os.path.join('raw_data','data.csv')

#Concatanating files
def concat_files(path,liste_files):
    data=pd.DataFrame()
    for file in liste_files:
        df = pd.read_csv(os.path.join(path, file),sep=";")
        data= pd.concat([data, df], axis=0, ignore_index=True)
    return data

#add city and city type to dataframe
def add_cities(path,file,dataframe):
    df = pd.read_csv(os.path.join(path,file), sep=";")
    df.rename(columns={"eNodeB ID": "eNodeB identity"}, inplace=True)
    return (dataframe.merge(df, on="eNodeB identity", how="left"))



#Transform date from str to date format
def str_to_date(dataframe,column):
    dataframe[column] = pd.to_datetime(dataframe[column], format='%Y-%m-%d')
    return dataframe

def replace_str_to_float(data,column_name):
    nb_ligne=data.shape[0]
    new_column=[]
    new_val=0
    for n in range(nb_ligne):
        new_val=data[column_name][n].replace(',','.')
        new_val=new_val.replace('/','')
        new_column.append(float(new_val))
    return new_column

#scaling data
def data_scaling(dataframe,columns_to_scal):
    rb_scaler = RobustScaler()
    for column in columns_to_scal:
        rb_scaler.fit(dataframe[[column]])
        dataframe[column+'.scaled'] = rb_scaler.transform(dataframe[[column]])
    return dataframe

#adding mission dates for each cell
def replace_missing_dates(df, start_date, end_date) -> pd.DataFrame :
    missing_date=pd.date_range(start = start_date, end = end_date ).difference(df["Date"])

    df_new=df.copy()
    if(len(missing_date)>0):
        for i in range(0,len(missing_date)):
            data_sub={'Date': missing_date[i],
                    'eNodeB identity': df['eNodeB identity'][0],
                    'Cell ID' : df['Cell ID'][0],
                    'Cell FDD TDD Indication' :df['Cell FDD TDD Indication'][0],
                    'Downlink EARFCN' :df['Downlink EARFCN'][0],
                    'Downlink bandwidth' : df['Downlink bandwidth'][0],
                    'LTECell Tx and Rx Mode': df['LTECell Tx and Rx Mode'][0],
                    'Trafic LTE.float': -10,
                    'L.Traffic.ActiveUser.Avg.float.scaled': -10,
                    'L.Traffic.User.Avg.float.scaled':-10,
                    'DL throughput_GRP.float.scaled': -10,
                    'DL PRB Usage.float.scaled' : -10,
                    'City' : df['City'][0],
                    'City Type':df['City Type'][0],
                    'eNodeB_identifier_int':df['eNodeB_identifier_int'][0],
                    'Trafic LTE.float.scaled': -10
                    }
            print ("...")
            new_row=pd.DataFrame([data_sub])
            df_new=pd.concat([df_new,new_row])
            df_new.sort_values('Date')
    return df_new

def replace_missing_dates_all(df) -> pd.DataFrame :
    cells=df[['eNodeB_identifier_int']]
    cells=cells.drop_duplicates()
    start_date=df['Date'].min()
    end_date=df['Date'].max()

    for index, row in cells.iterrows():
        df_cell=df[(df["eNodeB_identifier_int"]==row[0])]
        df_cell=df_cell.reset_index(drop=True)

        df_cell=replace_missing_dates(df_cell, start_date, end_date)
        if index==0:
            df_new=df_cell
        else:
            df_new=pd.concat([df_new,df_cell])

    return df_new

#encoding non numerical values
def encoding_data(dataframe):
    # Instantiate the OneHotEncoder
    ohe_Downlink_EARFCN = OneHotEncoder(sparse = False)
    # Fit encoder
    ohe_Downlink_EARFCN.fit(dataframe[['Downlink EARFCN']])
    # Transform the current "Downlink EARFCN" column
    dataframe[ohe_Downlink_EARFCN.get_feature_names_out()] = ohe_Downlink_EARFCN.transform(dataframe[['Downlink EARFCN']])

    # Instantiate the OneHotEncoder
    ohe_Downlink_bandwidth = OneHotEncoder(sparse = False)
    # Fit encoder
    ohe_Downlink_bandwidth.fit(dataframe[['Downlink bandwidth']])
    # Transform the current "Downlink bandwidth" column
    dataframe[ohe_Downlink_bandwidth.get_feature_names_out()] = ohe_Downlink_bandwidth.transform(dataframe[['Downlink bandwidth']])

    # Instantiate the OneHotEncoder
    ohe_LTECell_Tx_and_Rx_Mode = OneHotEncoder(sparse = False)
    # Fit encoder
    ohe_LTECell_Tx_and_Rx_Mode.fit(dataframe[["LTECell Tx and Rx Mode"]])
    # Transform the current "Street" column
    dataframe[ohe_LTECell_Tx_and_Rx_Mode.get_feature_names_out()] = ohe_LTECell_Tx_and_Rx_Mode.transform(dataframe[['LTECell Tx and Rx Mode']])

    # Instantiate the OneHotEncoder
    ohe_city = OneHotEncoder(sparse = False)
    # Fit encoder
    ohe_city.fit(dataframe[["City"]])
    # Display the detected categories
    # Transform the current "City" column
    dataframe[ohe_city.get_feature_names_out()] = ohe_city.transform(dataframe[['City']])

    # Instantiate the OneHotEncoder
    ohe_City_Type = OneHotEncoder(sparse = False)
    # Fit encoder
    ohe_City_Type.fit(dataframe[["City Type"]])
    # Transform the current "City Type" column
    dataframe[ohe_City_Type.get_feature_names_out()] = ohe_City_Type.transform(dataframe[['City Type']])

    return dataframe

def fill_missing_values(dataframe):
    dataframe.loc[dataframe['Trafic LTE.float'] ==-10, (
        'Cell FDD TDD Indication_CELL_FDD', 'Cell FDD TDD Indication_CELL_TDD',
       'Downlink EARFCN_Band_1', 'Downlink EARFCN_Band_2',
       'Downlink EARFCN_Band_3', 'Downlink EARFCN_Band_4',
       'Downlink EARFCN_Band_5', 'Downlink bandwidth_CELL_BW_N100',
       'Downlink bandwidth_CELL_BW_N50', 'LTECell Tx and Rx Mode_1T1R',
       'LTECell Tx and Rx Mode_2T2R', 'LTECell Tx and Rx Mode_2T4R',
       'LTECell Tx and Rx Mode_4T4R', 'LTECell Tx and Rx Mode_8T8R',
       'City_City_1', 'City_City_3', 'City_City_4', 'City_City_5',
       'City Type_Rural', 'City Type_Urbain', 'City Type_Urbain dense')] = -10
    return dataframe

def data_preprocessing():
    #files concat + merge with cities information
    data=add_cities(csv_path,file_city,concat_files(csv_path,files))
    data=str_to_date(data,date_column)
    #Replacing string value by float
    data['Trafic LTE.float']=replace_str_to_float(data, 'Trafic LTE')
    data['L.Traffic.ActiveUser.Avg.float']=replace_str_to_float(data, 'L.Traffic.ActiveUser.Avg')
    data['L.Traffic.User.Avg.float']=replace_str_to_float(data, 'L.Traffic.User.Avg')
    data['DL throughput_GRP.float']=replace_str_to_float(data, 'DL throughput_GRP')
    data['DL PRB Usage.float']=replace_str_to_float(data, 'DL PRB Usage(%)')
    #dropping non numerical values
    data.drop(columns=['Trafic LTE','L.Traffic.ActiveUser.Avg','L.Traffic.User.Avg','DL throughput_GRP','DL PRB Usage(%)'], inplace=True)
    #Scaling data
    data=data_scaling(data,columns_to_scal)
    #adding an ID
    data['eNodeB_identifier_int']=(((256*data['eNodeB identity'])+(data['Cell ID'])))
    #adding missig lines for cells
    data=replace_missing_dates_all(data)
    data=encoding_data(data)
    #drop colomns that will not be used for the model
    data.drop(columns=['Cell FDD TDD Indication','Downlink EARFCN','Downlink bandwidth','LTECell Tx and Rx Mode','City','City Type'], inplace=True)
    data.drop(columns=['L.Traffic.ActiveUser.Avg.float', 'L.Traffic.User.Avg.float','DL throughput_GRP.float', 'DL PRB Usage.float'], inplace=True)
    #for rows with missing trafic replace all features with a -10
    data=fill_missing_values(data)
    print (data.columns)

    data.to_csv(preproc_file)
    return preproc_file
