##Stochasticity:
#GHI

import pandas as pd
from datetime import datetime, timedelta

# Define the start and end dates
start_date = datetime.strptime('06/01 00:00', '%m/%d %H:%M')
end_date = datetime.strptime('08/31 23:59', '%m/%d %H:%M')

# Generate date strings with minute time step
date_strings = [(start_date + timedelta(minutes=i)).strftime('%m/%d %H:%M') for i in range(int((end_date - start_date).total_seconds() / 60) + 1)]

# Create the DataFrame
N = len(date_strings)
GHI_Full = pd.DataFrame(columns=['Date', '2016', '2017', '2018', '2019', '2022'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
GHI_Full ['Date'] = date_strings

Wind_Full = GHI_Full.copy()
Temp_Full = GHI_Full.copy()
Precipitation_Full = GHI_Full.copy()
Temp_Full = GHI_Full.copy()
#Edemand_Full = GHI_Full.copy()
#Wdemand_Full = GHI_Full.copy()
###############################################################################
#Collect Temperature data
###############################################################################
## Import and add data 2022
Input2022 = pd.read_csv('Input Data.csv')
GHI_Full['2022']=Input2022['GHI (W/m2)']
Wind_Full['2022']=Input2022['WindSpeedIOSN3 (m/s)']
Temp_Full['2022'] = Input2022['AirTemperature (degrees C)']
Precipitation_Full['2022'] = Input2022['Rain (in)']
#Wdemand_Full['2022'] = Input2022['Demand tPlusDifference (gal)']
#Edemand_Full['2022'] = Input2022['Demand (W)']

###############################################################################
#Collect GHI data
###############################################################################
## Import and add GHI data 2016
# Create the DataFrame
N = len(date_strings)
GHI = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
GHI['Date'] = date_strings

GHI_SML = pd.read_csv('2016-06-01_2016-08-23_ecb_pyranometer_E_Irradiance_Global_Horizontal_1.csv')

GHI_SML['Date'] = pd.to_datetime(GHI_SML['Date'],  errors='coerce')
GHI_SML['Date'] = GHI_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
GHI = pd.merge(GHI, GHI_SML[['Date', 'E_Irradiance_Global_Horizontal_1 ()']], how='left', left_on='Date', right_on='Date')

GHI_Full['2016']=GHI['E_Irradiance_Global_Horizontal_1 ()']

## Import and add GHI data 2017
# Create the DataFrame
N = len(date_strings)
GHI = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
GHI['Date'] = date_strings

GHI_SML = pd.read_csv('2017-06-01_2017-08-23_ecb_pyranometer_E_Irradiance_Global_Horizontal_1.csv')

GHI_SML['Date'] = pd.to_datetime(GHI_SML['Date'],  errors='coerce')
GHI_SML['Date'] = GHI_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
GHI = pd.merge(GHI, GHI_SML[['Date', 'E_Irradiance_Global_Horizontal_1 ()']], how='left', left_on='Date', right_on='Date')

GHI_Full['2017']=GHI['E_Irradiance_Global_Horizontal_1 ()']

## Import and add GHI data 2018
# Create the DataFrame
N = len(date_strings)
GHI = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
GHI['Date'] = date_strings

GHI_SML = pd.read_csv('2018-06-01_2018-08-23_ecb_pyranometer_E_Irradiance_Global_Horizontal_1.csv')

GHI_SML['Date'] = pd.to_datetime(GHI_SML['Date'],  errors='coerce')
GHI_SML['Date'] = GHI_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
GHI = pd.merge(GHI, GHI_SML[['Date', 'E_Irradiance_Global_Horizontal_1 ()']], how='left', left_on='Date', right_on='Date')

GHI_Full['2018']=GHI['E_Irradiance_Global_Horizontal_1 ()']

## Import and add GHI data 2019
# Create the DataFrame
N = len(date_strings)
GHI = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
GHI['Date'] = date_strings

GHI_SML = pd.read_csv('2019-06-01_2019-08-23_ecb_pyranometer_E_Irradiance_Global_Horizontal_1.csv')

GHI_SML['Date'] = pd.to_datetime(GHI_SML['Date'],  errors='coerce')
GHI_SML['Date'] = GHI_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
GHI = pd.merge(GHI, GHI_SML[['Date', 'E_Irradiance_Global_Horizontal_1 ()']], how='left', left_on='Date', right_on='Date')

GHI_Full['2019']=GHI['E_Irradiance_Global_Horizontal_1 ()']

###############################################################################
#NA Treatment


GHI_Full['Date'] = pd.to_datetime(GHI_Full['Date'], format='%m/%d %H:%M')

# Initialize a flag to keep track of whether we are in the time range or not
in_time_range = False

# Iterate over the rows of the DataFrame
for index, row in GHI_Full.iterrows():
    # Check if the time is between 20:10 and 5:04
    if (row['Date'].hour >= 20 and row['Date'].minute >= 10) or (row['Date'].hour < 5 or (row['Date'].hour == 5 and row['Date'].minute <= 4)):
        # Set the flag to True if we enter the time range
        in_time_range = True
        # Set the values of columns '2016', '2017', '2018', '2019' to zero
        GHI_Full.loc[index, ['2016', '2017', '2018', '2019']] = 0
    elif in_time_range:
        # If we are in the time range and the time condition is not met, set the flag to False
        in_time_range = False

# Columns to process
columns_to_process = ['2016', '2017', '2018', '2019']

# Iterate over each column
for column in columns_to_process:
    # Find the first row index with non-null value in the current column
    first_non_null_index = GHI_Full[column].first_valid_index()

    # Iterate over rows starting from the first row index
    for index in range(first_non_null_index, len(GHI_Full)):
        # Check if the current value is NA
        if pd.isna(GHI_Full.at[index, column]):
            # Initialize a counter to keep track of consecutive NAs
            na_counter = 1
            
            # Calculate rolling mean with window size 10
            for i in range(1, 11):
                if index - i >= 0 and pd.isna(GHI_Full.at[index - i, column]):
                    na_counter += 1
                else:
                    break
            
            # If consecutive NAs exceed 10, leave it as NA
            if na_counter <= 10:
                # Interpolate using rolling average with window size 10
                GHI_Full.at[index, column] = GHI_Full[column].iloc[max(0, index - 10):index].interpolate().iloc[-1]

GHI_Full['Date'] = date_strings
#GHI_Full.insert(0, '', pd.Series(range(len(GHI_Full))))
GHI_Full.to_csv('GHI_Full.csv', index=True)
###############################################################################
#Collect Wind data
###############################################################################
## Import and add GHI data 2016
# Create the DataFrame
N = len(date_strings)
Wnd = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
Wnd['Date'] = date_strings

Wnd_SML = pd.read_csv('2016-06-01_2016-08-23_ecb_acuDC_instantaneous.csv')

Wnd_SML['Date'] = pd.to_datetime(Wnd_SML['Date'],  errors='coerce')
Wnd_SML['Date'] = Wnd_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
Wnd = pd.merge(Wnd, Wnd_SML[['Date', 'windSpeedIOSN3 (m/s)']], how='left', left_on='Date', right_on='Date')

Wind_Full['2016']=Wnd['windSpeedIOSN3 (m/s)']

## Import and add GHI data 2017
# Create the DataFrame
N = len(date_strings)
Wnd = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
Wnd['Date'] = date_strings

Wnd_SML = pd.read_csv('2017-06-01_2017-08-23_ecb_acuDC_instantaneous.csv')

Wnd_SML['Date'] = pd.to_datetime(Wnd_SML['Date'],  errors='coerce')
Wnd_SML['Date'] = Wnd_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
Wnd = pd.merge(Wnd, Wnd_SML[['Date', 'windSpeedIOSN3 (m/s)']], how='left', left_on='Date', right_on='Date')

Wind_Full['2017']=Wnd['windSpeedIOSN3 (m/s)']

## Import and add GHI data 2018
# Create the DataFrame
N = len(date_strings)
Wnd = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
Wnd['Date'] = date_strings

Wnd_SML = pd.read_csv('2018-06-01_2018-08-23_ecb_acuDC_instantaneous.csv')

Wnd_SML['Date'] = pd.to_datetime(Wnd_SML['Date'],  errors='coerce')
Wnd_SML['Date'] = Wnd_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
Wnd = pd.merge(Wnd, Wnd_SML[['Date', 'windSpeedIOSN3 (m/s)']], how='left', left_on='Date', right_on='Date')

Wind_Full['2018']=Wnd['windSpeedIOSN3 (m/s)']

## Import and add GHI data 2019
# Create the DataFrame
N = len(date_strings)
Wnd = pd.DataFrame(columns=['Date'], index=pd.Series(range(0, N)))

# Assign the date strings to the "Date" column
Wnd['Date'] = date_strings

Wnd_SML = pd.read_csv('2019-06-01_2019-08-23_ecb_acuDC_instantaneous.csv')

Wnd_SML['Date'] = pd.to_datetime(Wnd_SML['Date'],  errors='coerce')
Wnd_SML['Date'] = Wnd_SML['Date'].dt.strftime('%m/%d %H:%M')

# Merge the two DataFrames based on the "Date" column
Wnd = pd.merge(Wnd, Wnd_SML[['Date', 'windSpeedIOSN3 (m/s)']], how='left', left_on='Date', right_on='Date')

Wind_Full['2019']=Wnd['windSpeedIOSN3 (m/s)']

# Columns to process
columns_to_process = ['2016', '2017', '2018', '2019']

# Iterate over each column
for column in columns_to_process:
    # Find the first row index with non-null value in the current column
    first_non_null_index = Wind_Full[column].first_valid_index()

    # Iterate over rows starting from the first row index
    for index in range(first_non_null_index, len(Wind_Full)):
        # Check if the current value is NA
        if pd.isna(Wind_Full.at[index, column]):
            # Initialize a counter to keep track of consecutive NAs
            na_counter = 1
            
            # Calculate rolling mean with window size 10
            for i in range(1, 11):
                if index - i >= 0 and pd.isna(Wind_Full.at[index - i, column]):
                    na_counter += 1
                else:
                    break
            
            # If consecutive NAs exceed 10, leave it as NA
            if na_counter <= 10:
                # Interpolate using rolling average with window size 10
                Wind_Full.at[index, column] = Wind_Full[column].iloc[max(0, index - 10):index].interpolate().iloc[-1]


Wind_Full.to_csv('Wind_Full.csv', index=True)
###############################################################################
#Collect Temp data Daily data: Source IOSN3
###############################################################################

Temp_2019 = pd.read_csv('2019.txt', sep="\s+", usecols=['#YY', 'MM', 'DD', 'hh', 'ATMP'],index_col=False).iloc[1:]
Temp_2018 = pd.read_csv('2018.txt', sep="\s+", usecols=['#YY', 'MM', 'DD', 'hh', 'ATMP'],index_col=False).iloc[1:]
Temp_2017 = pd.read_csv('2017.txt', sep="\s+", usecols=['#YY', 'MM', 'DD', 'hh', 'ATMP'],index_col=False).iloc[1:]
Temp_2016 = pd.read_csv('2016.txt', sep="\s+", usecols=['#YY', 'MM', 'DD', 'hh', 'ATMP'],index_col=False).iloc[1:]

Temp_2019['Date'] = pd.to_datetime(Temp_2019['#YY'].astype(str) + '-' + Temp_2019['MM'].astype(str) + '-' + Temp_2019['DD'].astype(str) + ' ' + Temp_2019['hh'].astype(str), format='%Y-%m-%d %H')
Temp_2018['Date'] = pd.to_datetime(Temp_2018['#YY'].astype(str) + '-' + Temp_2018['MM'].astype(str) + '-' + Temp_2018['DD'].astype(str) + ' ' + Temp_2018['hh'].astype(str), format='%Y-%m-%d %H')
Temp_2017['Date'] = pd.to_datetime(Temp_2017['#YY'].astype(str) + '-' + Temp_2017['MM'].astype(str) + '-' + Temp_2017['DD'].astype(str) + ' ' + Temp_2017['hh'].astype(str), format='%Y-%m-%d %H')
Temp_2016['Date'] = pd.to_datetime(Temp_2016['#YY'].astype(str) + '-' + Temp_2016['MM'].astype(str) + '-' + Temp_2016['DD'].astype(str) + ' ' + Temp_2016['hh'].astype(str), format='%Y-%m-%d %H')

Temp_2019.drop(columns=['#YY', 'MM', 'DD', 'hh'], inplace=True)
Temp_2018.drop(columns=['#YY', 'MM', 'DD', 'hh'], inplace=True)
Temp_2017.drop(columns=['#YY', 'MM', 'DD', 'hh'], inplace=True)
Temp_2016.drop(columns=['#YY', 'MM', 'DD', 'hh'], inplace=True)

Temp_2019 = Temp_2019[['Date', 'ATMP']]
Temp_2018 = Temp_2018[['Date', 'ATMP']]
Temp_2017 = Temp_2017[['Date', 'ATMP']]
Temp_2016 = Temp_2016[['Date', 'ATMP']]

Temp_2019['Date'] = pd.to_datetime(Temp_2019['Date']).dt.strftime('%m/%d %H:%M')
Temp_2018['Date'] = pd.to_datetime(Temp_2018['Date']).dt.strftime('%m/%d %H:%M')
Temp_2017['Date'] = pd.to_datetime(Temp_2017['Date']).dt.strftime('%m/%d %H:%M')
Temp_2016['Date'] = pd.to_datetime(Temp_2016['Date']).dt.strftime('%m/%d %H:%M')

Temp_Full = Temp_Full[['Date', '2022']]
Temp_Full = pd.merge(Temp_Full, Temp_2016, on="Date", how="left")
Temp_Full.rename(columns={"ATMP": "2016"}, inplace=True)
Temp_Full = pd.merge(Temp_Full, Temp_2017, on="Date", how="left")
Temp_Full.rename(columns={"ATMP": "2017"}, inplace=True)
Temp_Full = pd.merge(Temp_Full, Temp_2018, on="Date", how="left")
Temp_Full.rename(columns={"ATMP": "2018"}, inplace=True)
Temp_Full = pd.merge(Temp_Full, Temp_2019, on="Date", how="left")
Temp_Full.rename(columns={"ATMP": "2019"}, inplace=True)



for columns in ['2016', '2017', '2018', '2019']:
    for i in range(0,132480,60):
        
        Temp_Full[columns][(i+1):(i+60)]=Temp_Full[columns][i]
    
Temp_Full = Temp_Full.reindex(columns=['Date', '2016', '2017', '2018', '2019', '2022'])


Temp_Full.to_csv('Temp_Full.csv', index=True)

'''
###############################################################################
# Get a list of all global variables
global_vars = list(globals().keys())

# Get a list of all local variables
local_vars = list(locals().keys())

# Combine global and local variables and remove duplicates
all_vars = set(global_vars + local_vars)

# Define the datasets to keep
datasets_to_keep = ['Wind_Full', 'GHI_Full', 'Precipitation_Full', 'Temp_Full']

# Remove all variables except the desired datasets
for var_name in all_vars:
    if var_name not in datasets_to_keep:
        del globals()[var_name]
        
del var_name
del datasets_to_keep
del all_vars
del local_vars
'''
###############################################################################
#Collect Precipitation data
###############################################################################



