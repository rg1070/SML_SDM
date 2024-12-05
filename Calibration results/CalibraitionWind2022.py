###############################################################################
##Store all in a variable and run
SML_energy_water = r"""#########Energy Model##########################################################
Time = pd.Series(range(0,n))
#########Wind Model############################################################
###Variables
Wind_spd_IOSN3 = Data.loc[0:n-1,'WindSpeedIOSN3 (m/s)']
Height_conversion = 1.017
Wind_spd_adjusted = round(Wind_spd_IOSN3*Height_conversion,1).to_frame()

Wind_spd_adjusted = Wind_spd_adjusted.rename(columns={"WindSpeedIOSN3 (m/s)": "Wind"}).Wind

Wind_sens = 1

Wind_power = pd.merge(Wind_spd_adjusted, Power_Table, on ='Wind', how ='left').Power*Wind_eff*Wind_sens
Wind_gen = Wind_power/1000

#Stocks
Cumulative_Wind_gen_ini = pd.Series([0]) #initial value
Cumulative_Wind_gen_kWh = np.cumsum(pd.concat([Cumulative_Wind_gen_ini,Wind_power/1000/60], ignore_index = True))[:-1]
"""

##Model preparation
import pandas as pd
import numpy as np
import time

Dest = str(r'C:\Users\USNHIT\OneDrive - USNH\Roozbeh PhD research\Project 7 SML Water-Energy modeling\System Dynamics\SML Python SD Model')

Data = pd.read_csv(str(Dest)+'\Input Data.csv')
Power_Table = pd.read_csv(str(Dest)+'\Power table.csv')
Energy_Real = pd.read_csv(str(Dest)+'\Energy real data.csv')


##Wind Calibraition
n = 96480

E_Results = pd.DataFrame(columns = ['Efficiency','R2', 'MSE'], index = pd.Series(range(0,92)))


#finding the best Efficiency:
start = time.time()
for Ef in range(10, 101):
    Wind_eff = Ef/100
    exec(SML_energy_water)
    
    #PV generations accuracy
    x_value = Wind_gen
    y_value = Energy_Real.loc[:,'Wind (kW)']
    y_value = y_value.squeeze()
    correlation_matrix = np.corrcoef(x_value, y_value)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    MSE = np.square(np.subtract(x_value,y_value)).mean()
    print('E:',Ef/100,'R2:',r_squared, 'MSE:', MSE)
    
    E_Results.iloc[Ef-10,0] = Ef/100
    E_Results.iloc[Ef-10,1] = r_squared
    E_Results.iloc[Ef-10,2] = MSE

end = time.time()
print("The time of execution of above program is :", (end-start), "s")    
E_Results.to_csv(str(Dest)+'\Calibration results\Wind_calibration.csv', index = True)