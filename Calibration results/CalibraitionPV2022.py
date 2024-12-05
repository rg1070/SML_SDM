##PV module efficiency calibration
###############################################################################
##Store all in a variable and run
SML_energy_water = r"""#########Energy Model##########################################################
Time = pd.Series(range(0,n))
#########PV Model##############################################################
###Variables
Day_of_year = Data.loc[0:n-1,'Day of year']
Local_standard_time = Data.loc[0:n-1,'Civil time']
GHI_W_m2 = Data.loc[0:n-1,'GHI (W/m2)']
#GHI_W_m2[0:1440].plot()
GHI_kW_m2 = pd.Series(np.where(GHI_W_m2<25,0,GHI_W_m2/1000))
#GHI_kW_m2[0:1440].plot()
Slope_d = 20
Azimuth_d = 170
Slope_of_surface_r = Slope_d/180
Azimuth_gamma_r = Azimuth_d/180
Ground_ref_albedo = 0.1
Solar_constant = 1.367
Longitude_d = -70.78
Longitude_r = Longitude_d/180
Time_zone = -4
Latitute_Phi_d = 43.01
Latitute_Phi_r = Latitute_Phi_d/180
B = 360*((Day_of_year-1)/365)
E = 3.82*(7.5e-005+0.001868*np.cos(B)-0.032077*np.sin(B)-0.014615*np.cos(2*B)-0.04089*np.sin(2*B))
Solar_time_ts = Local_standard_time+(Longitude_r/15)-Time_zone+E
Hour_angle_omega = (Solar_time_ts-12)*15/180
Solar_declination_delta = 23.45*np.sin(((360*((284+Day_of_year)/365)))/180)/180 #radian
Angle_of_incident_Cos_theta = np.sin( Solar_declination_delta )*np.sin(Latitute_Phi_r)*np.cos( Slope_of_surface_r )-np.sin(Solar_declination_delta)*np.cos(Latitute_Phi_r)*np.sin(Slope_of_surface_r)*np.cos(Azimuth_gamma_r)+np.cos(Solar_declination_delta)*np.cos(Latitute_Phi_r)*np.cos(Slope_of_surface_r)*np.cos(Azimuth_gamma_r)+np.cos(Solar_declination_delta)*np.sin(Latitute_Phi_r)*np.sin(Slope_of_surface_r)*np.cos(Azimuth_gamma_r)*np.cos(Hour_angle_omega)+np.cos(Solar_declination_delta)*np.sin(Slope_of_surface_r)*np.sin(Azimuth_gamma_r)*np.sin(Hour_angle_omega)
Zenith_angle_Cos_theta_z = np.cos(Latitute_Phi_r)*np.cos(Solar_declination_delta)*np.cos(Hour_angle_omega)+np.sin(Latitute_Phi_r)*np.sin(Solar_declination_delta)
The_extraterrestrial_normal_radiation = Solar_constant*(1+0.033*np.cos(360*Day_of_year/365))
Rb = Angle_of_incident_Cos_theta/Zenith_angle_Cos_theta_z #The ratio of beam radiation on the tilted surface to beam radiation on the horizontal surface Rb
Extraterrestrial_h_radiation_avg = (12/3.1415)*The_extraterrestrial_normal_radiation*(np.cos(Latitute_Phi_r)*np.cos(Solar_declination_delta)*(np.sin(Hour_angle_omega+0.25)-np.sin(Hour_angle_omega))+(3.1415*0.25*np.sin(Latitute_Phi_r)*np.sin(Solar_declination_delta)/180)) #The extraterrestrial horizontal radiation averaged over the time step
Kt = GHI_kW_m2/(Extraterrestrial_h_radiation_avg) #Clearness index Kt
Diff_less = pd.Series(np.where(Kt<=0.22,GHI_kW_m2*(1-0.09*Kt),0))
Diff_more = pd.Series(np.where(Kt>0.80,GHI_kW_m2*0.165,0))
Diff_between = pd.Series(np.where(Kt>0.22 ,GHI_kW_m2*(0.9511-0.1604*Kt+4.388*Kt**2-16.638*Kt**3+12.336*Kt**4),0))
Diff_between = pd.Series(np.where(Kt>0.80 ,0,Diff_between))
Diff_radiation = Diff_less+Diff_more+Diff_between
Beam_radiation = GHI_kW_m2-Diff_radiation
Ai = Beam_radiation/Extraterrestrial_h_radiation_avg#Anisotropy index Ai
f_Cloud = np.sqrt(Beam_radiation/GHI_kW_m2).replace(np. nan,0) #Cloudiness factor f

PV_incident_radiation = (Beam_radiation+Diff_radiation*Ai)*Rb+Diff_radiation*(1-Ai)*((1+np.cos(Slope_of_surface_r))/2)*(1+f_Cloud*np.sin(Slope_of_surface_r/2)**3)+GHI_kW_m2*Ground_ref_albedo*((1-np.cos(Slope_of_surface_r))/2)

Old_panels = 233

Number_of_PV_installed = Old_panels + New_panels
Ambient_temp = Data.loc[0:n-1,'AirTemperature (degrees C)']

Solar_radiation_incident_on_PV_array_GHI = PV_incident_radiation*1000
Inverter_eff = 1
PV_derating_factor = 0.95
System_losses = 0.15
Average_res_PV_size = 1.83
Total_array_area = Average_res_PV_size*Number_of_PV_installed
Rated_cap_PV_array = Total_array_area*Module_eff
The_nominal_operating_cell_temp = (45+48)/2
Temp_coef_power = -0.0048
Max_power_point_eff_test_condition = 0.13 #The maximum power point efficiency under standard test conditions
Solar_radiation_NOCT = 0.8 #The solar radiation at which the NOCT is defined
Ambient_temp_NOCT = 20 #The ambient temperature at which the NOCT is defined
PV_cell_temp_std_test = 25
Target_temp_w_cooling = 80
Incident_radiation_at_std = 1
Degradation_rate = 0.005
PV_transmittance_cover_over_PV_array = 0.9 #The solar transmittance of any cover over the PV array
Solar_absorptance_PV = 0.9 #The solar absorptance of the PV array
System_degradation = Degradation_rate/(365*24*60)
PV_eff_at_max_power = Max_power_point_eff_test_condition #The efficiency of the PV array at its maximum power point
Solar_radiation_incident_on_PV_array_GHI = PV_incident_radiation*1000
coeff_heat_transfer_to_surrondings = Solar_absorptance_PV*PV_transmittance_cover_over_PV_array*Solar_radiation_NOCT/(The_nominal_operating_cell_temp-Ambient_temp_NOCT) #The coefficient of heat transfer to the surroundings
Temp_sens_analysis = 1
PV_cell_temp_in_current_time_step = (Ambient_temp*Temp_sens_analysis)+((Solar_radiation_incident_on_PV_array_GHI)/1000)*((Solar_absorptance_PV*PV_transmittance_cover_over_PV_array)/coeff_heat_transfer_to_surrondings)*(1-PV_eff_at_max_power/(Solar_absorptance_PV*PV_transmittance_cover_over_PV_array))
PV_cell_temp_under_cooling_in_current_time_step = pd.Series(np.where(PV_cell_temp_in_current_time_step>Target_temp_w_cooling,Target_temp_w_cooling,PV_cell_temp_in_current_time_step))
PV_sens = 1

PV_array_output = PV_sens*((Rated_cap_PV_array*PV_derating_factor*(((Solar_radiation_incident_on_PV_array_GHI))/Incident_radiation_at_std)*(1+Temp_coef_power*(PV_cell_temp_under_cooling_in_current_time_step-PV_cell_temp_std_test)))*Inverter_eff*(1-System_losses))*(1-System_degradation)**Time

PV_gen = PV_array_output/1000"""

##Model preparation
import pandas as pd
import numpy as np
import time

Dest = str(r'C:\Users\USNHIT\OneDrive - USNH\Roozbeh PhD research\Project 7 SML Water-Energy modeling\System Dynamics\SML Python SD Model')

Data = pd.read_csv(str(Dest)+'\Input Data.csv')
Power_Table = pd.read_csv(str(Dest)+'\Power table.csv')
Energy_Real = pd.read_csv(str(Dest)+'\Energy real data.csv')
Water_Real = pd.read_csv(str(Dest)+'\Water real data.csv')

#Decision Variables
n = 132480 #number of steps
RO_switch = np.ones(n)
Well_pump_switch = np.ones(n)

New_panels = 0
Additional_battery_kWh = 0

Wind_eff = 0.72

###############################################################################
#ME calibration with treatement of data!
#Parameter setup
n=96480

DataVsSim = pd.DataFrame(columns = ['SimSOC','SimPV',
                                 'SimGEN', 'RealSOC', 'RealPV', 'RealGEN','Treated_RealGEN'],
                                    index = pd.Series(range(0,n)))

ME_Results = pd.DataFrame(columns = ['Module efficiency','R2', 'MSE'], index = pd.Series(range(0,61)))


#Data treatment

DataVsSim.iloc[:,3] = Energy_Real.loc[:,"SOC (%)"]
DataVsSim.iloc[:,4] = Energy_Real.loc[:,"PV (kW) [ECB+Dorms]"]
DataVsSim.iloc[:,5] = Energy_Real.loc[:,"Diesel (kW)"]
DataVsSim.iloc[:,6] = Energy_Real.loc[:,"Diesel (kW)"]

# Generator data treatment: (Remove misgenerated data!)
print(DataVsSim.iloc[:,6].equals(DataVsSim.iloc[:,5]))
DataVsSim.loc[(DataVsSim['RealGEN'] > 0) & (DataVsSim['RealSOC'] >= 78), 'Treated_RealGEN' ] = 0
print(DataVsSim.iloc[:,6].equals(DataVsSim.iloc[:,5]))

#finding the best ME and treating PV data:
Counter = 0
start = time.time()
ME = 100
for ME in range(100, 401, 5):
    Module_eff = ME/1000
    exec(SML_energy_water)
    
    DataVsSim.iloc[:,1] = PV_gen
    
    DataVsSim_Filtered = DataVsSim[DataVsSim['RealSOC'] < 90]
    
    #PV generations accuracy
    x_value = DataVsSim_Filtered.iloc[:,1]
    y_value = DataVsSim_Filtered.iloc[:,4]
    y_value = y_value.squeeze()
    correlation_matrix = np.corrcoef(x_value, y_value)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    MSE = np.square(np.subtract(x_value,y_value)).mean()
    print('ME:',ME/1000,'R2:',r_squared, 'MSE:', MSE)
    
    ME_Results.iloc[Counter,0] = ME/1000
    ME_Results.iloc[Counter,1] = r_squared
    ME_Results.iloc[Counter,2] = MSE
    Counter=Counter+1


DataVsSim_Filtered.iloc[0:5*1440,1].plot()
DataVsSim_Filtered .iloc[0:5*1440,4].plot()

end = time.time()
print("The time of execution of above program is :", (end-start), "s")
    
ME_Results.to_csv(str(Dest)+'\Calibration results\ME_results_treated_PV_Data.csv', index = True)

print("Calibrated module efficiency",ME_Results.iloc[ME_Results[ME_Results['MSE'] == min(ME_Results.iloc[:,2])].index[0],0])

