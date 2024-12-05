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

PV_gen = PV_array_output/1000

#Stocks
Cumulative_PV_gen_ini = pd.Series([0]) #initial value
Cumulative_PV_gen_kWh = np.cumsum(pd.concat([Cumulative_PV_gen_ini,PV_array_output/1000/60], ignore_index = True))[:-1]

#########Demand Model##########################################################
###Variables
Demand = Data.loc[0:n-1,'Demand (W)']
Demand_sens = 1

#flows
Island_demand_load_kW = Demand*Demand_sens/1000

#Stocks
Cumulative_island_demand_ini = pd.Series([0]) #initial value
Cumulative_island_demand = np.cumsum(pd.concat([Cumulative_island_demand_ini,Island_demand_load_kW], ignore_index = True))
Cumulative_island_demand_kWh = Cumulative_island_demand/60

#Cumulative_island_demand_kWh.plot()

#########Diesel gen and BOS Models#############################################
###Variables


DCtoAC_eff = 0.925
Battery_total_cap_kWh = Old_batteery_kWh+Additional_battery_kWh
Battery_total_cap_kWm = Battery_total_cap_kWh*60
Initial_batt_percentage = 78 #%
Battery_ini_SOC = Battery_total_cap_kWm*Initial_batt_percentage/100
Diesel_gen_cap = 2*27
Gen_lower_bound = Battery_total_cap_kWm*0.70
Gen_upper_bound = Battery_total_cap_kWm*0.77

if Battery_total_cap_kWm==0: 
    Excess_gen_production=0 
else: Excess_gen_production=0.32


Energy_gain_from_gen = [None] * n
Excess_energy = [None] * n
Consumption_from_DC_BUS = [None] * n
DC_BUS = [None] * (n+1)
Previous_gen_switch = [None] * (n+1)
Current_gen_switch = [None] * n
Cumulative_diesel_elec_generation_kWh = pd.Series([0])
Gen_counter = [None] * n
Diesel_electricity_generation = [None] * n

## Defining initial states for the feedback loops
Previous_gen_switch[0] = 0
DC_BUS[0] = Battery_ini_SOC

#########Water Model###########################################################
#Decision Variables
RO_switch = np.ones(n)
Well_pump_switch = np.ones(n)

#Demand
SML_water_demand = Data.loc[0:n-1,'Demand flowRate (gal/min)']

#Stocks
Cistern_Tank = np.zeros(n+1)
Pressure_Tank = np.zeros(n+1)
Ground_Water = np.zeros(n+1)
Total_chlorine_consumption = np.zeros(n+1)

#Stocks size
Cistern_tank_size = 500
Pressure_tank_withdraw_size = 100

#Water generation
RO_desalination_rate = np.ones(n) * 5
RO_water_flow_rate = RO_desalination_rate * RO_switch

Well_water_withdrawal_rate = np.ones(n) * 14.5
Well_water_flow_rate = Well_water_withdrawal_rate * Well_pump_switch

Water_flow_to_pressure_tank = np.zeros(n)

#Excess water
Excess_water = np.zeros(n)

#Cistern to pressure tank and pressure tank
Cistern_pump_switch = np.ones(n)
Cistern_pump_flow_rate = np.ones(n) * 10

#Model Run
for i in range(0,n):
  
    #Diesel gen
    if Previous_gen_switch[i] == 1 and DC_BUS[i]<=Gen_upper_bound:
        Current_gen_switch[i] = 1
    elif DC_BUS[i]<=Gen_lower_bound:
        Current_gen_switch[i] = 1
    else:
        Current_gen_switch[i] = 0
        
    Previous_gen_switch[i+1] = Current_gen_switch[i]
    
    if Diesel_gen_cap >= ((1+Excess_gen_production)*(Demand[i]*Demand_sens/1000)) and Current_gen_switch[i] == 1:
        Diesel_electricity_generation[i] = ((1+Excess_gen_production)*(Demand[i]*Demand_sens/1000))
    else:
        Diesel_electricity_generation[i] = Current_gen_switch[i]*Diesel_gen_cap

    if Current_gen_switch[i] == 1 and Previous_gen_switch[i] == 0:
        Gen_counter[i] = 1
    else:
        Gen_counter[i] = 0
    
    ###BOS
    Energy_gain_from_gen[i] = Diesel_electricity_generation[i]*(Excess_gen_production)/(1+Excess_gen_production)
    Consumption_from_DC_BUS[i] = -(Current_gen_switch[i]-1)*Demand[i]*Demand_sens*DCtoAC_eff/1000 #Demand_sens?
    
    
    if (PV_gen[i] + Wind_gen[i] + Energy_gain_from_gen[i] - Consumption_from_DC_BUS[i])<0 and Battery_total_cap_kWm==0:
        Excess_energy[i] = PV_gen[i] + Wind_gen[i] + Energy_gain_from_gen[i] - Consumption_from_DC_BUS[i]
    elif (Battery_total_cap_kWm-DC_BUS[i])>(PV_gen[i] + Wind_gen[i] + Energy_gain_from_gen[i] - Consumption_from_DC_BUS[i]):
        Excess_energy[i] = 0
    else:
        Excess_energy[i] = (PV_gen[i] + Wind_gen[i] + Energy_gain_from_gen[i] - Consumption_from_DC_BUS[i])-(Battery_total_cap_kWm-DC_BUS[i])
        
    
    DC_BUS[i+1] = DC_BUS[i] + PV_gen[i] + Wind_gen[i] + Energy_gain_from_gen[i] - Excess_energy[i] - Consumption_from_DC_BUS[i]
    
    ##Water model
    if Pressure_Tank[i] + Cistern_pump_flow_rate[i] - SML_water_demand[i] <= Pressure_tank_withdraw_size:
        Cistern_pump_switch[i] = 1
    else:
        Cistern_pump_switch[i] = 0
    
    Pressure_Tank[i+1] = Pressure_Tank[i]+Cistern_pump_flow_rate[i]*Cistern_pump_switch[i]-SML_water_demand[i]
    
    Water_flow_to_pressure_tank[i] = Cistern_pump_flow_rate[i] * Cistern_pump_switch[i]
    
    if Cistern_Tank[i] + RO_water_flow_rate[i] + Well_water_flow_rate[i] - Water_flow_to_pressure_tank[i] <= Cistern_tank_size:
        Cistern_Tank[i+1] = Cistern_Tank[i] + RO_water_flow_rate[i] + Well_water_flow_rate[i] - Water_flow_to_pressure_tank[i]
    else:
        Excess_water[i] = Cistern_Tank[i] + RO_water_flow_rate[i] + Well_water_flow_rate[i] - Water_flow_to_pressure_tank[i] - Cistern_tank_size
        Cistern_Tank[i+1] = Cistern_tank_size

Gen_total_switch =  np.cumsum(Gen_counter)
DC_BUS = DC_BUS[:-1]

Battery_SOC_per_current = pd.Series(DC_BUS)*100/Battery_total_cap_kWm
Battery_SOC_per_previous = pd.concat([pd.Series(Initial_batt_percentage),pd.Series(Battery_SOC_per_current)], ignore_index = True)

Charge_Cycles = abs(Battery_SOC_per_current-Battery_SOC_per_previous)/(2*100)
Number_of_charge_cycles_cum = np.cumsum(Charge_Cycles)


#Diesel generation and BOS Stocks
Cumulative_diesel_elec_generation_kWh = np.cumsum(pd.Series(Diesel_electricity_generation)/60)
Cumulative_gen_counter = np.cumsum(Gen_counter)
Actual_consumption_from_DCBUS = pd.Series(Consumption_from_DC_BUS)*DCtoAC_eff
Cumulative_energy_consumption_kWh = np.cumsum((Actual_consumption_from_DCBUS+Diesel_electricity_generation)/60)

##Water model
Cistern_Tank = Cistern_Tank[:-1]
Pressure_Tank = Pressure_Tank[:-1]

#Groundwater
Groundwater_charge = Excess_water
Groundwater_discharge = Well_water_flow_rate

Ground_Water[1:] = Ground_Water[:-1] + Groundwater_charge - Groundwater_discharge
Ground_Water = np.cumsum(Ground_Water)
Ground_Water = Ground_Water[:-1]

#Treatment
Chlorine_rate_RO = np.ones(n) * 2
Chlorine_rate_well = np.ones(n) * 2

Chlorine_RO = RO_water_flow_rate * Chlorine_rate_RO / 1000000
Chlorine_Well = Well_water_flow_rate * Chlorine_rate_well / 1000000


Total_chlorine_consumption[1:] = Total_chlorine_consumption[:-1] + Chlorine_RO + Chlorine_Well
Total_chlorine_consumption = np.cumsum(Total_chlorine_consumption)
Total_chlorine_consumption = Total_chlorine_consumption[:-1]"""

##Model preparation
import pandas as pd
import numpy as np
import time

Dest = str(r'C:\Users\rg1070\OneDrive - University of New Hampshire\Roozbeh PhD research\Project 7 SML Water-Energy modeling\System Dynamics\SML Python SD Model')

Data = pd.read_csv(str(Dest)+'\Input Data.csv')
Power_Table = pd.read_csv(str(Dest)+'\Power table.csv')
Energy_Real = pd.read_csv(str(Dest)+'\Energy real data.csv')
Water_Real = pd.read_csv(str(Dest)+'\Water real data.csv')

#Decision Variables
n = 96480 #number of steps
RO_switch = np.ones(n)
Well_pump_switch = np.ones(n)

New_panels = 0
Old_batteery_kWh = 250
Additional_battery_kWh = 0

Module_eff = 0.155
Wind_eff = 0.72

#Model Run
start = time.time()
exec(SML_energy_water)
end = time.time()
print("The time of execution of above program is :", (end-start), "s")

###############################################################################
## Find caliberated capacity
Battery_cal = pd.DataFrame(columns = ['Battery Capacity', 'SOC MSE', 'SOC R2', 'Daily Gen MSE', 'Daily Gen R2'], 
                                     index = pd.Series(range(0,101)))
                                      
Data_daily=pd.DataFrame(columns = ['Date', 'Real diesel electricity generation', 'Simulation diesel electricity generation']
                                    , index = pd.Series(range(0,int(n/1440))))

DataVsSim = pd.DataFrame(columns = ['SimSOC', 'RealSOC'], index = pd.Series(range(0,n)))

DataVsSim.loc[:,'RealSOC'] = Energy_Real.loc[:,'SOC (%)']

#Writing dates in the dataset
for i in range(0,int(n/1440)):
    Data_daily.loc[i,'Date']=Energy_Real.loc[i*1440, 'Date']

x=0

#Writing real data daily diesel consumption in the dataset
for i in range(0,int(n/1440)):
    for j in range(0,1440):
        x=Energy_Real.loc[(i*1440)+j,'Diesel (kW)']+x
    Data_daily.iloc[i,1]=x/60
    x=0
    print(i+1)

#Writing sim data daily diesel consumption in the dataset
rowcounter = 0
x=0
for j in range(200, 301):
  Old_batteery_kWh = j
  exec(SML_energy_water)
  a=Diesel_electricity_generation
  print("Battery: ",j)
  #Converting minute based consumption to daily based kWh
  for k in range(0,int(n/1440)):
      for l in range(0,1440):
          x=a[(k*1440)+l]+x
      Data_daily.loc[k,'Simulation diesel electricity generation']=x/60 #Convert to kWh
      x=0

  #Model accuracy 
  x_value3 = Data_daily.loc[:,'Real diesel electricity generation'].astype(np.float64) #Real
  y_value3 = Data_daily.loc[:,'Simulation diesel electricity generation'].astype(np.float64) #Sim SOC
  y_value3 = y_value3.squeeze()
  correlation_matrix3 = np.corrcoef(x_value3, y_value3)
  correlation_xy3 = correlation_matrix3[0,1]
  r_squared3 = correlation_xy3**2
  MSE3 = np.square(np.subtract(x_value3,y_value3)).mean()
  
  DataVsSim.loc[:,'SimSOC'] = Battery_SOC_per_current

  #SOC comparison
  #Battery state of charge
  x_value1 = DataVsSim.loc[:,'RealSOC'] #Real SOC
  y_value1 = DataVsSim.loc[:,'SimSOC'] #Sim SOC
  y_value1 = y_value1.squeeze()
  correlation_matrix1 = np.corrcoef(x_value1, y_value1)
  correlation_xy1 = correlation_matrix1[0,1]
  r_squared1 = correlation_xy1**2
  MSE1 = np.square(np.subtract(x_value1,y_value1)).mean()
  
  #Writing the final dataset      
  Battery_cal.loc[rowcounter,'Battery Capacity'] = j
  Battery_cal.loc[rowcounter,'SOC MSE'] = MSE1
  Battery_cal.loc[rowcounter,'SOC R2'] = r_squared1
  Battery_cal.loc[rowcounter,'Daily Gen MSE'] = MSE3
  Battery_cal.loc[rowcounter,'Daily Gen R2'] = r_squared3
  rowcounter=rowcounter+1

Battery_cal.insert(1,'SOC MSE Score',1-(Battery_cal.loc[:,'SOC MSE']/max(Battery_cal.loc[:,'SOC MSE'])))
Battery_cal.insert(1,'SOC R2 Score',(Battery_cal.loc[:,'SOC R2']/max(Battery_cal.loc[:,'SOC R2'])))
Battery_cal.insert(1,'Daily Diesel MSE Score',1-(Battery_cal.loc[:,'Daily Gen MSE']/max(Battery_cal.loc[:,'Daily Gen MSE'])))
Battery_cal.insert(1,'Daily Diesel R2 Score',(Battery_cal.loc[:,'Daily Gen R2']/max(Battery_cal.loc[:,'Daily Gen R2'])))
Battery_cal.insert(1,'Total Score',Battery_cal.loc[:,'SOC MSE Score']+Battery_cal.loc[:,'SOC R2 Score']+Battery_cal.loc[:,'Daily Diesel MSE Score']+Battery_cal.loc[:,'Daily Diesel R2 Score'])


#Extracting final dataset
Battery_cal.to_csv(str(Dest)+'\Calibration results\Battery_calibration.csv', index = True)

