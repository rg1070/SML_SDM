import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Load csv files
Data22 = pd.read_csv('https://gitlab.com/roozbeh.ghasemi67/water-energy-rl/-/raw/main/Input%20Data%202022.csv?ref_type=heads')
Energy_Real22 = pd.read_csv('https://gitlab.com/roozbeh.ghasemi67/water-energy-rl/-/raw/main/Energy%20real%20data%202022.csv?ref_type=heads')
Water_Real22 = pd.read_csv('https://gitlab.com/roozbeh.ghasemi67/water-energy-rl/-/raw/main/Water%20real%20data%202022.csv?ref_type=heads')
Data23 = pd.read_csv('https://gitlab.com/roozbeh.ghasemi67/water-energy-rl/-/raw/main/Input%20Data%202023.csv?ref_type=heads')
Power_Table = pd.read_csv('https://gitlab.com/roozbeh.ghasemi67/water-energy-rl/-/raw/main/Power%20table.csv?ref_type=heads')
# Load the Excel file
His_Data = pd.ExcelFile('https://gitlab.com/roozbeh.ghasemi67/water-energy-rl/-/raw/main/Histocial%20data%20-%20Exogenous%20States.xlsx?ref_type=heads&inline=false')
###############################################################################
###SD Model - Modules
###############################################################################
###############################################################################
###############################################################################
###Time of the day and the year
#Input: time of the year or time of the day - Scalar
#Output: time of the year or time of the day - Scalar

def Time_Day(t,init_h=0): #0 means 12 am 1 means 12:01 anso on - minutes shift
    Civil_time= ((t-(1440*int(t/1440)))*(24/1440))+(init_h*(24/1440))
    return Civil_time


def Time_Year(t,init_day=152):
    Day_of_year = (init_day+(t/1440))//1
    return Day_of_year

###SML Energy Model
##1.PV Generation
#Input: Initial and Final Time Steps, Dataset
#Output: PV Generation Vector from Time t1=0 to t2 [kW]
def PV_Gen(t, GHI_W_m2, Ambient_temp, Day_of_year, Civil_time, PV_sens=1, Module_eff=0.155):
    #Inputs:
    Time = t
    Day_of_year = Time_Year(t)
    Local_standard_time = Time_Day(t)
    
    # Variables
    GHI_kW_m2 = 0 if GHI_W_m2 < 25 else GHI_W_m2 / 1000
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
    E = 3.82*(7.5e-005+0.001868*np.cos(B)-0.032077*np.sin(B) -
              0.014615*np.cos(2*B)-0.04089*np.sin(2*B))
    Solar_time_ts = Local_standard_time+(Longitude_r/15)-Time_zone+E
    Hour_angle_omega = (Solar_time_ts-12)*15/180
    Solar_declination_delta = 23.45 * \
        np.sin(((360*((284+Day_of_year)/365)))/180)/180  # radian
    Angle_of_incident_Cos_theta = np.sin(Solar_declination_delta)*np.sin(Latitute_Phi_r)*np.cos(Slope_of_surface_r)-np.sin(Solar_declination_delta)*np.cos(Latitute_Phi_r)*np.sin(Slope_of_surface_r)*np.cos(Azimuth_gamma_r)+np.cos(Solar_declination_delta)*np.cos(Latitute_Phi_r)*np.cos(
        Slope_of_surface_r)*np.cos(Azimuth_gamma_r)+np.cos(Solar_declination_delta)*np.sin(Latitute_Phi_r)*np.sin(Slope_of_surface_r)*np.cos(Azimuth_gamma_r)*np.cos(Hour_angle_omega)+np.cos(Solar_declination_delta)*np.sin(Slope_of_surface_r)*np.sin(Azimuth_gamma_r)*np.sin(Hour_angle_omega)
    Zenith_angle_Cos_theta_z = np.cos(Latitute_Phi_r)*np.cos(Solar_declination_delta)*np.cos(
        Hour_angle_omega)+np.sin(Latitute_Phi_r)*np.sin(Solar_declination_delta)
    The_extraterrestrial_normal_radiation = Solar_constant * \
        (1+0.033*np.cos(360*Day_of_year/365))
    # The ratio of beam radiation on the tilted surface to beam radiation on the horizontal surface Rb
    Rb = Angle_of_incident_Cos_theta/Zenith_angle_Cos_theta_z
    Extraterrestrial_h_radiation_avg = (12/3.1415)*The_extraterrestrial_normal_radiation*(np.cos(Latitute_Phi_r)*np.cos(Solar_declination_delta)*(np.sin(Hour_angle_omega+0.25)-np.sin(
        Hour_angle_omega))+(3.1415*0.25*np.sin(Latitute_Phi_r)*np.sin(Solar_declination_delta)/180))  # The extraterrestrial horizontal radiation averaged over the time step
    Kt = GHI_kW_m2/(Extraterrestrial_h_radiation_avg)  # Clearness index Kt
    
    Diff_less = GHI_kW_m2*(1-0.09*Kt) if Kt <= 0.22 else 0
    Diff_more = GHI_kW_m2*0.165 if Kt > 0.80 else 0
    Diff_between = GHI_kW_m2*(0.9511-0.1604*Kt+4.388*Kt**2-16.638*Kt**3+12.336*Kt**4) if Kt > 0.22 else 0
    Diff_between = 0 if Kt > 0.80 else Diff_between
     
    Diff_radiation = Diff_less+Diff_more+Diff_between
    Beam_radiation = GHI_kW_m2-Diff_radiation
    Ai = Beam_radiation/Extraterrestrial_h_radiation_avg  # Anisotropy index Ai
    # Cloudiness factor f
    f_Cloud = 0 if GHI_kW_m2==0 else np.sqrt(Beam_radiation/GHI_kW_m2)
        
    PV_incident_radiation = (Beam_radiation+Diff_radiation*Ai)*Rb+Diff_radiation*(1-Ai)*((1+np.cos(Slope_of_surface_r))/2)*(
        1+f_Cloud*np.sin(Slope_of_surface_r/2)**3)+GHI_kW_m2*Ground_ref_albedo*((1-np.cos(Slope_of_surface_r))/2)

    # PV generation
    Old_panels = 233
    New_panels = 0
    Number_of_PV_installed = Old_panels + New_panels
    
    
    Solar_radiation_incident_on_PV_array_GHI = PV_incident_radiation*1000
    Inverter_eff = 1
    PV_derating_factor = 0.95
    System_losses = 0.15
    Average_res_PV_size = 1.83
    Total_array_area = Average_res_PV_size*Number_of_PV_installed
    Rated_cap_PV_array = Total_array_area*Module_eff
    The_nominal_operating_cell_temp = (45+48)/2
    Temp_coef_power = -0.0048
    # The maximum power point efficiency under standard test conditions
    Max_power_point_eff_test_condition = 0.13
    Solar_radiation_NOCT = 0.8  # The solar radiation at which the NOCT is defined
    Ambient_temp_NOCT = 20  # The ambient temperature at which the NOCT is defined
    PV_cell_temp_std_test = 25
    Target_temp_w_cooling = 80
    Incident_radiation_at_std = 1
    Degradation_rate = 0.005
    # The solar transmittance of any cover over the PV array
    PV_transmittance_cover_over_PV_array = 0.9
    Solar_absorptance_PV = 0.9  # The solar absorptance of the PV array
    System_degradation = Degradation_rate/(365*24*60)
    # The efficiency of the PV array at its maximum power point
    PV_eff_at_max_power = Max_power_point_eff_test_condition
    Solar_radiation_incident_on_PV_array_GHI = PV_incident_radiation*1000
    coeff_heat_transfer_to_surrondings = Solar_absorptance_PV*PV_transmittance_cover_over_PV_array*Solar_radiation_NOCT / \
        (The_nominal_operating_cell_temp -
         Ambient_temp_NOCT)  # The coefficient of heat transfer to the surroundings
    Temp_sens_analysis = 1
    PV_cell_temp_in_current_time_step = (Ambient_temp*Temp_sens_analysis)+((Solar_radiation_incident_on_PV_array_GHI)/1000)*(
        (Solar_absorptance_PV*PV_transmittance_cover_over_PV_array)/coeff_heat_transfer_to_surrondings)*(1-PV_eff_at_max_power/(Solar_absorptance_PV*PV_transmittance_cover_over_PV_array))
    
    PV_cell_temp_under_cooling_in_current_time_step = Target_temp_w_cooling if PV_cell_temp_in_current_time_step > Target_temp_w_cooling else PV_cell_temp_in_current_time_step
    
    PV_array_output = PV_sens*((Rated_cap_PV_array*PV_derating_factor*(((Solar_radiation_incident_on_PV_array_GHI))/Incident_radiation_at_std)*(
        1+Temp_coef_power*(PV_cell_temp_under_cooling_in_current_time_step-PV_cell_temp_std_test)))*Inverter_eff*(1-System_losses))*(1-System_degradation)**Time

    PV_gen = PV_array_output/1000
    
    return PV_gen

##2.Wind Generation
#Input: Initial and Final Time Steps, Input Dataset, Power Table
#Output: Wind Generation Vector from Time t1=0 to t2 [kW]
def Wind_Gen(Wind_spd_IOSN3 , Power_Table, Wind_sens = 1, Wind_eff = 0.72):
    # Variables
    Height_conversion = 1.017
    Wind_spd_adjusted = round(Wind_spd_IOSN3*Height_conversion, 1)
    
    if Wind_spd_adjusted>20.5:
        Wind_power = 11500.0
    else:
        row = Power_Table[Power_Table["Wind"] == Wind_spd_adjusted]
        
        Wind_power = row["Power"].values[0] if not row.empty else None
    
    Wind_gen = Wind_power/1000

    return Wind_gen

##3.Diesel Generation
#Input: Previous Gen Switch, DC BUS Energy [kWm], Total_Demand [W]
#Output: Diesel Generation [kW], Current Gen Switch (0 or 1), Excess Gen (o or 0.32) - 3 Scalars

def Diesel_Gen(Gen_switch, DC_BUS, Total_Demand, PV_gen, Wind_gen, Battery_total_cap_kWh=250 , Gen_upper_bound=0.78, Gen_lower_bound=0.70, Diesel_gen_cap=2*27):
    # Diesel gen
    
    Battery_total_cap_kWm = Battery_total_cap_kWh*60
    SOC = DC_BUS/Battery_total_cap_kWm
    
    if 0<=SOC<=Gen_lower_bound and Gen_switch == 1 and Total_Demand/1000<(PV_gen+Wind_gen) and (Gen_upper_bound-SOC)*Battery_total_cap_kWm<(PV_gen+Wind_gen-Total_Demand/1000):
        Gen_switch = 0
    elif 0<=SOC<=Gen_lower_bound and Gen_switch == 1 and Total_Demand/1000<(PV_gen+Wind_gen) and (Gen_upper_bound-SOC)*Battery_total_cap_kWm>=(PV_gen+Wind_gen-Total_Demand/1000):
        Gen_switch = 1
    elif 0<=SOC<=Gen_lower_bound and Gen_switch == 1 and Total_Demand/1000>=(PV_gen+Wind_gen):
        Gen_switch = 1
        
    elif 0<=SOC<=Gen_lower_bound and Gen_switch == 0 and Total_Demand/1000<(PV_gen+Wind_gen) and (Gen_lower_bound-SOC)*Battery_total_cap_kWm<(PV_gen+Wind_gen-Total_Demand/1000):
        Gen_switch = 0
    elif 0<=SOC<=Gen_lower_bound and Gen_switch == 0 and Total_Demand/1000<(PV_gen+Wind_gen) and (Gen_lower_bound-SOC)*Battery_total_cap_kWm>=(PV_gen+Wind_gen-Total_Demand/1000):
        Gen_switch = 1
    elif 0<=SOC<=Gen_lower_bound and Gen_switch == 0 and Total_Demand/1000>=(PV_gen+Wind_gen):
        Gen_switch = 1
    
    
    
    
    elif Gen_lower_bound<SOC<Gen_upper_bound and Gen_switch == 1 and Total_Demand/1000<(PV_gen+Wind_gen) and (Gen_upper_bound-SOC)*Battery_total_cap_kWm<(PV_gen+Wind_gen-Total_Demand/1000):
        Gen_switch = 0
    elif Gen_lower_bound<SOC<Gen_upper_bound and Gen_switch == 1 and Total_Demand/1000<(PV_gen+Wind_gen) and (Gen_upper_bound-SOC)*Battery_total_cap_kWm>=(PV_gen+Wind_gen-Total_Demand/1000):
        Gen_switch = 1
    elif Gen_lower_bound<SOC<Gen_upper_bound and Gen_switch == 1 and Total_Demand/1000>=(PV_gen+Wind_gen):
        Gen_switch = 1
    
    elif Gen_lower_bound<SOC<Gen_upper_bound and Gen_switch == 0 and Total_Demand/1000<(PV_gen+Wind_gen):
        Gen_switch = 0
    elif Gen_lower_bound<SOC<Gen_upper_bound and Gen_switch == 0 and Total_Demand/1000>=(PV_gen+Wind_gen) and (SOC-Gen_lower_bound)*Battery_total_cap_kWm<(Total_Demand/1000-(PV_gen+Wind_gen)):
        Gen_switch = 1
    elif Gen_lower_bound<SOC<Gen_upper_bound and Gen_switch == 0 and Total_Demand/1000>=(PV_gen+Wind_gen) and (SOC-Gen_lower_bound)*Battery_total_cap_kWm>=(Total_Demand/1000-(PV_gen+Wind_gen)):
        Gen_switch = 0



    elif Gen_upper_bound <=SOC and Gen_switch == 1 and Total_Demand/1000<(PV_gen+Wind_gen):
        Gen_switch = 0
    elif Gen_upper_bound <=SOC and Gen_switch == 1 and Total_Demand/1000>=(PV_gen+Wind_gen) and (SOC-Gen_lower_bound)*Battery_total_cap_kWm>=(Total_Demand/1000-(PV_gen+Wind_gen)):
        Gen_switch = 0
    elif Gen_upper_bound <=SOC and Gen_switch == 1 and Total_Demand/1000>=(PV_gen+Wind_gen) and (SOC-Gen_lower_bound)*Battery_total_cap_kWm<(Total_Demand/1000-(PV_gen+Wind_gen)):
        Gen_switch = 1
    
    elif Gen_upper_bound <=SOC and Gen_switch == 0 and Total_Demand/1000<(PV_gen+Wind_gen):
        Gen_switch = 0
    elif Gen_upper_bound <=SOC and Gen_switch == 0 and Total_Demand/1000>=(PV_gen+Wind_gen) and (SOC-Gen_lower_bound)*Battery_total_cap_kWm<(Total_Demand/1000-(PV_gen+Wind_gen)):
        Gen_switch = 1
    elif Gen_upper_bound <=SOC and Gen_switch == 0 and Total_Demand/1000>=(PV_gen+Wind_gen) and (SOC-Gen_lower_bound)*Battery_total_cap_kWm>=(Total_Demand/1000-(PV_gen+Wind_gen)):
        Gen_switch = 0
    
    else:
        Gen_switch = 10
    
    
    if Battery_total_cap_kWm == 0:
        Excess_gen_production = 0
    else:
        Excess_gen_production = 0.32

    if Diesel_gen_cap >= ((1+Excess_gen_production)*(Total_Demand/1000)) and Gen_switch == 1:
        Diesel_electricity_generation = ((1+Excess_gen_production)*(Total_Demand/1000))
    else:
        Diesel_electricity_generation = Gen_switch*Diesel_gen_cap
    
    return Diesel_electricity_generation, Gen_switch, Excess_gen_production

#Diesel_Gen(0, 10590, 6.4) #Test-Output: Two Scalars
##4.Balance of System
#Input: Previous Gen Switch, DC BUS Energy [kWm], Total_Demand [W]
#Output: Diesel Generation [kW], Current Gen Switch (0 or 1), Excess Gen production - 3 Scalars

def BoSys(DC_BUS, PV_gen, Wind_gen, Diesel_electricity_generation, Total_Demand, Excess_gen_production, Gen_switch, DCtoAC_eff=0.925, Battery_total_cap_kWh=250):
    # BOS
    Battery_total_cap_kWm = Battery_total_cap_kWh*60
    
    Energy_gain_from_gen = Diesel_electricity_generation * \
        (Excess_gen_production)/(1+Excess_gen_production)
    Consumption_from_DC_BUS = -(Gen_switch-1) * \
        Total_Demand*DCtoAC_eff/1000  # Demand_sens?

    if (PV_gen + Wind_gen + Energy_gain_from_gen - Consumption_from_DC_BUS) < 0 and Battery_total_cap_kWm == 0:
        Excess_energy = PV_gen + Wind_gen + Energy_gain_from_gen - Consumption_from_DC_BUS
        Excess_energy_signal = 1
    elif (Battery_total_cap_kWm-DC_BUS) > (PV_gen + Wind_gen + Energy_gain_from_gen - Consumption_from_DC_BUS):
        Excess_energy = 0
        Excess_energy_signal = 0
    else:
        Excess_energy = (PV_gen + Wind_gen + Energy_gain_from_gen-Consumption_from_DC_BUS)-(Battery_total_cap_kWm-DC_BUS)
        Excess_energy_signal = 1
    
    DC_BUS = DC_BUS + PV_gen + Wind_gen + Energy_gain_from_gen - Excess_energy - Consumption_from_DC_BUS

    Battery_SOC = DC_BUS*100/Battery_total_cap_kWm

    return DC_BUS, Battery_SOC, Excess_energy, Excess_energy_signal

#BoSys(14360, 0.6, 1.26, 6.4*1.32, 6.4, 6.4*0.32, 1) #Test-Output: Four Scalars

###############################################################################
###SML Water Model

##1.Desalination
#Input: RO Switch (0 or 1)
#Output: Ro Flow Rate [gal] - Scalar

def RO(RO_switch, RO_desalination_rate=3):
    
    RO_water_flow=RO_desalination_rate * RO_switch
    
    return RO_water_flow

##2.Well Water
#Input: Well Switch (0 or 1)
#Output: Well Water Flow Rate [gal] - Scalar

def Well(Well_pump_switch, Well_rate = 14.5):
    
    Well_water_flow = Well_rate * Well_pump_switch
    
    return Well_water_flow

##3.Pressure Tank
#Input: P Tank Level [gal], Water  Demand [gal]
#Output: Updated presure Tank Level, Excess water [gal], Water Shortage Signal (0 or 1) - 3 Scalars

def PTank(Pressure_Tank, Cistern_Tank, WDemand, Cistern_pump_switch, Cistern_pump_flow_rate=14.5, Pressure_tank_withdraw_size=5000, P_tank_up = 0.8, P_tank_low = 0.2, Water_shortage=0):
  
    if Pressure_Tank <=0:
        if Cistern_Tank>=Cistern_pump_flow_rate:
            Cistern_pump_switch = 1
            Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
            Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
            Water_shortage = 0
        else:
            Cistern_pump_switch = 0
            Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
            Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
            Pressure_Tank = 0
            Water_shortage= -Pressure_Tank
        
    elif Pressure_Tank <=P_tank_low*Pressure_tank_withdraw_size:
        if Cistern_Tank>=Cistern_pump_flow_rate:
            Cistern_pump_switch = 1
            Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
            Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
            Water_shortage = 0
        else:
            Cistern_pump_switch = 0
            Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
            Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
            if Pressure_Tank <=0:
                Pressure_Tank = 0
                Water_shortage= -Pressure_Tank
             
    elif P_tank_low*Pressure_tank_withdraw_size<= Pressure_Tank <=P_tank_up*Pressure_tank_withdraw_size:
        if Cistern_pump_switch == 0:
            Cistern_pump_switch = 0
            Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
            Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
            Water_shortage = 0
            
        elif Cistern_pump_switch == 1:
            if Cistern_Tank>=Cistern_pump_flow_rate:
                Cistern_pump_switch = 1
                Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
                Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
                Water_shortage = 0
            else:
                Cistern_pump_switch = 0
                Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
                Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
                Water_shortage = 0
            
    elif P_tank_up*Pressure_tank_withdraw_size<= Pressure_Tank:
        Cistern_pump_switch = 0
        Water_flow_to_pressure_tank = Cistern_pump_flow_rate*Cistern_pump_switch
        Pressure_Tank = Pressure_Tank+Water_flow_to_pressure_tank-WDemand
        Water_shortage = 0
        
    else:
        Cistern_pump_switch = 9999
        Water_flow_to_pressure_tank = 9999
        Pressure_Tank = 9999
        Water_shortage = 9999
  
    
  
 
    return Cistern_pump_switch, Pressure_Tank, Water_flow_to_pressure_tank, Water_shortage

##4.Cistern Tank
#Input: Cistern Tank Level [gal], Well and RO Flow [gal], Water Flow To P Tank [gal]
#Output: Updated Cistern Tank Level, Excess water [gal], Water Shortage Signal (0 or 1) - 3 Scalars

def Cistern(Cistern_Tank, Well_water_flow, RO_water_flow, Water_flow_to_pressure_tank, Cistern_tank_size=14000):
    
    if 0 <= Cistern_Tank + RO_water_flow + Well_water_flow -\
        Water_flow_to_pressure_tank <= Cistern_tank_size:
        
        Excess_water = 0

        Cistern_Tank = Cistern_Tank + RO_water_flow + \
            Well_water_flow - Water_flow_to_pressure_tank
        
    elif Cistern_Tank + RO_water_flow + Well_water_flow -\
        Water_flow_to_pressure_tank >= Cistern_tank_size:
        
        Excess_water = Cistern_Tank + RO_water_flow + Well_water_flow -\
            Water_flow_to_pressure_tank-Cistern_tank_size

        Cistern_Tank = Cistern_tank_size
        
    else:
        
        Excess_water = 0
        Cistern_Tank = 0
        
    return Cistern_Tank, Excess_water



##6.Water Treatment
#Input: RO and Well Water Flow [gal]
#Output: Chlorine for Well and RO Water [gal] - 2 Scalars

def Treat(RO_water_flow, Well_water_flow,Chlorine_rate_RO=2, Chlorine_rate_well=2):
    
    Chlorine_RO = RO_water_flow * Chlorine_rate_RO / 1000000
    Chlorine_Well = Well_water_flow * Chlorine_rate_well / 1000000
    
    if Chlorine_RO>0:
        Ch_RO_E_Sig = 1
    else: Ch_RO_E_Sig = 0
    
    if Chlorine_Well>0:
        Ch_Well_E_Sig = 1
    else: Ch_Well_E_Sig = 0
    
    return Chlorine_RO, Chlorine_Well, Ch_RO_E_Sig, Ch_Well_E_Sig

##7.Well Height
#Input: Well Height [m], Excess Water and Well Water Flow [gal], Catchment area [ft^2], Average reservoir area [ft^2], rainfall [in]
#Output: Updated Well Height [m] - Scalar

def GroundWater(H_Well, Excess_water, Well_water_flow, R , A=1097):    
    
    Precipitation = R*(1/12)*A*7.48 #1ft = 12in, 1ft3 = 7.48 gal 
    
    Groundwater_charge = Excess_water+Precipitation
    Groundwater_discharge = Well_water_flow

    H_Well = H_Well + (Groundwater_charge - Groundwater_discharge)/(A*7.48/12)
    
    return H_Well
###############################################################################
###SML Demand Model
##1.Total Energy Demand
#Input: Demand Data [W], time step [min], Water Flow of RO/Well/to Pressure Tank [gal]
#Output: Total Energy Demand at Time t [W] - Scalar

def E_Demand(t, RO_switch, Well_pump_switch, Water_flow_to_pressure_tank, Chlorine_RO, Chlorine_Well, Demand, Edemand_sens=1, RO_power = 3370, Well_pump_power = 1050, Seawater_intake_power = 1050, Cistern_pump_power = 1050, RO_treat_power = 1, Well_treat_power = 1):
    
    
    
    #Water Energy Demand
    Seawater_totalP = Seawater_intake_power*RO_switch
    RO_totalP = RO_power*RO_switch
    Well_totalP = Well_pump_power*Well_pump_switch
    
    Cistern_totalP = Cistern_pump_power*Water_flow_to_pressure_tank/14.5
    
    RO_treat_totalP = RO_treat_power*Chlorine_RO
    Well_treat_totalP = Well_treat_power*Chlorine_Well
    
    #Total Energy Demand
    Total_Demand = Demand+Seawater_totalP+RO_totalP+Well_totalP+Cistern_totalP+RO_treat_totalP+Well_treat_totalP
    
    return Total_Demand

###############################################################################
###SML Model Feedback:
#Cost Score
#Constants:
Dep_SeaPump_m = 0.000875983
Dep_CisPump_m = 0.000240487
Dep_WllPump_m = 0.000221969
Dep_Gen_m = 0.016666667
Dep_RO_m = 0.00152207
P_Diesel = 5.39
Battery_total_cap_kWh=250 #Should I recalibrate for 2022? Constant loss?
P_Batt = 875*Battery_total_cap_kWh
f=0.0903/60 #gallons per kWm
Max_Ren_Gen_kWm = 36.67+12.56

Max_Cost = (max(Data22.loc[:, 'Demand (W)'])+1050+3370+1050+1050+1+1)*1.32*f*P_Diesel+Max_Ren_Gen_kWm*f*P_Diesel+Dep_SeaPump_m+Dep_CisPump_m+Dep_WllPump_m+Dep_Gen_m+Dep_RO_m+P_Batt*(max(Data22.loc[:, 'Demand (W)'])*100/(250*60000))/200

#Sustainability score
CF_SeaPump_m = 0.836*3.37/60
CF_CisPump_m = 0.836*3.37/60
CF_WllPump_m = 0.836*3.37/60
CF_Gen_m = 0.31 # Does this include diesel consumption? Ifg yes I have to exclude diesel carbon intensity
CF_RO_m = 7.30594E-07
CF_Diesel_kWm = 1.186396/60 # per kWmin of diesel electricity generation
Battery_total_cap_kWh=250 
CF_Batt = 92.5*Battery_total_cap_kWh

Max_CF = (max(Data22.loc[:, 'Demand (W)'])+1050+3370+1050+1050+1+1)*1.32*f*CF_Diesel_kWm+Max_Ren_Gen_kWm*f*P_Diesel+CF_SeaPump_m+CF_CisPump_m+CF_WllPump_m+CF_Gen_m+CF_RO_m+CF_Batt*(max(Data22.loc[:, 'Demand (W)'])*100/(250*60000))/200

#Resilience score
Lowest_H_Well = 100
Highest_H_Well = 300

#Cost score values:
def CostScore(RO_switch, Well_pump_switch, Gen_switch, Cistern_pump_switch,  Diesel_electricity_generation, Excess_energy, Battery_SOC, Battery_SOC_p, f=f, P_Diesel=P_Diesel, Dep_SeaPump_m=Dep_SeaPump_m ,Dep_CisPump_m=Dep_CisPump_m ,Dep_WllPump_m=Dep_WllPump_m ,Dep_Gen_m=Dep_Gen_m , Dep_RO_m=Dep_RO_m ,P_Batt=P_Batt ,Max_Cost=Max_Cost ):
    
    Score_C = 1-(Diesel_electricity_generation*f*P_Diesel+Excess_energy*f*P_Diesel+Dep_SeaPump_m*RO_switch+Dep_CisPump_m*Cistern_pump_switch+Dep_WllPump_m*Well_pump_switch+Dep_Gen_m*Gen_switch+Dep_RO_m*RO_switch+P_Batt*abs(Battery_SOC-Battery_SOC_p)/200)/Max_Cost
    
    return Score_C

#Sustainability Score:
def SusScore(RO_switch, Well_pump_switch, Gen_switch, Cistern_pump_switch,  Diesel_electricity_generation, Excess_energy, Battery_SOC, Battery_SOC_p, f=f, CF_Diesel_kWm=CF_Diesel_kWm, CF_SeaPump_m=CF_SeaPump_m, CF_CisPump_m=CF_CisPump_m, CF_WllPump_m=CF_WllPump_m, CF_Gen_m=CF_Gen_m, CF_RO_m=CF_RO_m, CF_Batt=CF_Batt, Max_CF=Max_CF):
    
    Score_S = 1-(Diesel_electricity_generation*f*CF_Diesel_kWm+Excess_energy*f*CF_Diesel_kWm+CF_SeaPump_m*RO_switch+CF_CisPump_m*Cistern_pump_switch+CF_WllPump_m*Well_pump_switch+CF_Gen_m*Gen_switch+CF_RO_m*RO_switch+CF_Batt*abs(Battery_SOC-Battery_SOC_p)/200)/Max_CF
    
    return Score_S


#Reliability Score:
def ReliScore(H_Well, Highest_H_Well=Highest_H_Well, Lowest_H_Well=Lowest_H_Well):
    
    Score_R = (H_Well-Lowest_H_Well)/(Highest_H_Well-Lowest_H_Well)
    
    return Score_R

###############################################################################
###SD Model - Run
###############################################################################
###Run
def SML(N, RO_switch_v, Well_pump_switch_v,    WDemand_v, EDemand_v, GHI_W_m2_v, Ambient_temp_v, Wind_spd_IOSN3_v, R_v, Power_Table=Power_Table, Cistern_pump_switch = 0, Cistern_Tank = 14000, Pressure_Tank = 4000, Battery_SOC = 78, Gen_switch = 0, H_Well = 188.8, Battery_total_cap_kWh=250, init_day=152,init_h=0):
    
    #Initial value correction
    DC_BUS = Battery_SOC*Battery_total_cap_kWh*60/100
   
    #Making results Series
    PV_gen_ = []
    Wind_gen_ = []
    Diesel_electricity_generation_ = []
    Excess_energy_ = []
    Excess_energy_signal_ = []
    Battery_SOC_ = []
    Cistern_Tank_ = []
    Pressure_Tank_ = []
    H_Well_ = []
    Gen_switch_ = []
    Water_shortage_ = []
    Excess_water_ = []
    Total_EDemand_ = []
    DC_BUS_ = []
    Cistern_pump_switch_=[]
    
    RO_water_flow_ = []
    Well_water_flow_ = []
    Water_flow_to_pressure_tank_ = []
    Water_Demand_ = []
    
    Score_C_ = []
    Score_S_ = []
    Score_R_ = []
    
    RO_switch_ = []
    Well_switch = []
    
    #Store initial values in results
    Battery_SOC_.append(Battery_SOC)#
    DC_BUS_.append(DC_BUS)#
    Cistern_Tank_.append(Cistern_Tank)#
    Pressure_Tank_.append(Pressure_Tank)#
    H_Well_.append(H_Well)#
    Gen_switch_.append(Gen_switch)#
    Cistern_pump_switch_.append(Cistern_pump_switch)#
    
    #Time Dependent Variables
    for t in tqdm(range(0,N), desc='SD Model Simulation: ', unit='Steps'):
        ##Input states (Vectors to scalars)
        RO_switch = RO_switch_v[t]
        Well_pump_switch = Well_pump_switch_v[t]
        WDemand = WDemand_v[t]
        EDemand = EDemand_v[t]
        GHI_W_m2 = GHI_W_m2_v[t]
        Ambient_temp = Ambient_temp_v[t]
        Wind_spd_IOSN3 = Wind_spd_IOSN3_v[t]
        R = R_v[t]
       
        #Energy water model
        Civil_time = Time_Day(t,init_h=init_h)
        Day_of_year = Time_Year(t,init_day=init_day)
        PV_gen = PV_Gen(t, GHI_W_m2, Ambient_temp, Day_of_year, Civil_time)
        Wind_gen = Wind_Gen(Wind_spd_IOSN3 , Power_Table)
        
        RO_water_flow = RO(RO_switch)
        Well_water_flow = Well(Well_pump_switch)
        Cistern_pump_switch, Pressure_Tank, Water_flow_to_pressure_tank, Water_shortage = PTank(Pressure_Tank, Cistern_Tank, WDemand, Cistern_pump_switch)
        Cistern_Tank, Excess_water = Cistern(Cistern_Tank, Well_water_flow, RO_water_flow, Water_flow_to_pressure_tank)
        Chlorine_RO, Chlorine_Well, Ch_RO_E_Sig, Ch_Well_E_Sig = Treat(RO_water_flow, Well_water_flow)
        H_Well = GroundWater(H_Well, Excess_water, Well_water_flow, R)
        Total_EDemand = E_Demand(t, RO_switch, Well_pump_switch, Water_flow_to_pressure_tank, Ch_RO_E_Sig, Ch_Well_E_Sig, EDemand)
        
        #Diesel_electricity_generation, Gen_switch, Excess_gen_production = Diesel_Gen(Gen_switch, DC_BUS, Total_EDemand)
        Battery_SOC_p = DC_BUS*100/(250*60)
        
        Diesel_electricity_generation, Gen_switch, Excess_gen_production = Diesel_Gen(Gen_switch, DC_BUS, Total_EDemand, PV_gen, Wind_gen)
        DC_BUS, Battery_SOC, Excess_energy, Excess_energy_signal = BoSys(DC_BUS, PV_gen, Wind_gen, Diesel_electricity_generation, Total_EDemand, Excess_gen_production, Gen_switch)
        
        #Reward
        Score_C = CostScore(RO_switch, Well_pump_switch, Gen_switch, Cistern_pump_switch,  Diesel_electricity_generation, Excess_energy, Battery_SOC, Battery_SOC_p)
        Score_S = SusScore(RO_switch, Well_pump_switch, Gen_switch, Cistern_pump_switch,  Diesel_electricity_generation, Excess_energy, Battery_SOC, Battery_SOC_p)
        Score_R = ReliScore(H_Well)
        
        #Append results
        PV_gen_.append(PV_gen)
        Wind_gen_.append(Wind_gen)
        Diesel_electricity_generation_.append(Diesel_electricity_generation)
        Excess_energy_.append(Excess_energy)
        Excess_energy_signal_.append(Excess_energy_signal)
        Water_shortage_.append(Water_shortage)
        Excess_water_.append(Excess_water)
        Total_EDemand_.append(Total_EDemand)
        RO_water_flow_.append(RO_water_flow)
        Well_water_flow_.append(Well_water_flow)
        Water_flow_to_pressure_tank_.append(Water_flow_to_pressure_tank)
        Water_Demand_.append(WDemand)
        
        Score_C_.append(Score_C)
        Score_S_.append(Score_S)
        Score_R_.append(Score_R)
        
        Battery_SOC_.append(Battery_SOC)#
        DC_BUS_.append(DC_BUS)#
        Cistern_Tank_.append(Cistern_Tank)#
        Pressure_Tank_.append(Pressure_Tank)#
        H_Well_.append(H_Well)#
        Gen_switch_.append(Gen_switch)#
        Cistern_pump_switch_.append(Cistern_pump_switch)#
        
        RO_switch_.append(RO_switch)
        Well_switch.append(Well_pump_switch)
           
    #Exclude final values from results
    Battery_SOC_=Battery_SOC_[:-1]
    DC_BUS_=DC_BUS_[:-1]
    Cistern_Tank_=Cistern_Tank_[:-1]
    Pressure_Tank_=Pressure_Tank_[:-1]
    H_Well_=H_Well_[:-1]
    Gen_switch_=Gen_switch_[:-1]
    Cistern_pump_switch_=Cistern_pump_switch_[:-1]
    
    #Results dataframe
    Results = pd.DataFrame(columns=[''], index=pd.Series(range(0, (N))))
    Results[''] = range(N)

    Results.insert(1, 'PV_Gen_kW', PV_gen_)
    Results.insert(1, 'Wind_Gen_kW', Wind_gen_)
    
    Results.insert(1, 'Water_flow_to_pressure_tank', Water_flow_to_pressure_tank_)
    
    Results.insert(1, 'Water_Shortage', Water_shortage_)
    Results.insert(1, 'Water_Excess', Excess_water_)
    Results.insert(1, 'Excess_E_Signal', Excess_energy_signal_)
    Results.insert(1, 'Excess_E_kW', Excess_energy_)
    Results.insert(1, 'Diesel_Gen_kW', Diesel_electricity_generation_)
    Results.insert(1, 'Score_C', Score_C_)
    Results.insert(1, 'Score_S', Score_S_)
    Results.insert(1, 'Score_R', Score_R_)
    
    #States:
    Results.insert(1, 'Precipitation', R_v)
    Results.insert(1, 'Wind_Spd_m/s', Wind_spd_IOSN3_v)
    Results.insert(1, 'Temp_C', Ambient_temp_v)
    Results.insert(1, 'GHI_W/m2', GHI_W_m2_v)
    Results.insert(1, 'Total_EDemand_W', Total_EDemand_)
    Results.insert(1, 'Total_ELoad_W', EDemand)
    Results.insert(1, 'Water_Demand', Water_Demand_)
        
    Results.insert(1, 'Ground_Water_Depth', H_Well_)
    Results.insert(1, 'Cistern_Tank', Cistern_Tank_)
    Results.insert(1, 'Pressure_Tank', Pressure_Tank_)
    Results.insert(1, 'SOC_c', Battery_SOC_)
    Results.insert(1, 'Cistern Switch', Cistern_pump_switch_)
    Results.insert(1, 'Gen_Switch', Gen_switch_)
    
    Results.insert(1, 'RO_water_flow', RO_water_flow_)
    Results.insert(1, 'Well_water_flow', Well_water_flow_)
    Results.insert(1, 'RO_Switch', RO_switch_)
    Results.insert(1, 'Well_Switch', Well_switch)

    #Results = Results.reindex(
    #    sorted(Results.columns[1:len(Results.columns)]), axis=1)
    Results.insert(1, 'Date', pd.date_range(
        start='2022-06-01 00:00', end='2022-08-31 23:59', freq='1min'))
    
    return Results









