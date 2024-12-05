import pandas as pd
import time
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from SML_WE_SD_Modular_2022_V11 import *

Data = pd.read_csv('Input Data.csv', index_col=0)
Power_Table = pd.read_csv('Power table.csv')
Energy_Real = pd.read_csv('Energy real data.csv', index_col=0)
Water_Real = pd.read_csv('Water real data.csv', index_col=0)

###############################################################################
#Calibration - 50 days - from 6/25/2022 00:00 to 8/13/2022 23:59 - 75% - 72000 mins - indx: 34560-106559
###############################################################################
R_v = Data['Rain (in)']
Well_water_flow_v = Water_Real['Well water production pulse difference (gal)']

H_Well = Water_Real['Well depth (inch)'][34560]
Excess_water=0

def GroundWater(H_Well, Excess_water, Well_water_flow, R , A=1094):    
    
    Precipitation = R*(1/12)*A*7.48 #1ft = 12in, 1ft3 = 7.48 gal 
    
    Groundwater_charge = Excess_water+Precipitation
    Groundwater_discharge = Well_water_flow

    H_Well = H_Well + (Groundwater_charge - Groundwater_discharge)/(A*7.48/12)
    
    return H_Well

def Watershed(Well_water_flow_v, R_v, A, H_Well, Excess_water):
    
    for t in range(34560,106560):
        
        H_Sim.append(H_Well)
        
        H_Well = GroundWater(H_Well, Excess_water, Well_water_flow_v[t], R_v[t] , A)
    return H_Sim

## Calibration process
MSE={}
R2={}
for A in tqdm(range(1,1501),desc='Calibration for Area', unit='Steps'):
    
    H_Sim = Watershed(Well_water_flow_v, R_v, A, H_Well, Excess_water)
    H_Real = Water_Real.loc[34560:106559,'Well depth (inch)'].interpolate(method='linear', window=3, min_periods=1).reset_index(drop=True)
    df = pd.DataFrame({'H_Sim': H_Sim, 'H_Real': H_Real})
    #df = df.dropna(subset=['H_Real'])

    #MSE
    mse = mean_squared_error(df['H_Sim'], df['H_Real'])
    
    #R2
    slope, intercept, r_value, p_value, std_err = linregress(df['H_Sim'], df['H_Real'])
    r_squared = r_value ** 2
    
    MSE[(A)]=mse
    R2[(A)]=r_squared
    
min_value = min(MSE.values())
A = [key for key, value in MSE.items() if value == min_value][0]
print("Calibrated A = ",A)

#A = 1097
print("Calibrated A = ",A)
#Run the model based on the calibration result:
H_Sim = []
H_Sim = Watershed(Well_water_flow_v, R_v, A, H_Well, Excess_water)
H_Real = Water_Real.loc[34560:106559,'Well depth (inch)'].interpolate(method='linear', window=3, min_periods=1).reset_index(drop=True)
df = pd.DataFrame({'H_Sim': H_Sim, 'H_Real': H_Real})
df = df.dropna(subset=['H_Real'])

###############################################################################
#Validation - 9 days - from 8/14/2022 00:00 to 8/22/2022 23:59 - 15% - 12960 mins - indx: 106560-119519
###############################################################################

#Run the model based on the calibration result:
def Watershed(Well_water_flow_v, R_v, A, H_Well, Excess_water):
    
    for t in range(106560,119520):
        
        H_Sim.append(H_Well)
        
        H_Well = GroundWater(H_Well, Excess_water, Well_water_flow_v[t], R_v[t] , A)
    return H_Sim

H_Well = Water_Real['Well depth (inch)'][106560]
Excess_water=0

H_Sim = Watershed(Well_water_flow_v, R_v, A, H_Well, Excess_water)
H_Real = Water_Real.loc[34560:119519,'Well depth (inch)'].interpolate(method='linear', window=3, min_periods=1).reset_index(drop=True)
df = pd.DataFrame({'H_Sim': H_Sim, 'H_Real': H_Real})
df = df.dropna(subset=['H_Real'])

#Calibration Results
#MSE
mse_cal = mean_squared_error(df['H_Sim'].iloc[:72001], df['H_Real'].iloc[:72001])
print("Min MSE in calibration = ", mse_cal)

#MAE
mae_cal = np.abs(df['H_Sim'].iloc[:72001].values - df['H_Real'].iloc[:72001].values).mean()
print("MAE in calibration = ", mae_cal)

#R2
slope, intercept, r_value, p_value, std_err = linregress(df['H_Sim'].iloc[:72001], df['H_Real'].iloc[:72001])
r_squared_cal = r_value ** 2
print("Calibration R2 = ", r_squared_cal)



#Validation Results
#MSE
mse_val = mean_squared_error(df['H_Sim'].iloc[72000:], df['H_Real'].iloc[72000:])
print("Test set MSE = ", mse_val)

#MAE
mae_val = np.abs(df['H_Sim'].iloc[72000:].values - df['H_Real'].iloc[72000:].values).mean()
print("MAE in calibration = ", mae_val)

#R2
slope, intercept, r_value, p_value, std_err = linregress(df['H_Sim'].iloc[72000:], df['H_Real'].iloc[72000:])
r_squared_val = r_value ** 2
print("Test set R2 = ", r_squared_val)


# Plotting
#%matplotlib Inline
# Create a time array based on the range of data
time_steps = range(len(df['H_Real']))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['H_Real'], label='Real Data', linewidth=3)
plt.plot(df.index, df['H_Sim'], label='Simulated Data', linewidth=3)

# Set x-axis label and ticks with larger font size
plt.xlabel('Time Steps (Minutes)', fontweight='bold', fontsize=14)

# Set y-axis label with larger font size
plt.ylabel('Well Depth (inch)', fontweight='bold', fontsize=14)

# Set plot title with larger font size
plt.title('Groundwater Module Calibration and Validation\n', fontweight='bold', fontsize=25)

# Set x-axis limits
plt.xlim(-100, 85000)

# Add x-axis grid lines
plt.grid(axis='x')

# Add y-axis grid lines
plt.grid(axis='y')

# Add legend with larger font size and gray box inside the graph on the top right
plt.legend(fontsize='large', loc='upper right', shadow=True, fancybox=True, bbox_to_anchor=(0.25, 0.15), frameon=True)

# Add shade for data before 72000
plt.axvspan(-100, 72000, color='lightgrey', alpha=0.4)

# Add shade from x-axis step 72000 to the end
plt.axvspan(72000, 85000, color='lightgrey', alpha=0.7)

# Add text boxes
plt.text(0.42, 0.75, "    (85%)\nMSE=1.60\nMAE=0.99\nR2=96.94%", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.87, 0.75, "    (15%)\nMSE=1.16\nMAE=0.69\nR2=71.80%", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Add text boxes
plt.text(40000, 175, "Calibration", fontsize=18, fontweight='bold', ha='center', va='center')
plt.text(78500, 175, "Validation", fontsize=18, fontweight='bold', ha='center', va='center')

# Save the plot with the highest resolution to a TIFF file
plt.savefig('Watershed_model_calibration2.tiff', dpi=600)

# Show plot
plt.tight_layout()
plt.show()

