############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Forecasting
# Lesson: Holt-Winters

# Citation: 
# PEREIRA, V. (2018). Project: Forecasting, File: Python-Forecasting-04-Holt-Winters.py, GitHub repository: <https://github.com/Valdecy/Forecasting-03-Holt>

############################################################################

# Installing Required Libraries
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import mean_squared_error
from math import sqrt

################     Part 1 - Holt-Winters' Method    #############################

# Function: HW
def holt_winters(timeseries, alpha = 0.2, beta = 0.1, gama = 0.1, m = 12, graph = True, horizon = 0, trend = "multiplicative", seasonality = "multiplicative"):
    
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    holt   = pd.DataFrame(np.nan, index = timeseries.index, columns = ['Holt'])
    holt_A = pd.DataFrame(np.nan, index = timeseries.index, columns = ['A'])
    holt_T = pd.DataFrame(np.nan, index = timeseries.index, columns = ['T'])
    holt_S = pd.DataFrame(np.nan, index = timeseries.index, columns = ['S'])
    n = 1
    
    for i in range(m-1, len(timeseries) - 1):  
        
        # No seasonality          
        if (i == m-1 and trend == "none" and seasonality == "none"):
            holt_A.iloc[i, 0] = float(timeseries.iloc[0,:])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0]
          
        elif (i == m-1 and trend == "additive" and seasonality == "none"):
            holt_A.iloc[i, 0] = float(timeseries.iloc[0,:])
            holt_T.iloc[i, 0] = 0.0 
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]
          
        elif (i == m-1 and trend == "multiplicative" and seasonality == "none"):
            holt_A.iloc[i, 0] = float(timeseries.iloc[0,:])
            holt_T.iloc[i, 0] = 1.0
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]
          
        elif (i > m-1 and trend == "none" and seasonality == "none"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:])) + (1 - alpha)*(holt_A.iloc[i - 1, 0])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0]
            last = float(holt.iloc[i,0])
    
        elif (i > m-1 and trend == "additive" and seasonality == "none"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:])) + (1 - alpha)*(holt_A.iloc[i - 1, 0] + holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] - holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]
            last = float(holt.iloc[i,0])
            
        elif (i > m-1 and trend == "multiplicative" and seasonality == "none"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:])) + (1 - alpha)*(holt_A.iloc[i - 1, 0] * holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] / holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] * n*holt_T.iloc[i, 0]
            last = float(holt.iloc[i,0])
        
        # Additive seasonality
        if (i == m-1 and trend == "none" and seasonality == "additive"):
            for j in range(0, i + 1):
                holt_S.iloc[j, 0] = float(timeseries.iloc[j,:]) - float(timeseries.iloc[0:i+1,:].mean())
            holt_A.iloc[i, 0] = float(timeseries.iloc[i,:]) - holt_S.iloc[i, 0]
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + holt_S.iloc[i - m + 1, 0]
          
        elif (i == m-1 and trend == "additive" and seasonality == "additive"):
            for j in range(0, i + 1):
                holt_S.iloc[j, 0] = float(timeseries.iloc[j,:]) - float(timeseries.iloc[0:i+1,:].mean())
            holt_A.iloc[i, 0] = float(timeseries.iloc[i,:]) - holt_S.iloc[i, 0]
            holt_T.iloc[i, 0] = 0.0 
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] +  holt_T.iloc[i, 0] + holt_S.iloc[i - m + 1, 0]
          
        elif (i == m-1 and trend == "multiplicative" and seasonality == "additive"):
            for j in range(0, i + 1):
                holt_S.iloc[j, 0] = float(timeseries.iloc[j,:]) - float(timeseries.iloc[0:i+1,:].mean())
            holt_A.iloc[i, 0] = float(timeseries.iloc[i,:]) - holt_S.iloc[i, 0]
            holt_T.iloc[i, 0] = 1.0 
            holt.iloc[i + n, 0]  =  (holt_A.iloc[i, 0] *  holt_T.iloc[i, 0]) + holt_S.iloc[i - m + 1, 0]
          
        elif (i > m-1 and trend == "none" and seasonality == "additive"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:]) - holt_S.iloc[i - m, 0]) + (1 - alpha)*(holt_A.iloc[i - 1, 0])
            holt_S.iloc[i, 0] = gama*(float(timeseries.iloc[i,:]) - holt_A.iloc[i - 1, 0]) + (1 - gama)*(holt_S.iloc[i - m, 0])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + holt_S.iloc[i - m + 1, 0]
            last = float(holt.iloc[i,0])
    
        elif (i > m-1 and trend == "additive" and seasonality == "additive"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:]) - holt_S.iloc[i - m, 0]) + (1 - alpha)*(holt_A.iloc[i - 1, 0] + holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] - holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt_S.iloc[i, 0] = gama*(float(timeseries.iloc[i,:]) - holt_A.iloc[i - 1, 0] - holt_T.iloc[i - 1, 0]) + (1 - gama)*(holt_S.iloc[i - m, 0])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0] + holt_S.iloc[i - m + 1, 0]
            last = float(holt.iloc[i,0])
            
        elif (i > m-1 and trend == "multiplicative" and seasonality == "additive"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:]) - holt_S.iloc[i - m, 0]) + (1 - alpha)*(holt_A.iloc[i - 1, 0] * holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] / holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt_S.iloc[i, 0] = gama*(float(timeseries.iloc[i,:]) - holt_A.iloc[i - 1, 0]* holt_T.iloc[i - 1, 0]) + (1 - gama)*(holt_S.iloc[i - m, 0])
            holt.iloc[i + n, 0]  =  (holt_A.iloc[i, 0] * n*holt_T.iloc[i, 0]) + holt_S.iloc[i - m + 1, 0]
            last = float(holt.iloc[i,0])
        
        # Multiplicative seasonality    
        if (i == m-1 and trend == "none" and seasonality == "multiplicative"):
            for j in range(0, i + 1):
                holt_S.iloc[j, 0] = float(timeseries.iloc[j,:]) / float(timeseries.iloc[0:i+1,:].mean())
            holt_A.iloc[i, 0] = float(timeseries.iloc[i,:]) / holt_S.iloc[i, 0]
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] * holt_S.iloc[i - m  + 1, 0]
          
        elif (i == m-1 and trend == "additive" and seasonality == "multiplicative"):
            for j in range(0, i + 1):
                holt_S.iloc[j, 0] = float(timeseries.iloc[j,:]) / float(timeseries.iloc[0:i+1,:].mean())
            holt_A.iloc[i, 0] = float(timeseries.iloc[i,:]) / holt_S.iloc[i, 0]
            holt_T.iloc[i, 0] = 0.0 
            holt.iloc[i + n, 0]  =  (holt_A.iloc[i, 0] + holt_T.iloc[i, 0]) * holt_S.iloc[i - m + 1, 0]
          
        elif (i == m-1 and trend == "multiplicative" and seasonality == "multiplicative"):
            for j in range(0, i + 1):
                holt_S.iloc[j, 0] = float(timeseries.iloc[j,:]) / float(timeseries.iloc[0:i+1,:].mean())
            holt_A.iloc[i, 0] = float(timeseries.iloc[i,:]) / holt_S.iloc[i, 0]
            holt_T.iloc[i, 0] = 1.0 
            holt.iloc[i + n, 0]  = holt_A.iloc[i, 0] * holt_T.iloc[i, 0] * holt_S.iloc[i - m + 1, 0]
          
        elif (i > m-1 and trend == "none" and seasonality == "multiplicative"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:]) / holt_S.iloc[i - m, 0]) + (1 - alpha)*(holt_A.iloc[i - 1, 0])
            holt_S.iloc[i, 0] = gama*(float(timeseries.iloc[i,:]) / holt_A.iloc[i - 1, 0]) + (1 - gama)*(holt_S.iloc[i - m, 0])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] * holt_S.iloc[i - m + 1, 0]
            last = float(holt.iloc[i,0])
    
        elif (i > m-1 and trend == "additive" and seasonality == "multiplicative"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:]) / holt_S.iloc[i - m, 0]) + (1 - alpha)*(holt_A.iloc[i - 1, 0] + holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] - holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt_S.iloc[i, 0] = gama*(float(timeseries.iloc[i,:]) / (holt_A.iloc[i - 1, 0] + holt_T.iloc[i - 1, 0])) + (1 - gama)*(holt_S.iloc[i - m, 0])
            holt.iloc[i + n, 0]  =  (holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]) * holt_S.iloc[i - m + 1, 0]
            last = float(holt.iloc[i,0])
            
        elif (i > m-1 and trend == "multiplicative" and seasonality == "multiplicative"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:]) / holt_S.iloc[i - m, 0]) + (1 - alpha)*(holt_A.iloc[i - 1, 0] * holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] / holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt_S.iloc[i, 0] = gama*(float(timeseries.iloc[i,:]) / (holt_A.iloc[i - 1, 0]* holt_T.iloc[i - 1, 0])) + (1 - gama)*(holt_S.iloc[i - m, 0])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] * n*holt_T.iloc[i, 0] * holt_S.iloc[i - m + 1, 0]
            last = float(holt.iloc[i,0])
     
    if horizon > 0: 
        time_horizon = len(timeseries) + horizon 
        time_horizon_index = pd.date_range(timeseries.index[0], periods = time_horizon, freq = timeseries.index.inferred_freq) 
        pred = pd.DataFrame(np.nan, index = time_horizon_index, columns = ["Prediction"])
        for i in range(0, horizon):
            pred.iloc[len(timeseries) + i] = last
        pred = pred.iloc[:,0]
    
    rms = sqrt(mean_squared_error(timeseries.iloc[(m):,0], holt.iloc[(m):,0]))
    timeseries = timeseries.iloc[:,0]
    holt = holt.iloc[:,0]
    
    if graph == True and horizon <= 0:
        style.use('ggplot')
        plt.plot(timeseries)
        plt.plot(holt)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.xticks(rotation = 90)
        plt.show()
    elif graph == True and horizon > 0:
        style.use('ggplot')
        plt.plot(timeseries)
        plt.plot(holt)
        plt.plot(pred)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.xticks(rotation = 90)
        plt.show()
   
    return holt, last, rms

    ############### End of Function ##############

# Brute Force Optmization    
def optimize_holt_winters(timeseries, trend = "multiplicative", seasonality = "multiplicative", m = 12):
    error = pd.DataFrame(columns = ['alpha', 'beta', 'gama', 'rmse'])
    count = 0
    for alpha in range(0, 11):
        for beta in range(0, 11):
            for gama in range(0, 11):
                print("alpha = ", alpha/10, " beta = ", beta/10, " gama = ", gama/10)
                holt, last, rms = holt_winters(timeseries, alpha = alpha/10, beta = beta/10, gama = gama/10, m = m, graph = False, horizon = 0, trend = trend, seasonality = seasonality)
                error.loc[count] = [alpha/10, beta/10, gama/10, rms]
                count = count + 1
    return error, error.loc[error['rmse'].idxmin()]
    ############### End of Function ##############
    
######################## Part 2 - Usage ####################################
 
# Load Dataset 
df = pd.read_csv('Python-Forecasting-04-Dataset.txt', sep = '\t')

# Transform Dataset to a Time Series
X = df.iloc[:,:]
X = X.set_index(pd.DatetimeIndex(df.iloc[:,0])) # First column as row names
X = X.iloc[:,1]

# Calling Functions
holt_winters(X, alpha = 0.2, beta = 0.1, gama = 0.2, m = 4, graph = True, horizon = 0, trend = "multiplicative", seasonality = "multiplicative")

opt = optimize_holt_winters(X, trend = "multiplicative", seasonality = "multiplicative", m = 4)

########################## End of Code #####################################
