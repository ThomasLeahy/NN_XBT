# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:31:12 2018

@author: thomaspatrickleahy francescponsllopis
"""

# packages
import netCDF4 as nc4
import numpy as np
import datetime as dt
import string
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix



def extract_date(date):
    """
    This function returns a python date-time object
    from an int8 date in format, e.g. 19780101

   Input Parameters:
    -----------------

   * date = int8 date in format, e.g. 19780101

   Returns:
    --------

   * dateobj = python date-time object

   """
    datestr = date.astype('str') # Convert to string to chop the date into year, month, day
    yr = datestr[0:4]
    mn = datestr[4:6]
    dy = datestr[6:]
    # Date-time can throw an error occasionally if day number in month is exceeded
    try:
        dateobj = dt.date(int(yr), int(mn), int(dy))
    except:
        dateobj = dt.date(int(yr), int(mn), int(15)) # Make day 15th if problems..
    return dateobj


indir='./Downloads/data_ocean/' ### input path
outdir = ''


imeta_dict = {}

for year in list(range(1966, 2016)):
    stryr = str(year)
    ncfile  = 'iquod_xbt_'+stryr+'.nc'

    print("Processing from file "+ncfile+"...")

    f = nc4.Dataset(indir+ncfile,'r', format='NETCDF4')

    depths = f.variables['z']
    depth_sizes = f.variables['z_row_size']
    country = f.variables['country']
    date = f.variables['date']
    instr_name = f.variables['Temperature_Instrument']
    instr_imeta = f.variables['Temperature_Instrument_intelligentmetadata'] # Flag for iMetaData (True or False)
    WOD_unique = f.variables['wod_unique_cast'] # Unique WOD cast number
    d_index = np.cumsum(depth_sizes)-1 # Create index for maximum depths
    max_depths = depths[d_index] # This seems to work fine - note that there are some negative depth values!!
    #
    mask = instr_imeta[:].data # Determine profiles that contain iMetaData
    mindex = np.where(mask != 1)
    #
    #    # Restrict all information to those pertaining to iMetaData
    country = country[mindex]
    instr_name = instr_name[mindex]
    WOD_unique = WOD_unique[mindex]
    max_depths = max_depths[mindex]
    depth_sizes = depth_sizes[mindex]
#
# Create an empty dictionary to store the salient information..
    
    for ii, ww in enumerate(WOD_unique):
        cntry     = b''.join(country[ii].data.tolist()).decode("utf-8")
        iMetaTim  = b''.join(instr_name[ii].data.tolist()).decode("utf-8")
        max_depth = max_depths[ii]
        pdate   = extract_date(date[ii])
        newlist   = [cntry, max_depth, pdate, iMetaTim]
        imeta_dict.update({ww:newlist})
            

    
       ## Loop through all the types and append to an array and look for uniques and
        ## and count the occurances of each
    types = np.array([])
    for k in imeta_dict.keys():
        types = np.append(types, imeta_dict[k][3])
    unique_types = np.unique(types)
    
    max_depths = np.array([])
    for k in imeta_dict.keys():
            max_depths = np.append(max_depths, imeta_dict[k][1])
            
    dates_array = np.array([])
    for k in imeta_dict.keys():
            dates_array = np.append(dates_array, imeta_dict[k][2])
    
    country_array = np.array([])
    for k in imeta_dict.keys():
            country_array = np.append(country_array, imeta_dict[k][0])
    
    ### arrays of unique probe types, and features are created
    
    
    probe_dict = {}      
    unique_types_list = list(unique_types)
    type_code = list(np.arange(0,len(unique_types)))
    
    for i in range(len(unique_types_list)):
        probe_dict[type_code[i]] = unique_types_list[i]
    
    df= pd.DataFrame({'types':list(types)})
    df['idx']=pd.Categorical(df['types'].astype(str)).codes
    types_input=np.array(df['idx'])
    
       
    df1=pd.DataFrame({'country':list(country_array)})
    df1['idx'] = pd.Categorical(df1['country'].astype(str)).codes
    country_input = np.array(df1['idx'])
    
    ## data frame to enable us to look up the probe types
    ## also creates inputs    
    
    
    # remove missing data
    if sum(np.isnan(max_depths)) > 0:
        things_to_erase = np.argwhere(np.isnan(max_depths))
        
        max_depths = np.delete(max_depths, things_to_erase)
        dates_array = np.delete(dates_array, things_to_erase)
        country_array = np.delete(country_array, things_to_erase)
        types_input = np.delete(types_input, things_to_erase)
    
    date0=dt.date(2010,1,1)
    dates = np.array([])
    for i in range(len(dates_array)):
        dates=np.append(dates,(dates_array[i]-date0).days)
    
    # create the feature matrix
    X_data = []
    for x in range(0,len(max_depths)):
        X_data.append([max_depths[x], dates[x], country_input[x]])
        
        
    
    
    
        
    X_data = np.array(X_data)
    
    y = types_input
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y)
    
    #scaling
    scaler = StandardScaler()
    
    # Fit only to the training data
    print("training")
    scaler.fit(X_train)
    
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(10,10,)) # num hidden layers and nodes 
    mlp.fit(X_train,y_train)
    
    predictions = mlp.predict(X_test)

    ### Output data   # uncomment if required
#    nam = 'report'+stryr+'.txt'
#    nam2 = 'cmatrix'+stryr+'.csv'
#    file = open(nam,"w")
#    file.write(classification_report(y_test,predictions))
#    
#    np.savetxt(nam2, confusion_matrix(y_test,predictions) , delimiter=",")
#    file.close()
