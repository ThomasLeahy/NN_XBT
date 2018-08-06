# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:28:26 2017

@author: thomasleahy francescponsllopis
"""
import netCDF4 as nc4
import numpy as np
import datetime as dt
import itertools
import string
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from io import BytesIO
import csv
import random

random.seed(101)
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
    dat = datestr.data
    wrdat = dat.astype('str')
    datestr = wrdat[()]
    yr = datestr[0:4]
    mn = datestr[4:6]
    dy = datestr[6:]
    # Date-time can throw an error occasionally if day number in month is exceeded
    try:
        dateobj = dt.date(int(yr), int(mn), int(dy))
    except:
        dateobj = dt.date(int(yr), int(mn), int(15)) # Make day 15th if problems..
    return dateobj


indir=''
outdir = ''

#year = 1987

for year in range(1966,2013):
#for year in [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2008, 2009, 2011, 2012, 2013, 2014, 2015, 2016]:
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
    
       # We could insert Matt's iMeta information here and write the whole lot out to file..
    imeta_dict = {}

    for ii, ww in enumerate(WOD_unique):
        cntry     = b''.join(country[ii].data.tolist()).decode("utf-8")
        iMetaTim  = b''.join(instr_name[ii].data.tolist()).decode("utf-8")
        max_depth = max_depths[ii]
        #depth_size = depth_sizes[ii]
        pdate   = extract_date(date[ii])
    #        iMetaMatt = iquod_v01_imeta(max_depth, pdate, cntry) # Use v0.1 algorithm on profile info
        newlist   = [cntry, max_depth, pdate, iMetaTim]
        imeta_dict.update({ww:newlist})
            
    print("data loaded")
    #    # Write the summary information to file..
    ##    
    ##    pickle.dump( imeta_dict, open( outdir + 'iquod_xbt_'+stryr+'_iMetaData_summary.pickle', 'wb' ) )
    
    
       ## Loop through all the types and append to an array and look for uniques and
        ## and count the occurances of each

    
    
    types = np.array([])
    for k in imeta_dict.keys():
        types = np.append(types, imeta_dict[k][3])
    

    str123 = "UNKNOWN"
    places = np.array([])
    for i in range(1,len(types)):
        check = types[i].find(str123)
        if check != -1:
            places = np.append(places, int(i))
    places.astype(int)
    types = np.delete(types, places)
    
    max_depths = np.array([])
    for k in imeta_dict.keys():
            max_depths = np.append(max_depths, imeta_dict[k][1])
    max_depths = np.delete(max_depths, places)
            
            
    dates_array = np.array([])
    for k in imeta_dict.keys():
            dates_array = np.append(dates_array, imeta_dict[k][2])
    dates_array = np.delete(dates_array, places)

    
    country_array = np.array([])
    for k in imeta_dict.keys():
            country_array = np.append(country_array, imeta_dict[k][0])
    country_array = np.delete(country_array, places)

    depth_sizes = np.delete(depth_sizes, places)
    
    print("arrays created")
    
    if sum(np.isnan(max_depths)) > 0:
        things_to_erase = np.argwhere(np.isnan(max_depths))
        
        max_depths = np.delete(max_depths, things_to_erase)
        dates_array = np.delete(dates_array, things_to_erase)
        country_array = np.delete(country_array, things_to_erase)
        types = np.delete(types, things_to_erase)
    
    print("nans removed")
    
    unique_types = np.unique(types)

    probe_dict = {}      
    unique_types_list = list(unique_types)
    type_code = list(np.arange(0,len(unique_types)))
    T4=-1
    T5=-1
    T7=-1
    for i in range(len(unique_types_list)):
        probe_dict[type_code[i]] = unique_types_list[i]
        if unique_types_list[i]=='XBT: T4 (SIPPICAN)':
            T4=type_code[i]
        if unique_types_list[i]=='XBT: T5 (SIPPICAN)':
            T5=type_code[i]
        if unique_types_list[i]=='XBT: T7 (SIPPICAN)':
            T7=type_code[i]

    
    df= pd.DataFrame({'types':list(types)})
    df['idx']=pd.Categorical(df['types'].astype(str)).codes
    types_input=np.array(df['idx'])
    
    #unique_country = np.unique(country_array)    
      
    
    df1=pd.DataFrame({'country':list(country_array)})
    df1['idx'] = pd.Categorical(df1['country'].astype(str)).codes
    country_input = np.array(df1['idx'])
    
#    nam = 'types_dict30per'+stryr+'.txt'
#    file = open(nam,"w")
#    file2 = open(nam2,"w")
#    file.write(str(df.groupby('types').agg({"idx": ['count']})))
#    file.close() 
#    
    date0=dt.date(2010,1,1)
    dates = np.array([])
    for i in range(len(dates_array)):
        dates=np.append(dates,(dates_array[i]-date0).days)
    
    
    enc = OneHotEncoder()

    cuni =np.unique(country_input)
    enc.fit(cuni.reshape(-1, 1))
    country_input=enc.transform(country_input.reshape(-1,1)).toarray()

    
    X_data = []
    for x in range(0,len(max_depths)):
        todo = ([max_depths[x], dates[x], country_input[x].tolist()])
        
        flat=[]
        flat.append(todo[0])
        flat.append(todo[1])
        for i in range(0, len(todo[2])-1):
            flat.append(todo[2][i])
        X_data.append(flat)
            

        
        
    
    
    
        
    X_data = np.array(X_data)
    
    y = types_input
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, random_state = 1, test_size=0.25)
    
    scaler = StandardScaler()
    
    # Fit only to the training data
    print("training")
    scaler.fit(X_train)
    
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(10,10,)) # num hidden layers and nodes 
    search_dict = {"tol":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    clf=GridSearchCV(mlp, search_dict)
    clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)
    
    
    positions = np.array([T4, T5, T7])
    
    ### Output data   # uncomment if required
    nam = 'report_new2_'+stryr+'.txt'
    nam2 = 'cmatrix_new2_'+stryr+'.csv'
    nam3 = 'tolselect_new2_'+stryr+'.csv'
    nam4 = 'positions_new2_'+stryr+'.csv'
    file = open(nam,"w")
    file.write(classification_report(y_test,predictions))
    np.savetxt(nam2, confusion_matrix(y_test,predictions) , delimiter=",")
    np.savetxt(nam3, clf.cv_results_["rank_test_score"] , delimiter=",")
    np.savetxt(nam4, positions , delimiter=",")
    file.close()
#    
