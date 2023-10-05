#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:58:45 2023

@author: thiagovg
"""

"""GOES-16 products processing

Fire product name structure: 

OR_ABI-L2-FDCF-M3_G16_sYYYYJJJHHMMSSs_eYYYYJJJHHMMSSs_cYYYYJJJHHMMSSs.nc
"""

###############################################################################
# Import the libraries 
import os
from io import BytesIO
import s3fs
import xarray as xr
import numpy as np
import glob
from pyproj import Proj
import pandas as pd
import warnings

#Functions
###############################################################################
# Get one sample file of the satellite to extract the geometric variables and do the calculations
# to transform the coordinate matrix of the satellite in a latitude and longitude matrix
# The coordinate variables remain constant so it in not necessary to do this calculation more than once

def get_lat_lon(file_system):
    
    files = file_system.ls('noaa-goes16/ABI-L2-FDCF/2020/'+str(200).zfill(3)+'/'+str(15).zfill(2)+'/') # list of 6 files for 2020 day 200, UTC time 15:00, 15:10, 15:20, ..., 15:50

    with fs.open(files[0], 'rb') as f:
      ds0 = xr.open_dataset(BytesIO(f.read()), engine='h5netcdf') #ds0 is the variable containing all the information of the file downloaded
    
    sat_h = ds0.goes_imager_projection.perspective_point_height[0] # hight of the satellite
    sat_lon = ds0.goes_imager_projection.longitude_of_projection_origin[0] # longitude position of the satellite
    sat_sweep = ds0.goes_imager_projection.sweep_angle_axis[0] # sweep angle
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep) # With these variables we can calculate the projection of the satellite coordinates on the latitude and longitude coordinates
    X = np.array(ds0.x) * sat_h
    Y = np.array(ds0.y) * sat_h
    XX, YY = np.meshgrid(X,Y)
    rlon, rlat = p(XX, YY, inverse=True) 
    
    return rlat,rlon

###############################################################################
# Get a list of all the files within the period of time informed 

def get_files(s_year,e_year,s_day,e_day,s_hour,e_our):
    print('Getting file names')
    aux = []
    for y in range(s_year,e_year+1):
        for d in range(s_day,e_day):
            for j in range(s_hour,e_our):
                FD = fs.ls('noaa-goes16/ABI-L2-FDCF/'+str(y)+'/'+str(d).zfill(3)+'/'+str(j).zfill(2)+'/') # list of 6 files for every hour within the range informed
                aux = np.append(aux,FD)
    return aux

###############################################################################
# For the box informed make a matrix for the central lat and lon for a grid of 0.5°x0.5° 
# The matrix will have a column of lats and lon and a third column for the FEER coeficient correspondent to that element of the grid

def get_indexes_v3(min_lon,max_lon,min_lat,max_lat,rlat,rlon,dados_feer):
    
    #Within the box informed divide on a grid of 0.5°x0.5°
    centers_lon = np.linspace(minlon+0.25, maxlon-0.25,num=int(maxlon-minlon)*2)
    centers_lat = np.linspace(minlat+0.25, maxlat-0.25,num=int(maxlat-minlat)*2)
    #To be able to calculate the correspondent FEER coeficient we need a grid of 1°x1° also
    centers_lon2 = np.linspace(minlon+0.5, maxlon-0.5,num=int(maxlon-minlon))
    centers_lat2 = np.linspace(minlat+0.5, maxlat-0.5,num=int(maxlat-minlat))
    

    aux_list = []
    
    
    for i in range(0,len(centers_lat2)):
        #Get the FEER coeficients that match the lat and lon of the 1°x1° grid within the box informed
        df2 = dados_feer.loc[(dados_feer['Latitude'] == centers_lat2[i]) & (dados_feer['Longitude'] <= (maxlon) )
                            & (dados_feer['Longitude'] >= (minlon) ),'Ce_850'].to_numpy()
        aux_list = np.append(aux_list,df2)
        #We need to repeat the longitudes to create a structure for the matrix similar to that informed on the FEER data
        if i == 0:
            aux2 = np.repeat(centers_lat2[i],len(centers_lon2))
            lat_lon_feer2 = np.column_stack((aux2,centers_lon2))
        else:
            aux2 = np.repeat(centers_lat2[i],len(centers_lon2))
            aux2 = np.column_stack((aux2,centers_lon2))
            lat_lon_feer2 = np.vstack((lat_lon_feer2,aux2))
    lat_lon_feer2 = np.column_stack((lat_lon_feer2,aux_list))

    #We did the same structure for the 0.5°x0.5° grid
    for i in range(0,len(centers_lat)):

        if i == 0:
            aux = np.repeat(centers_lat[i],len(centers_lon))
            lat_lon_feer = np.column_stack((aux,centers_lon))
            # latlon = np.vstack((latlon,au2))
        else:
            # print(i)
            aux = np.repeat(centers_lat[i],len(centers_lon))
            aux2 = np.column_stack((aux,centers_lon))
            lat_lon_feer = np.vstack((lat_lon_feer,aux2))
    
    #Now we compare the matrices and create the column for the FEER coeficients correspondent to each element of the 0.5°x0.5° grid
    FEER_ce = []
    for j in range(0, len(lat_lon_feer)):
        for n in range(0,len(lat_lon_feer2)):
            if((lat_lon_feer[j,0]==lat_lon_feer2[n,0]+0.25)or(lat_lon_feer[j,0]==lat_lon_feer2[n,0]-0.25)):
                if((lat_lon_feer[j,1]==lat_lon_feer2[n,1]+0.25)or(lat_lon_feer[j,1]==lat_lon_feer2[n,1]-0.25)):
                    FEER_ce = np.append(FEER_ce,lat_lon_feer2[n,2])

    #Add the column and create a matrix of the central lats and lons and the FEER coeficients 
    lat_lon_feer = np.column_stack((lat_lon_feer,FEER_ce))
    matrix = lat_lon_feer

    #Finally, we create a matrix that will be used for select the data satellite for each element of the 0.5°x0.5° grid
    #The first element is not for a element, but for the whole box informed, it was used in previous versions and can be used in tests
    I = np.where((rlat>=min_lat)&(rlat<=max_lat)&(rlon>=min_lon)&(rlon<=max_lon))
    index_list = []
    index_list.insert(0,I)

    for k in range(0,len(matrix)):
        aux1=np.where((rlat>=matrix[k,0]-0.25)&(rlat<=matrix[k,0]+0.25)&(rlon>=matrix[k,1]-0.25)&(rlon<=matrix[k,1]+0.25))
        index_list.insert(k+1,aux1)

    return index_list,matrix

###############################################################################
# Process the GOES data 
def process_data_v6(rlat,rlon,files,matrix,indexes):
    #Open the folder to save the date later
    os.chdir(outdir)
    #Start the loop throught the files listed before
    for i in range(0,len(files)):
      with fs.open(files[i], 'rb') as f:
          
        ds = xr.open_dataset(BytesIO(f.read()), engine='h5netcdf')
        try: #Try and except to avoid a rare and harmless process error
            
            #Get some information about the time and data using the file name structure
            prodbase = files[i].split('/')[5][:23] 
            starttime=files[i].split(prodbase)[1].split('_')[0]
            year,julian,hhmm=starttime[:4],starttime[4:7],starttime[7:11]
            plottitle=year+','+julian+','+hhmm
            fpart = starttime+','+plottitle
            #Processing message
            print('Processing year: {}, day: {}, hour: {}'.format(year,julian,hhmm)) #Processing Message    
            #This 'code' variable is used to transfom the hours and minutes in days percentage for future plots
            code = int(julian)+(int(hhmm)/100)/24+(int(hhmm) % 100)/60/24
            #####################################################################
            
            #Get the data Power = FRP, Temp = Temperature
            P = np.array(ds.Power)
            T = np.array(ds.Temp)
            # A = np.array(ds.Area)
            
            #Select the data over the element of the grid created
            P_box_amazon = P[indexes[1]]
            T_box_amazon = T[indexes[1]]
            array_P_box_amazon = P_box_amazon[~np.isnan(P_box_amazon)]
            array_T_box_amazon = T_box_amazon[~np.isnan(T_box_amazon)]
            
            #Some calculations with this data
            lat = matrix[0,0] #save the central lat
            lon = matrix[0,1] #save the central lon
            sum_frp = np.sum(array_P_box_amazon) #sum of the FRP
            N_frp = len(array_P_box_amazon) #N of data used for FRP
            FRE = np.sum(array_P_box_amazon*600) #Fire energy
            RE = np.sum(array_P_box_amazon*matrix[0,2]) #Rate of emission of TPM
            ME = np.sum(array_P_box_amazon*matrix[0,2]*600) #Mass of emission of TPM

            MCO2 = np.sum(array_P_box_amazon*Ce_CO2*matrix[0,2]*600) #Mass of emission of CO2
            MCO = np.sum(array_P_box_amazon*Ce_CO*matrix[0,2]*600) #Mass of emission of CO
            MCH4 = np.sum(array_P_box_amazon*Ce_CH4*matrix[0,2]*600) #Mass of emission of CH4

            #Std of the Mass of emission of CO2,CO,CH4
            s_RCO2 = array_P_box_amazon*sigma_Ce_CO2*matrix[0,2] 
            s_RCO = array_P_box_amazon*sigma_Ce_CO*matrix[0,2] 
            s_RCH4 = array_P_box_amazon*sigma_Ce_CH4*matrix[0,2] 
            s_MCO2 = (s_RCO2*600)**2
            s_MCO = (s_RCO*600)**2
            s_MCH4 = (s_RCH4*600)**2
            s_sum_me_CO2 = np.sqrt(np.sum(s_MCO2))
            s_sum_me_CO = np.sqrt(np.sum(s_MCO))
            s_sum_me_CH4 = np.sqrt(np.sum(s_MCH4))

            #mean of the FRP and temperature
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_temp = np.nanmean(array_T_box_amazon)
                mean_frp = np.nanmean(array_P_box_amazon)
            if np.isnan(mean_temp):
                mean_temp = -9999
            if np.isnan(mean_frp):
                mean_frp = -9999

            #Save the data processed
            results = str(code) +','+ str(lat)+','+ str(lon)+','+ str(sum_frp)+','+ str(N_frp)+','\
                +str(FRE) +','+ str(RE)+','+ str(ME)+','+ str(MCO2)+','+ str(s_sum_me_CO2)+','\
                +str(MCO) +','+ str(s_sum_me_CO)+','+ str(MCH4)+','+ str(s_sum_me_CH4)+','+ str(mean_frp)+','+str(mean_temp)
            #Compose results in a string and save them
            outstring = fpart+','+results + '\n'
            outfn.writelines(outstring)
            
            #Do the same for each element of the grid created
            for k in range(1,len(matrix)):

                P_box_amazon = P[indexes[k+1]]
                array_P_box_amazon = P_box_amazon[~np.isnan(P_box_amazon)]
                T_box_amazon = T[indexes[k+1]]
                array_T_box_amazon = T_box_amazon[~np.isnan(T_box_amazon)]
                
                lat = matrix[k,0]
                lon = matrix[k,1]
                
                sum_frp = np.sum(array_P_box_amazon)
                N_frp = len(array_P_box_amazon)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_temp = np.nanmean(array_T_box_amazon)
                    mean_frp = np.nanmean(array_P_box_amazon)
                if np.isnan(mean_temp):
                    mean_temp = -9999
                if np.isnan(mean_frp):
                    mean_frp = -9999
                
                FRE = np.sum(array_P_box_amazon*600)
                RE = np.sum(array_P_box_amazon*matrix[k,2])
                ME = np.sum(array_P_box_amazon*matrix[k,2]*600)

                MCO2 = np.sum(array_P_box_amazon*Ce_CO2*matrix[k,2]*600)
                MCO = np.sum(array_P_box_amazon*Ce_CO*matrix[k,2]*600)
                MCH4 = np.sum(array_P_box_amazon*Ce_CH4*matrix[k,2]*600)

                
                s_RCO2 = array_P_box_amazon*sigma_Ce_CO2*matrix[k,2]
                s_RCO = array_P_box_amazon*sigma_Ce_CO*matrix[k,2]
                s_RCH4 = array_P_box_amazon*sigma_Ce_CH4*matrix[k,2]
                s_MCO2 = (s_RCO2*600)**2
                s_MCO = (s_RCO*600)**2
                s_MCH4 = (s_RCH4*600)**2
                s_sum_me_CO2 = np.sqrt(np.sum(s_MCO2))
                s_sum_me_CO = np.sqrt(np.sum(s_MCO))
                s_sum_me_CH4 = np.sqrt(np.sum(s_MCH4))


                results = str(code) +','+ str(lat)+','+ str(lon)+','+ str(sum_frp)+','+ str(N_frp)+','\
                    +str(FRE) +','+ str(RE)+','+ str(ME)+','+ str(MCO2)+','+ str(s_sum_me_CO2)+','\
                    +str(MCO) +','+ str(s_sum_me_CO)+','+ str(MCH4)+','+ str(s_sum_me_CH4)+','+ str(mean_frp)+','+str(mean_temp)
                outstring = fpart+','+results + '\n'
                outfn.writelines(outstring)
           

        except OSError as error:
             print(error)
    #Close the file    
    outfn.close()
    return print('Done')
###############################################################################

# Define input and output folders and file
datadir = '/home/...' # Folder to save the csv with the data processed
datadir2 = '/home/...' # Folder with the csv with the FEER coeficient data

#Read the FEER data
data = sorted(glob.glob(datadir2+'/FEER*.csv'))
feer_data = pd.read_csv(data[0])

#Define folder and name of the file
outdir  = datadir+'output/' #output folder is optional
Ano =2020
outfile = outdir+'goes_data_emission_rate_amazon_definitive_box_'+str(Ano)+'_150_350.csv' #name of the file used
###############################################################################

###############################################################################
#Define constants
#Emission factors used for the emission estimates
EF_CO2 = 1620 # +- 70 g/kg_burned this value is for Tropical forest biome. Andreae, 2019
s_EF_CO2 = 70
EF_CO = 104 # +- 39 g/kg_burned this value is for Tropical forest biome. Andreae, 2019
s_EF_CO = 39
EF_CH4 = 6.5 # +- 1.6 g/kg_burned this value is for Tropical forest biome. Andreae, 2019
s_EF_CH4 = 1.6
EF_TPM = 8.7 # +- 3.1 g/kg_burned this value is for Tropical forest biome. Andreae, 2019
s_EF_TPM = 3.1

#Acording to Nguyen and Wooster, 2020 the species emission can be obtained by
#Ce_species = (EF_species/EF_TMP)*Ce_TPM

Ce_CO2 = (EF_CO2/EF_TPM) #kg/MJ 
Ce_CO = (EF_CO/EF_TPM)#kg/MJ
Ce_CH4 = (EF_CH4/EF_TPM) #kg/MJ

sigma_Ce_CO2 = np.sqrt((s_EF_CO2/EF_TPM)**2+(EF_CO2*s_EF_TPM/(EF_TPM)**2)**2) 
sigma_Ce_CO = np.sqrt((s_EF_CO/EF_TPM)**2+(EF_CO*s_EF_TPM/(EF_TPM)**2)**2) 
sigma_Ce_CH4 = np.sqrt((s_EF_CH4/EF_TPM)**2+(EF_CH4*s_EF_TPM/(EF_TPM)**2)**2) 
###############################################################################

###############################################################################
# Define a box informing the min and max latitudes and longitudes
# minlon,maxlon,minlat,maxlat=-57,-54,-9,-6 #Amazon small box
minlon,maxlon,minlat,maxlat=-72,-48,-11,-3 #Amazon definitive box 
###############################################################################

###############################################################################
#Define the file header
aux1 ='sat,year,julian,hhmm,code,central_lat,central_lon,FRP(MW),N_FRP,FRE(MJ),RE(kg/s),ME(kg),CO_2(kg),sigma_CO_2(kg),CO(kg),sigma_CO(kg),CH4(kg),sigma_CH4(kg),mean_FRP(MW),mean_temp(K)\n'
header = aux1
outstring=''
outfn = open(outfile, 'w')
outfn.writelines(header) 

###############################################################################

###############################################################################
# Initialize S3 file system that is responsible to access the GOES data from the cloud
fs = s3fs.S3FileSystem(anon=True)

###############################################################################,
#Starting message
print
print('Compiling statistics')
print 

#call the second function
rlat,rlon = get_lat_lon(fs)

Indexes,M = get_indexes_v3(minlon,maxlon,minlat,maxlat,rlat,rlon,feer_data)
print('Got indexes and matrix')

#Inform start and end year, days and hours
#Due to the volume of data in one year is recomended to process one year at a time
start_year,end_year,start_day,end_day,start_hour,end_our = Ano,Ano,150,350,0,24

#call the second function
data_list = get_files(start_year,end_year,start_day,end_day,start_hour,end_our)
print('Data listed')


print('Starting process data')
process_data_v6(rlat,rlon,data_list,M,Indexes)

#Done

