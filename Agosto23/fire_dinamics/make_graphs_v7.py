# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:40:48 2023

@author: thiag
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 19:34:16 2023

@author: thiagovg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:14:52 2023

@author: thiagovg
"""


import pandas as pd
from matplotlib import pyplot
import glob
import numpy as np
import math as mt
from scipy import stats
from scipy.optimize import curve_fit
# import uncertainties.unumpy as unp
# import uncertainties as unc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from scipy.signal import chirp, find_peaks, peak_widths,find_peaks_cwt
# from timezonefinder import TimezoneFinder
# from timezones import tz_utils


datadir = 'C:/Users/thiag/OneDrive/Documentos/Projeto_Mestrado/Agosto23/fire_dinamics/output' #Windows Version
# datadir = '/home/thiagovg//Documentos/Projeto_Mestrado/Maio23/FRP_amazon/fire_dinamics_graphs/output' #Ubuntu Version

# data_FRP = sorted(glob.glob(datadir+'/*FRP.csv'))
# data_TEMP = sorted(glob.glob(datadir+'/*temp.csv'))
# data_DQF = sorted(glob.glob(datadir+'/*DQF.csv'))
# data_Area = sorted(glob.glob(datadir+'/*Area.csv'))
# data = sorted(glob.glob(datadir+'/*rate_test_cerrado_box_v7.csv'))
Ano=2020
data = sorted(glob.glob(datadir+'/*'+str(Ano)+'_full.csv'))
# data = sorted(glob.glob(datadir+'/*fire_dinamics_amazon_small_box_full_v1.csv'))


# dados_FRP =  pd.read_csv(data_FRP[0])
# dados_TEMP =  pd.read_csv(data_TEMP[0])
# dados_DQF =  pd.read_csv(data_DQF[0])
# dados_Area = pd.read_csv(data_Area[0])
Data_fire = pd.read_csv(data[0])
# aux1 ='sat,year,julian,hhmm,code,central_lat,central_lon,FRP(MW),N_FRP\n'
aux1 ='sat,year,julian,hhmm,code,central_lat,central_lon,FRP(MW),N_FRP,Temp,N_temp,N_smold,N_flame\n'


# tf = TimezoneFinder()

header=aux1

def get_data(matrix,dados_emission):
    header = list(dados_emission)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    pos3=header.index('FRP(MW)')
    column3=dados_emission.iloc[:,pos3]
    FRP = np.array(column3.dropna(), dtype=pd.Series)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    # pos5=header.index('year')
    # column5=dados_emission.iloc[:,pos5]
    # year = np.array(column5.dropna(), dtype=pd.Series)
    
    # pos6=header.index('julian')
    # column6=dados_emission.iloc[:,pos6]
    # julian = np.array(column6.dropna(), dtype=pd.Series)
    
    # code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    
    index = np.where((lat==matrix[0,0]) & (lon==matrix[0,1]) & (FRP>=0) &(code<350)&(code>150))
    
    FRP_aux = FRP[index]
    
    peaks, _ = find_peaks(FRP_aux,height=75,distance=50, prominence=750)
    results_width = peak_widths(FRP_aux, peaks, rel_height=0.95)
    fire_duration = (results_width[0]*10)/60
    peak_start = np.int_(np.floor(results_width[2][:]))
    peak_end = np.int_(np.ceil(results_width[3][:]))
    
    for i in range(1,len(matrix)):
        # index = np.where((lat==matrix[i,0]) & (lon==matrix[i,1]) & (FRP>=0))
        index = np.where((lat==matrix[i,0]) & (lon==matrix[i,1]) & (FRP>=0) &(code<350)&(code>150))
        # lon_index = np.where(lon==-56.75)
        # print(index)
        FRP_aux = FRP[index]
        peaks_aux, _ = find_peaks(FRP_aux,height=75,distance=50, prominence=750)
        results_width_aux = peak_widths(FRP_aux, peaks_aux, rel_height=0.95)
        fire_duration_aux = (results_width_aux[0]*10)/60
        # print(fire_duration_aux)
        peak_start_aux = np.int_(np.floor(results_width_aux[2][:]))
        peak_end_aux = np.int_(np.ceil(results_width_aux[3][:]))
        
        
        fire_duration = np.append(fire_duration,fire_duration_aux)
        peaks = np.append(peaks,peaks_aux) 
        peak_start = np.append(peak_start,peak_start_aux)
        peak_end = np.append(peak_end,peak_end_aux)
    print(len(fire_duration))
    print(len(peaks))
    print(len(peak_start))
    print(len(peak_end))
    return fire_duration,peaks,peak_start,peak_end

def fire_duration_graph(fire_duration,filename):
    
    pyplot.clf()
    pyplot.title(str(Ano))
    pyplot.ylabel('Frequência Relativa')
    pyplot.xlabel('Fire duration hours')
    
    # print('Duração média dos focos durante o ano em hrs')
    # print(np.mean(fire_duration))
    
    pyplot.hist(fire_duration, bins=range(int(min(fire_duration)), 17, 1),density=True)
    Mean = 'Média: {:.2f}hrs\nMediana: {:.2f}\nN: {:.0f}'.format(np.mean(fire_duration),np.median(fire_duration),len(fire_duration))
    pyplot.legend([Mean],loc='best')

    pyplot.savefig(filename+'.pdf',dpi=200)
    
    return

def hist_start_to_peak_graph(peaks,peak_start,filename):
    
    pyplot.clf()
    pyplot.title(str(Ano))
    pyplot.xlabel('Início ao pico em horas')
    pyplot.ylabel('Frequência Relativa')
       
    # print(peaks)
    # print(peak_start)
    # diff_start_to_peak =  ((peaks - peak_start)*10)/60
    diff_start_to_peak =  np.subtract(peaks,peak_start)
    diff_start_to_peak_minutes =  diff_start_to_peak*10
    diff_start_to_peak_hours = diff_start_to_peak_minutes/60
    print(np.max(diff_start_to_peak_hours))
       
    pyplot.hist(diff_start_to_peak_hours, bins=range(int(min(diff_start_to_peak_hours)), 11, 1),density=True)
    # pyplot.hist(diff_start_to_peak_hours,bins=18)
    # pyplot.hist()
    Mean = 'Média: {:.2f}hrs\nMediana: {:.2f}\nN: {:.0f}'.format(np.mean(diff_start_to_peak_hours),np.median(diff_start_to_peak_hours),len(diff_start_to_peak_hours))
    pyplot.legend([Mean],loc='best')
       
    pyplot.savefig(filename+'.pdf',dpi=200)
    
    return

def hist_peak_to_end_graph(peaks,peak_end,filename):
    pyplot.clf()
    pyplot.title(str(Ano))
    pyplot.xlabel('Pico ao fim em horas')
    pyplot.ylabel('Frequência Relativa')

    # print(peaks)
    # print(peak_start)
    # diff_start_to_peak =  ((peaks - peak_start)*10)/60
    diff_end_to_peak =  np.subtract(peak_end,peaks)
    diff_end_to_peak_minutes =  diff_end_to_peak*10
    diff_end_to_peak_hours = diff_end_to_peak_minutes/60
    print(np.max(diff_end_to_peak_hours))

    pyplot.hist(diff_end_to_peak_hours, bins=range(int(min(diff_end_to_peak_hours)), 11, 1),density=True)
    # pyplot.hist(diff_start_to_peak_hours,bins=18)
    # pyplot.hist()
    Mean = 'Média: {:.2f}hrs\nMediana: {:.2f}\nN: {:.0f}'.format(np.mean(diff_end_to_peak_hours),np.median(diff_end_to_peak_hours),len(diff_end_to_peak_hours))
    pyplot.legend([Mean],loc='best')

    pyplot.savefig(filename+'.pdf',dpi=200)
    return

def FRP_stats_v4(dados_emission,filename):
    header = list(dados_emission)
    
    # pos1=header.index('code')
    # column1=dados_emission.iloc[:,pos1]
    # code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    pos3=header.index('FRP(MW)')
    column3=dados_emission.iloc[:,pos3]
    FRP = np.array(column3.dropna(), dtype=pd.Series)
    
    pos4=header.index('hhmm')
    column4=dados_emission.iloc[:,pos4]
    hhmm = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5=header.index('year')
    column5=dados_emission.iloc[:,pos5]
    year = np.array(column5.dropna(), dtype=pd.Series)
    
    pos6=header.index('julian')
    column6=dados_emission.iloc[:,pos6]
    julian = np.array(column6.dropna(), dtype=pd.Series)
    
    code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('FRP(MW)')
    # 7.75,-55.25
    index = np.where((lat==-7.75) & (lon==-55.25) & (FRP>0))
    FRP_index = FRP[index]
    code_full = code2[index]
    # sat = sat[index]
    pyplot.ylim(np.min(FRP_index)*1.1,np.max(FRP_index)*1.1)
    pyplot.plot(code_full,FRP_index, '*', markersize=3,label='FRP')
    
    # peaks = []
    # for j in range(0,4):
    #     #& (code>=2021+(j/10)) &(code<2021+(j+1)/10)
    #     index_partial = np.where((lat==-7.75) & (lon==-55.25) & (FRP>0))
    #     FRP_index_partial = FRP[index_partial]
    #     code_partial = code2[index_partial]
    #     peaks_partial, _ = find_peaks(FRP_index_partial,height=75,distance=50, prominence=750)# 35->75
        
        
    #     results_partial = peak_widths(FRP_index_partial, peaks_partial, rel_height=0.8)
    #     print(results_partial[0])
    #     print(len(results_partial))
        
    #     print('Duração de fogo de cada pico em hrs')
    #     print((results_partial[0]*10)/60)
    #     # print(results_half)
    #     # # print(results_half[2][28])
    #     # # print(results_half[3][28])
    #     print('Duração média dos focos durante o ano em hrs')
    #     print(np.mean((results_partial[0]*10)/60))
    #     peaks = np.append(peaks,peaks_partial)
    #     if len(results_partial[0]) > 0:
    #         positions = np.arange(0,len(FRP_index_partial))
    #         # print(positions)
    #         diff_start = np.absolute(positions-results_partial[2][0])
    #         diff_end = np.absolute(positions-results_partial[3][0])
    #         index_diff_start = diff_start.argmin()
    #         index_diff_end = diff_end.argmin()
    #         for i in range(1,len(peaks_partial)):
    #             diff_start = np.absolute(positions-results_partial[2][i])
    #             diff_end = np.absolute(positions-results_partial[3][i])
    #             index_diff_start_aux = diff_start.argmin()
    #             index_diff_end_aux = diff_end.argmin()
    #             index_diff_start = np.append(index_diff_start,index_diff_start_aux)
    #             index_diff_end = np.append(index_diff_end,index_diff_end_aux)
    #         pyplot.plot(code_partial[peaks_partial], FRP_index_partial[peaks_partial], "x")
    #         pyplot.hlines(results_partial[0],code_partial[index_diff_start],code_partial[index_diff_end], color="C2")
        

    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def FRP_stats_v5(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code2 = np.array(column1.dropna(), dtype=pd.Series)
    
    # pos2=header.index('central_lat')
    # column2=dados_emission.iloc[:,pos2]
    # lat = np.array(column2.dropna(), dtype=pd.Series)
    
    # pos4=header.index('central_lon')
    # column4=dados_emission.iloc[:,pos4]
    # lon = np.array(column4.dropna(), dtype=pd.Series)

    pos3=header.index('FRP(MW)')
    column3=dados_emission.iloc[:,pos3]
    FRP = np.array(column3.dropna(), dtype=pd.Series)
    
    # pos4=header.index('hhmm')
    # column4=dados_emission.iloc[:,pos4]
    # hhmm = np.array(column4.dropna(), dtype=pd.Series)
    
    # pos5=header.index('year')
    # column5=dados_emission.iloc[:,pos5]
    # year = np.array(column5.dropna(), dtype=pd.Series)
    
    # pos6=header.index('julian')
    # column6=dados_emission.iloc[:,pos6]
    # julian = np.array(column6.dropna(), dtype=pd.Series)
    
    # code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    

    pyplot.clf()
    pyplot.title(str(Ano))
    pyplot.ylabel('FRP(MW)')
    pyplot.xlabel('Dias Julianos')
    # 7.75,-55.25
    # index = np.where((lat==-7.75) & (lon==-55.25) & (FRP>=0))
    index = np.where( (FRP>=0))
    FRP_index = FRP[index]
    code_full = code2[index]
    # sat = sat[index]
    pyplot.ylim(np.min(FRP_index)*1.1,np.max(FRP_index)*1.1)
    pyplot.plot(code_full,FRP_index, '*', markersize=3,color='black',label='Valor total de FRP na região')
    pyplot.legend(loc='best')
    # peaks = []
    # for j in range(0,4):
        #& (code>=2021+(j/10)) &(code<2021+(j+1)/10)
    # index_partial = np.where((lat==-7.75) & (lon==-55.25) & (FRP>0))
    # FRP_index_partial = FRP[index_partial]
    # code_partial = code2[index_partial]
    # peaks, _ = find_peaks(FRP_index,height=75,distance=50, prominence=750)# 35->75
    
    
    # results_partial = peak_widths(FRP_index, peaks, rel_height=0.95)
    # print(results_partial[0])
    # print(len(results_partial))
    
    # print('Duração de fogo de cada pico em hrs')
    # print((results_partial[0]*10)/60)
    # # print(results_half)
    # # # print(results_half[2][28])
    # # # print(results_half[3][28])
    # print('Duração média dos focos durante o ano em hrs')
    # print(np.mean((results_partial[0]*10)/60))
    # peaks = np.append(peaks,peaks_partial)
    # if len(results_partial[0]) > 0:
    # positions = np.arange(0,len(FRP_index_partial))
    # print(positions)
    
    # diff_start = np.absolute(positions-results_partial[2][0])
    # diff_end = np.absolute(positions-results_partial[3][0])
    # index_diff_start = diff_start.argmin()
    # index_diff_end = diff_end.argmin()
    
    # peak_start_pos = np.int_(np.floor(results_partial[2][:]))
    # peak_end_pos = np.int_(np.ceil(results_partial[3][:]))

    
    # for i in range(1,len(peaks_partial)):
    #     diff_start = np.absolute(positions-results_partial[2][i])
    #     diff_end = np.absolute(positions-results_partial[3][i])
    #     index_diff_start_aux = diff_start.argmin()
    #     index_diff_end_aux = diff_end.argmin()
    #     index_diff_start = np.append(index_diff_start,index_diff_start_aux)
    #     index_diff_end = np.append(index_diff_end,index_diff_end_aux)
    # pyplot.plot(code_full[peaks], FRP_index[peaks], "x")
    # pyplot.hlines(results_partial[0],code_full[peak_start_pos],code_full[peak_end_pos], color="C2")


    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return



def get_indexes_v2(min_lon,max_lon,min_lat,max_lat):
    centers_lon = np.linspace(min_lon+0.25, max_lon-0.25,num=int(max_lon-min_lon)*2)
    centers_lat = np.linspace(min_lat+0.25, max_lat-0.25,num=int(max_lat-min_lat)*2)
    
    for i in range(0,len(centers_lat)):
        # df2 = dados_feer.loc[(dados_feer['Latitude'] == centers_lat[i]) & (dados_feer['Longitude'] <= maxlon )
        #                    & (dados_feer['Longitude'] >= minlon ),'Ce_850'].to_numpy()
        # aux_list = np.append(aux_list,df2)
        if i == 0:
            aux = np.repeat(centers_lat[i],len(centers_lon))
            lat_lon_feer = np.column_stack((aux,centers_lon))
            # latlon = np.vstack((latlon,au2))
        else:
            # print(i)
            aux = np.repeat(centers_lat[i],len(centers_lon))
            aux2 = np.column_stack((aux,centers_lon))
            lat_lon_feer = np.vstack((lat_lon_feer,aux2))
    print(lat_lon_feer)
    matrix = lat_lon_feer

    return matrix

def Temp_stats_v5(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    pos3=header.index('Temp')
    column3=dados_emission.iloc[:,pos3]
    temp = np.array(column3.dropna(), dtype=pd.Series)
    
    # pos4=header.index('hhmm')
    # column4=dados_emission.iloc[:,pos4]
    # code = np.array(column4.dropna(), dtype=pd.Series)
    
    # pos5=header.index('year')
    # column5=dados_emission.iloc[:,pos5]
    # year = np.array(column5.dropna(), dtype=pd.Series)
    
    # pos6=header.index('julian')
    # column6=dados_emission.iloc[:,pos6]
    # julian = np.array(column6.dropna(), dtype=pd.Series)
    
    # code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('Temp(K)')
    # 7.75,-55.25
    index = np.where((lat==-7.75) & (lon==-55.25) & (temp>=0))
    index_smoldering = np.where((lat==-7.75) & (lon==-55.25) & (temp>=500) & (temp<=700))
    index_flamming = np.where((lat==-7.75) & (lon==-55.25) & (temp>=800) & (temp<=1200))
    temp_index = temp[index]
    code_full = code[index]
    temp_flame = temp[index_flamming]
    temp_smold = temp[index_smoldering]
    code_flame = code[index_flamming]
    code_smold = code[index_smoldering]
    # sat = sat[index]
    # pyplot.ylim(np.min(temp_index)*1.1,np.max(temp_index)*1.1)
    pyplot.plot(code_full,temp_index, '*', markersize=3,label='Temp')
    pyplot.plot(code_smold,temp_smold, '*', markersize=3,color='purple',label='Temp_smoldering')
    pyplot.plot(code_flame,temp_flame, '.', markersize=3,color='red',label='Temp_flaming')



    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def N_frpXN_temp(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('N_FRP')
    column1=dados_emission.iloc[:,pos1]
    N_frp = np.array(column1.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    pos3=header.index('N_temp')
    column3=dados_emission.iloc[:,pos3]
    N_temp = np.array(column3.dropna(), dtype=pd.Series)
    
    # pos4=header.index('hhmm')
    # column4=dados_emission.iloc[:,pos4]
    # code = np.array(column4.dropna(), dtype=pd.Series)
    
    # pos5=header.index('year')
    # column5=dados_emission.iloc[:,pos5]
    # year = np.array(column5.dropna(), dtype=pd.Series)
    
    # pos6=header.index('julian')
    # column6=dados_emission.iloc[:,pos6]
    # julian = np.array(column6.dropna(), dtype=pd.Series)
    
    # code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.xlabel('N_FRP')
    pyplot.ylabel('N_TEMP')
    # 7.75,-55.25
    index = np.where((lat==-7.75) & (lon==-55.25))
    N_frp_index = N_frp[index]
    N_temp_index = N_temp[index]
    # sat = sat[index]
    # pyplot.ylim(np.min(temp_index)*1.1,np.max(temp_index)*1.1)
    pyplot.plot(N_frp_index,N_temp_index, '*', markersize=3)



    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def N_flame_smold(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    
    # pos4=header.index('hhmm')
    # column4=dados_emission.iloc[:,pos4]
    # code = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5=header.index('N_flame')
    column5=dados_emission.iloc[:,pos5]
    N_flame = np.array(column5.dropna(), dtype=pd.Series)
    
    pos6=header.index('N_smold')
    column6=dados_emission.iloc[:,pos6]
    N_smold = np.array(column6.dropna(), dtype=pd.Series)
    
    # code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.xlabel('code')
    pyplot.ylabel('N_flaming & N_smoldering')
    # 7.75,-55.25
    index = np.where((lat==-7.75) & (lon==-55.25))
    code_full = code[index]
    N_flame_index = N_flame[index]
    N_smold_index = N_smold[index]

    # sat = sat[index]
    # pyplot.ylim(np.min(temp_index)*1.1,np.max(temp_index)*1.1)
    # pyplot.plot(code_full,temp_index, '*', markersize=3,label='Temp')
    pyplot.plot(code_full,N_smold_index, '*', markersize=3,color='purple',label='N_smoldering')
    pyplot.plot(code_full,N_flame_index, '.', markersize=3,color='red',label='N_flaming')



    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def N_flame_smold_hist_box(dados_emission,filename,matrix):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    
    pos4=header.index('N_temp')
    column4=dados_emission.iloc[:,pos4]
    N_temp = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5=header.index('N_flame')
    column5=dados_emission.iloc[:,pos5]
    N_flame = np.array(column5.dropna(), dtype=pd.Series)
    
    pos6=header.index('N_smold')
    column6=dados_emission.iloc[:,pos6]
    N_smold = np.array(column6.dropna(), dtype=pd.Series)
    
    # code2 = year + (julian+(hhmm/100)/24+(hhmm % 100)/60/24)/365
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.xlabel('Proportion N_flame/N_temp')
    # pyplot.ylabel('N_flaming & N_smoldering')
    # 7.75,-55.25
    # index = np.where((lat==-7.75) & (lon==-55.25))
    # code_full = code[index]
    # N_flame_index = N_flame[index]
    # N_smold_index = N_smold[index]
    
    
    index = np.where((lat==matrix[0,0]) & (lon==matrix[0,1]) & (N_flame>0) & (N_smold>0) )
    # lon_index = np.where(lon==-56.75)
    # print(index)
    N_temp_index = N_temp[index]
    N_flame_index = N_flame[index]
    N_smold_index = N_smold[index]
    prop_flame = N_flame_index/N_temp_index
    prop_smold = N_smold_index/N_temp_index

    array_prop_flame = prop_flame*100
    array_prop_smold = prop_smold*100
    
    for i in range(1,len(matrix)):
        index = np.where((lat==matrix[i,0]) & (lon==matrix[i,1]) & (N_flame>0) & (N_smold>0) )
        # print(index)
        N_temp_index = N_temp[index]
        N_flame_index = N_flame[index]
        N_smold_index = N_smold[index]
        prop_flame = N_flame_index/N_temp_index
        prop_smold = N_smold_index/N_temp_index
    
        array_prop_flame_aux = prop_flame*100
        array_prop_smold_aux = prop_smold*100
        
        array_prop_flame = np.append(array_prop_flame,array_prop_flame_aux)
        array_prop_smold = np.append(array_prop_smold,array_prop_smold_aux)
    # fire_duration = np.append(fire_duration,fire_duration_partial)

    # sat = sat[index]
    # pyplot.ylim(np.min(temp_index)*1.1,np.max(temp_index)*1.1)
    # pyplot.plot(code_full,temp_index, '*', markersize=3,label='Temp')
    # pyplot.plot(code_full,N_smold_index, '*', markersize=3,color='purple',label='N_smoldering')
    # pyplot.plot(code_full,N_flame_index, '.', markersize=3,color='red',label='N_flaming')
    pyplot.hist(array_prop_flame)
    pyplot.savefig(filename+'flame.png',dpi=200)
    pyplot.clf()
    pyplot.xlabel('Proportion N_smold/N_temp')
    pyplot.hist(array_prop_smold)
    pyplot.savefig(filename+'smold.png',dpi=200)


    
##########################################################################################################

    return

# minlon,maxlon,minlat,maxlat=-57,-54,-9,-6 #Amazon small box
minlon,maxlon,minlat,maxlat=-72,-48,-11,-3 #Amazon definitive box
# minlon,maxlon,minlat,maxlat=-72,-48,-9,-6 #Amazon big box v2
M = get_indexes_v2(minlon,maxlon,minlat,maxlat)
fire_durations,position_peaks,position_peaks_start,position_peak_end = get_data(M,Data_fire)
fire_duration_graph(fire_durations,'Fire_durations_histogram_'+str(Ano)+'_definitive_box_150_350')
hist_start_to_peak_graph(position_peaks,position_peaks_start,'Start_to_peak_histogram_'+str(Ano)+'_definitive_box_150_350')
hist_peak_to_end_graph(position_peaks,position_peak_end,'Peak_to_end_histogram_'+str(Ano)+'_definitive_box_150_350')
# fire_duration_x_FRP_emitted_v2(Data_fire,'fire_duration_x_FRP_emitted_'+str(Ano)+'_definitive_box',M) #library problem on windows
###############################################################################
# N_flame_smold(dados_emission,'N_flame_N_smold')
# N_frpXN_temp(dados_emission,'N_frp_N_temp')
# Temp_stats_v5(dados_emission,'mean_temp')
# N_flame_smold_hist_box(dados_emission,'proportion_hist',M)
##############################################################################
#Plots for dissertation
FRP_stats_v5(Data_fire, 'Plot_FRP_definitive_box_amazon_'+str(Ano)+'_full_no_peaks')

print('Done')

#=julian+TRUNCAR(hhmm/100,0)/24+mod(hhmm,100)/60/24