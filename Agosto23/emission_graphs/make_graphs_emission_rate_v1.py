# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:07:55 2023

@author: thiag
"""



import pandas as pd
from matplotlib import pyplot
import glob
import numpy as np
import math as mt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from scipy.signal import chirp, find_peaks, peak_widths,find_peaks_cwt
from numba import jit
from matplotlib import cm
from matplotlib.colors import LightSource
# from timezonefinder import TimezoneFinder
# from timezones import tz_utils

# tf = TimezoneFinder()


datadir = 'C:/Users/thiag/OneDrive/Documentos/Projeto_Mestrado/Agosto23/fire_dinamics/output'

# data_FRP = sorted(glob.glob(datadir+'/*FRP.csv'))
# data_TEMP = sorted(glob.glob(datadir+'/*temp.csv'))
# data_DQF = sorted(glob.glob(datadir+'/*DQF.csv'))
# data_Area = sorted(glob.glob(datadir+'/*Area.csv'))
# data = sorted(glob.glob(datadir+'/*rate_test_cerrado_box_v7.csv'))
Ano=2020
data = sorted(glob.glob(datadir+'/*'+str(Ano)+'_full.csv'))

# dados_FRP =  pd.read_csv(data_FRP[0])
# dados_TEMP =  pd.read_csv(data_TEMP[0])
# dados_DQF =  pd.read_csv(data_DQF[0])
# dados_Area = pd.read_csv(data_Area[0])
dados_emission = pd.read_csv(data[0])
aux1 ='sat,year,julian,hhmm,code,central_lat,central_lon,FRP(MW),N_FRP,FRE(MJ),RE(kg/s),ME(kg),CO_2(Kg),sigma_CO_2(Kg),CO(kg),sigma_CO(Kg),CH4(kg),sigma_CH4(Kg)\n'


header='sat,year,julian,hhmm,code,central_lat,central_lon,FRP(MW),N_FRP,FRE(MJ),RE(kg/s),ME(kg),CO_2(Kg),sigma_CO_2(Kg),CO(kg),sigma_CO(Kg),CH4(kg),sigma_CH4(Kg)'

def CO2_stats(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    

    pos3=header.index('CO_2(Kg)')
    column3=dados_emission.iloc[:,pos3]
    co2 = np.array(column3.dropna(), dtype=pd.Series)
    
    pos4=header.index('sigma_CO_2(Kg)')
    column4=dados_emission.iloc[:,pos4]
    sigma_co2 = np.array(column4.dropna(), dtype=pd.Series)
    

    pyplot.clf()
    
    pyplot.ylabel('CO_2(kg)')
    pyplot.title(str(Ano))
    pyplot.xlabel('Dias Julianos')
    
    index = np.where( (code<350)&(code>150))
    co2 = co2[index]
    sigma_co2 = sigma_co2[index]
    code = code[index]
    
    # print(np.max(co2))
    # print(type(np.max(co2)))
    pyplot.ylim(np.min(co2),np.max(co2)*1.1)
    #pyplot.plot(code, co2, '*', markersize=3,label='temp_avg')
    pyplot.errorbar(code, co2, yerr=sigma_co2,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO_2')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'test.png',dpi=200)
##########################################################################################################

    return


def CO_stats(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    

    pos3=header.index('CO(kg)')
    column3=dados_emission.iloc[:,pos3]
    co = np.array(column3.dropna(), dtype=pd.Series)
    
    pos4=header.index('sigma_CO(Kg)')
    column4=dados_emission.iloc[:,pos4]
    sigma_co = np.array(column4.dropna(), dtype=pd.Series)
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('CO(kg)')
    pyplot.title(str(Ano))
    pyplot.xlabel('Dias Julianos')
    
    
    pyplot.ylim(np.min(co),np.max(co)*1.1)
    #pyplot.plot(code, co2, '*', markersize=3,label='temp_avg')
    pyplot.errorbar(code, co, yerr=sigma_co,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'test.png',dpi=200)
##########################################################################################################

    return

def CH4_stats(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('CH4(kg)')
    column3=dados_emission.iloc[:,pos3]
    CH4 = np.array(column3.dropna(), dtype=pd.Series)
    # ,CH4(kg),sigma_CH4(Kg)
    
    
    pos4=header.index('sigma_CH4(Kg)')
    column4=dados_emission.iloc[:,pos4]
    sigma_ch4 = np.array(column4.dropna(), dtype=pd.Series)
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.title(str(Ano))
    pyplot.xlabel('Dias Julianos')

    
    
    pyplot.ylim(np.min(CH4),np.max(CH4)*1.1)
    #pyplot.plot(code, co2, '*', markersize=3,label='temp_avg')
    pyplot.errorbar(code, CH4, yerr=sigma_ch4,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'test.png',dpi=200)
##########################################################################################################

    return

def TPM_stats(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('ME(kg)')
    column3=dados_emission.iloc[:,pos3]
    TPM = np.array(column3.dropna(), dtype=pd.Series)
    
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.title(str(Ano))
    pyplot.xlabel('Dias Julianos')
    
    
    pyplot.ylim(np.min(TPM),np.max(TPM)*1.1)
    pyplot.plot(code, TPM, '*', markersize=3,label='TPM')
    #pyplot.errorbar(code, ch4, yerr=sigma_ch4,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def TPM_stats_v2(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('ME(kg)')
    column3=dados_emission.iloc[:,pos3]
    TPM = np.array(column3.dropna(), dtype=pd.Series)
    
    
    index = np.where((code>=2022) & (code<2023))
    # print(code[index])    
    # print(len(code[index]))
    newcode = np.unique(code[index])
    print(newcode)    
    print(len(newcode))
    TPM_days = np.sum(TPM[code==newcode[0]])
    for i in range(1,len(newcode)):
        TPM_days_aux = np.sum(TPM[code==newcode[i]])
        TPM_days = np.append(TPM_days,TPM_days_aux)
    print(TPM_days)    
    print(len(TPM_days))
    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('TPM(kg)')
    pyplot.xlabel('Percentage of the year')
    
    
    pyplot.ylim(0,np.max(TPM_days)*1.1)
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    pyplot.plot(newcode, TPM_days, '*', markersize=3,label='TPM')
    Total = 'Total emission: {:e} Tg'.format(np.sum(TPM_days[TPM_days>0])*1e-9)
    pyplot.legend([Total],loc='best')
    #pyplot.errorbar(code, ch4, yerr=sigma_ch4,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def TPM_stats_v3(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('ME(kg)')
    column3=dados_emission.iloc[:,pos3]
    TPM = np.array(column3.dropna(), dtype=pd.Series)
    
    
    index = np.where((code>=2021) & (code<2022))
    # print(code[index])    
    # print(len(code[index]))
    newcode = np.unique(code[index])
    print(newcode)    
    print(len(newcode))
    indexes=[]
    for i in range(0,len(newcode)):
        indexes_aux=np.where(code==newcode[i])
        indexes.insert(i,indexes_aux)
    print('indexes finish')
    TPM_days = np.sum(TPM[indexes[0]])
    for j in range(1,len(indexes)):
        TPM_days_aux = np.sum(TPM[indexes[j]])
        TPM_days = np.append(TPM_days,TPM_days_aux)
    print(TPM_days)    
    print(len(TPM_days))
    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('TPM(kg)')
    pyplot.xlabel('Percentage of the year')
    
    
    pyplot.ylim(0,np.max(TPM_days)*1.1)
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    pyplot.plot(newcode, TPM_days, '*', markersize=3,label='TPM')
    Total = 'Total emission: {:.2e} Tg'.format(np.sum(TPM_days[TPM_days>0])*1e-9)
    pyplot.legend([Total],loc='best')
    #pyplot.errorbar(code, ch4, yerr=sigma_ch4,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def CO2_stats_v2(dados_emission,filename,indexes,Newcode):
    header = list(dados_emission)
    
    
    pos3=header.index('CO_2(Kg)')
    column3=dados_emission.iloc[:,pos3]
    co2 = np.array(column3.dropna(), dtype=pd.Series)
    
    pos4=header.index('sigma_CO_2(Kg)')
    column4=dados_emission.iloc[:,pos4]
    sigma_co2 = np.array(column4.dropna(), dtype=pd.Series)
    

    co2_days = np.sum(co2[indexes[0]])
    sigma_co2_days =  np.sqrt(np.sum(sigma_co2[indexes[0]]**2))
    for j in range(1,len(indexes)):
        co2_days_aux = np.sum(co2[indexes[j]])
        sigma_co2_days_aux =  np.sqrt(np.sum(sigma_co2[indexes[j]]**2))
        co2_days = np.append(co2_days,co2_days_aux)
        sigma_co2_days = np.append(sigma_co2_days,sigma_co2_days_aux)
    # print(TPM_days)    
    # print(len(TPM_days))
    
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('CO_2(kg)')
    pyplot.xlabel('Percentage of the year')
    
    # print(np.max(co2))
    # print(type(np.max(co2)))
    pyplot.ylim(0,np.max(co2_days)*1.1)
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #pyplot.plot(code, co2, '*', markersize=3,label='temp_avg')
    pyplot.errorbar(Newcode, co2_days, yerr=sigma_co2_days,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO_2')
    Total = 'Total emission: {:.2e} Tg'.format(np.sum(co2_days[co2_days>0])*1e-9)
    pyplot.legend([Total],loc='best')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'test.png',dpi=200)
##########################################################################################################

    return

def CO_stats_v2(dados_emission,filename,indexes,Newcode):
    header = list(dados_emission)
    
    
    pos3=header.index('CO(kg)')
    column3=dados_emission.iloc[:,pos3]
    co = np.array(column3.dropna(), dtype=pd.Series)
    
    pos4=header.index('sigma_CO(Kg)')
    column4=dados_emission.iloc[:,pos4]
    sigma_co = np.array(column4.dropna(), dtype=pd.Series)
    

    co_days = np.sum(co[indexes[0]])
    sigma_co_days =  np.sqrt(np.sum(sigma_co[indexes[0]]**2))
    for j in range(1,len(indexes)):
        co_days_aux = np.sum(co[indexes[j]])
        sigma_co_days_aux =  np.sqrt(np.sum(sigma_co[indexes[j]]**2))
        co_days = np.append(co_days,co_days_aux)
        sigma_co_days = np.append(sigma_co_days,sigma_co_days_aux)
    # print(TPM_days)    
    # print(len(TPM_days))
    
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('CO(kg)')
    pyplot.xlabel('Percentage of the year')
    
    # print(np.max(co2))
    # print(type(np.max(co2)))
    pyplot.ylim(0,np.max(co_days)*1.1)
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #pyplot.plot(code, co2, '*', markersize=3,label='temp_avg')
    pyplot.errorbar(Newcode, co_days, yerr=sigma_co_days,ls ='none', marker = '.',markersize=4,ecolor='red',label='CO_2')
    Total = 'Total emission: {:.2e} Tg'.format(np.sum(co_days[co_days>0])*1e-9)
    pyplot.legend([Total],loc='best')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'test.png',dpi=200)
##########################################################################################################

    return

def CH4_stats_v2(dados_emission,filename,indexes,Newcode):
    header = list(dados_emission)
    
    
    pos3=header.index('CH4(kg)')
    column3=dados_emission.iloc[:,pos3]
    CH4 = np.array(column3.dropna(), dtype=pd.Series)
    
    pos4=header.index('sigma_CH4(Kg)')
    column4=dados_emission.iloc[:,pos4]
    sigma_CH4 = np.array(column4.dropna(), dtype=pd.Series)
    

    CH4_days = np.sum(CH4[indexes[0]])
    sigma_CH4_days =  np.sqrt(np.sum(sigma_CH4[indexes[0]]**2))
    for j in range(1,len(indexes)):
        CH4_days_aux = np.sum(CH4[indexes[j]])
        sigma_CH4_days_aux =  np.sqrt(np.sum(sigma_CH4[indexes[j]]**2))
        CH4_days = np.append(CH4_days,CH4_days_aux)
        sigma_CH4_days = np.append(sigma_CH4_days,sigma_CH4_days_aux)
    # print(TPM_days)    
    # print(len(TPM_days))
    
    

    pyplot.clf()
    pyplot.title(filename)
    pyplot.ylabel('CH4(kg)')
    pyplot.xlabel('Percentage of the year')
    
    # print(np.max(co2))
    # print(type(np.max(co2)))
    pyplot.ylim(0,np.max(CH4_days)*1.1)
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #pyplot.plot(code, co2, '*', markersize=3,label='temp_avg')
    pyplot.errorbar(Newcode, CH4_days, yerr=sigma_CH4_days,ls ='none', marker = '.',markersize=4,ecolor='red',label='CH4')
    Total = 'Total emission: {:.2e} Tg'.format(np.sum(CH4_days[CH4_days>0])*1e-9)
    pyplot.legend([Total],loc='best')
    # pyplot.plot(code, temp_max,'.',markersize=3,label='max')
    # pyplot.plot(code, temp_min,'.',markersize=3,label='min')
    # pyplot.plot(hhmm, temp_N,markersize=3,label='N')
    # pyplot.legend(loc='right', bbox_to_anchor=(1,0.85), ncol=1,framealpha=0,fontsize = 8.5)
    #pyplot.legend()
    pyplot.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def TPM_mapping(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('ME(kg)')
    column3=dados_emission.iloc[:,pos3]
    TPM = np.array(column3.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    # index = np.where((code>=2021) & (code<2022))
    
    lats = np.unique(lat)
    lons = np.unique(lon)
    
    lats = lats[::-1]

    matrix_TPM = np.zeros( (6, 48) ,dtype='float64')
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    for k in range(0,len(lats)):
        for n in range(0,len(lons)):
            matrix_TPM[k,n]= np.sum(TPM[(lat==lats[k])&(lon==lons[n])&(code>=2021) & (code<2022)])
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    # X, Y = np.meshgrid(lats, lons)
    
    
    
    pyplot.clf()
    fig, ax = pyplot.subplots()
    # ax = pyplot.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))
    


# plt.colorbar(im, cax=cax)
    # pyplot.title(filename)
    # pyplot.ylabel('lat')
    # pyplot.xlabel('lon')
    
    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_TPM,cmap='inferno', extent = extend)
    

    
    #set aspect ratio to 8
    ratio = 1
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename,fontsize=8)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    pyplot.colorbar(im,label="TPM emisson(Kg)")
    
    # ax.set_aspect(0.01)
    
    pyplot.show()

    fig.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def TPM_mapping_v2(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('ME(kg)')
    column3=dados_emission.iloc[:,pos3]
    TPM = np.array(column3.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    area = 27.75**2
    
    # index = np.where((code>=2021) & (code<2022))
    
    lats = np.unique(lat)
    lons = np.unique(lon)
    
    lats = lats[::-1]
    
    matrix_TPM = np.zeros( (6, 48) ,dtype='float64')
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    for k in range(0,len(lats)):
        for n in range(0,len(lons)):
            matrix_TPM[k,n]= np.sum(TPM[(lat==lats[k])&(lon==lons[n])&(code>=2022) & (code<2023)])*1e-3/area
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    # X, Y = np.meshgrid(lats, lons)
    
    
    
    pyplot.clf()
    fig, ax = pyplot.subplots()
    # ax = pyplot.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))
    


# plt.colorbar(im, cax=cax)
    # pyplot.title(filename)
    # pyplot.ylabel('lat')
    # pyplot.xlabel('lon')
    
    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_TPM,cmap='inferno', extent = extend)
    

    
    #set aspect ratio to 8
    # ratio = 1
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename,fontsize=8)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    pyplot.colorbar(im,label="TPM emisson(Ton/km^2) 100 days")
    
    # ax.set_aspect(0.01)
    
    pyplot.show()

    fig.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def CO2_mapping(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('CO_2(Kg)')
    column3=dados_emission.iloc[:,pos3]
    CO2 = np.array(column3.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    area = 27.75**2
    
    
    lats = np.unique(lat)
    lons = np.unique(lon)
    
    lats = lats[::-1]
    
    matrix_CO2 = np.zeros( (6, 48) ,dtype='float64')
    # print(matrix_CO2)
    # print(matrix_CO2.shape)
    for k in range(0,len(lats)):
        for n in range(0,len(lons)):
            matrix_CO2[k,n]= np.sum(CO2[(lat==lats[k])&(lon==lons[n])&(code>=2021) & (code<2022)])*1e-3/area
    # print(matrix_CO2)
    # print(matrix_CO2.shape)
    X, Y = np.meshgrid(lats, lons)
    
    
    
    pyplot.clf()
    fig, ax = pyplot.subplots()
    # ax = pyplot.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))
    


# plt.colorbar(im, cax=cax)
    # pyplot.title(filename)
    # pyplot.ylabel('lat')
    # pyplot.xlabel('lon')
    
    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_CO2,cmap='BuPu', extent = extend)
    

    
    #set aspect ratio to 8
    ratio = 1
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename,fontsize=8)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = pyplot.colorbar(im,label="CO2 emisson(Ton/km^2) 100 days")
    cb.formatter.set_powerlimits((0, 0))
    # ax.set_aspect(0.01)
    
    pyplot.show()

    fig.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def CO_mapping(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('CO(kg)')
    column3=dados_emission.iloc[:,pos3]
    CO = np.array(column3.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5=header.index('julian')
    column5=dados_emission.iloc[:,pos5]
    julian = np.array(column5.dropna(), dtype=pd.Series)
    
    area = 27.75**2
    
    
    lats = np.unique(lat)
    lons = np.unique(lon)
    
    lats = lats[::-1]
    
    matrix_CO = np.zeros( (6, 48) ,dtype='float64')
    # print(matrix_CO)
    # print(matrix_CO.shape)
    for k in range(0,len(lats)):
        for n in range(0,len(lons)):
            matrix_CO[k,n]= np.sum(CO[(lat==lats[k])&(lon==lons[n])&(code>=2021) & (code<2022) & (julian == 247)])*1e-3/area
    # print(matrix_CO)
    # print(matrix_CO.shape)
    X, Y = np.meshgrid(lats, lons)
    
    
    
    pyplot.clf()
    fig, ax = pyplot.subplots()
    # ax = pyplot.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))
    


# plt.colorbar(im, cax=cax)
    # pyplot.title(filename)
    # pyplot.ylabel('lat')
    # pyplot.xlabel('lon')
    
    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_CO,cmap='BuPu', extent = extend)
    

    
    #set aspect ratio to 8
    ratio = 1
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename,fontsize=8)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    pyplot.colorbar(im,label="CO emisson(Ton/km^2) day")
    # cb.formatter.set_powerlimits((0, 0))
    # ax.set_aspect(0.01)
    
    pyplot.show()

    fig.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def CH4_mapping(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('CH4(kg)')
    column3=dados_emission.iloc[:,pos3]
    CH4 = np.array(column3.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    area = 27.75**2
    
    
    lats = np.unique(lat)
    lons = np.unique(lon)
    
    lats = lats[::-1]
    
    matrix_CH4 = np.zeros( (6, 48) ,dtype='float64')
    
    # print(matrix_CH4)
    # print(matrix_CH4.shape)
    for k in range(0,len(lats)):
        for n in range(0,len(lons)):
            matrix_CH4[k,n]= np.sum(CH4[(lat==lats[k])&(lon==lons[n])&(code>=2021) & (code<2022)])*1e-3/area
    # print(matrix_CH4)
    # print(matrix_CH4.shape)
    X, Y = np.meshgrid(lats, lons)
    
    
    
    pyplot.clf()
    fig, ax = pyplot.subplots()
    # ax = pyplot.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))
    


# plt.colorbar(im, cax=cax)
    # pyplot.title(filename)
    # pyplot.ylabel('lat')
    # pyplot.xlabel('lon')
    
    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_CH4,cmap='BuPu', extent = extend)
    

    
    #set aspect ratio to 8
    ratio = 1
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename,fontsize=8)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = pyplot.colorbar(im,label="CH4 emisson(Ton/km^2) 100 days")
    cb.formatter.set_powerlimits((0, 0))
    # ax.set_aspect(0.01)
    
    pyplot.show()

    fig.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

def N_mapping(dados_emission,filename):
    header = list(dados_emission)
    
    pos1=header.index('code')
    column1=dados_emission.iloc[:,pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)
    
    pos3=header.index('N_FRP')
    column3=dados_emission.iloc[:,pos3]
    N_FRP = np.array(column3.dropna(), dtype=pd.Series)
    
    pos2=header.index('central_lat')
    column2=dados_emission.iloc[:,pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)
    
    pos4=header.index('central_lon')
    column4=dados_emission.iloc[:,pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    
    lats = np.unique(lat)
    lons = np.unique(lon)
    lats = lats[::-1]
    
    matrix_N_FRP = np.zeros( (6, 48) ,dtype='float64')
    # print(matrix_CH4)
    # print(matrix_CH4.shape)
    for k in range(0,len(lats)):
        for n in range(0,len(lons)):
            matrix_N_FRP[k,n]= np.sum(N_FRP[(lat==lats[k])&(lon==lons[n])&(code>=2021) & (code<2022)])
    # print(matrix_CH4)
    # print(matrix_CH4.shape)
    # X, Y = np.meshgrid(lats, lons)
    
    
    
    pyplot.clf()
    fig, ax = pyplot.subplots()
    # ax = pyplot.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))
    


# plt.colorbar(im, cax=cax)
    # pyplot.title(filename)
    # pyplot.ylabel('lat')
    # pyplot.xlabel('lon')
    
    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_N_FRP,cmap='viridis', extent = extend)
    

    
    #set aspect ratio to 8
    ratio = 1
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename,fontsize=8)
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    pyplot.colorbar(im,label="N fires 100 days")
    # cb.formatter.set_powerlimits((0, 0))
    # ax.set_aspect(0.01)
    
    pyplot.show()

    fig.savefig(filename+'.png',dpi=200)
##########################################################################################################

    return

# def Start_hours_mapping(dados_emission,filename):
#     header = list(dados_emission)
    
#     pos1=header.index('code')
#     column1=dados_emission.iloc[:,pos1]
#     code = np.array(column1.dropna(), dtype=pd.Series)
    
#     pos3=header.index('N_FRP')
#     column3=dados_emission.iloc[:,pos3]
#     N_FRP = np.array(column3.dropna(), dtype=pd.Series)
    
#     pos2=header.index('central_lat')
#     column2=dados_emission.iloc[:,pos2]
#     lat = np.array(column2.dropna(), dtype=pd.Series)
    
#     pos4=header.index('central_lon')
#     column4=dados_emission.iloc[:,pos4]
#     lon = np.array(column4.dropna(), dtype=pd.Series)
    
#     pos5=header.index('FRP(MW)')
#     column5=dados_emission.iloc[:,pos5]
#     FRP = np.array(column5.dropna(), dtype=pd.Series)
    
#     pos6=header.index('hhmm')
#     column6=dados_emission.iloc[:,pos6]
#     hhmm = np.array(column6.dropna(), dtype=pd.Series)
    
#     area = 27.75**2
    
#     # index = np.where((code>=2021) & (code<2022))
#     # index = np.where((lat==matrix[0,0]) & (lon==matrix[0,1]) & (FRP>=0))
    
#     lats = np.unique(lat)
#     lons = np.unique(lon)
    

    
#     matrix_N_FRP = np.zeros( (6, 48) ,dtype='float64')
#     # print(matrix_CH4)
#     # print(matrix_CH4.shape)
#     for k in range(0,len(lats)):
#         for n in range(0,len(lons)):
            
#             index = np.where((lat==lats[k]) & (lon==lons[n]) & (FRP>=0) & (code>=2021) & (code<2022))
            
#             FRP_aux = FRP[index]
#             hhmm_aux = hhmm[index]
#             tz = tf.timezone_at(lng=lons[n], lat=lats[k])
#             time_zone = np.ones_like(hhmm_aux)*int(tz_utils.format_tz_by_name(tz)[0])
#             local_time_aux = np.subtract(hhmm_aux,time_zone*(-1))
            
#             mask = (local_time_aux<0)
#             local_time_aux[mask]+=2400
#             # hhmm_aux = 
            
#             peaks_aux, _ = find_peaks(FRP_aux,height=75,distance=50, prominence=750)
#             results_width = peak_widths(FRP_aux, peaks_aux, rel_height=0.95)
#             # fire_duration_partial = (results_width[0]*10)/60
            
#             index_diff_start_partial = np.int_(np.floor(results_width[2][:]))
#             # peak_end_pos = np.int_(np.floor(results_width[3][:]))

#             start_hours_local = local_time_aux[index_diff_start_partial]
            
#             matrix_N_FRP[k,n]= start_hours_local
#     # print(matrix_CH4)
#     # print(matrix_CH4.shape)
#     # X, Y = np.meshgrid(lats, lons)
    
    
    
#     pyplot.clf()
#     fig, ax = pyplot.subplots()
#     # ax = pyplot.gca()
# # im = ax.imshow(np.arange(100).reshape((10,10)))
    


# # plt.colorbar(im, cax=cax)
#     # pyplot.title(filename)
#     # pyplot.ylabel('lat')
#     # pyplot.xlabel('lon')
    
#     extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
#     # pyplot.pcolormesh(Y, X, matrix_TPM)
#     # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
#     im = ax.imshow(matrix_N_FRP,cmap='viridis', extent = extend)
    

    
#     #set aspect ratio to 8
#     ratio = 1
#     x_left, x_right = ax.get_xlim()
#     y_low, y_high = ax.get_ylim()
#     ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
#     ax.set_xlabel('lon')
#     ax.set_ylabel('lat')
#     ax.set_title(filename,fontsize=8)
    
#     # create an axes on the right side of ax. The width of cax will be 5%
#     # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#     # divider = make_axes_locatable(ax)
#     # cax = divider.append_axes("right", size="5%", pad=0.05)
    
#     pyplot.colorbar(im,label="N fires 100 days")
#     # cb.formatter.set_powerlimits((0, 0))
#     # ax.set_aspect(0.01)
    
#     pyplot.show()

#     fig.savefig(filename+'.png',dpi=200)
# ##########################################################################################################

#     return

# def get_code_indexes(dados_emission):
#     header = list(dados_emission)
    
#     pos1=header.index('code')
#     column1=dados_emission.iloc[:,pos1]
#     code = np.array(column1.dropna(), dtype=pd.Series)
    
    
#     index = np.where((code>=2022) & (code<2023))
#     # print(code[index])    
#     # print(len(code[index]))
#     newcode = np.unique(code[index])
#     print(newcode)    
#     print(len(newcode))
#     indexes=[]
#     for i in range(0,len(newcode)):
#         indexes_aux=np.where(code==newcode[i])
#         indexes.insert(i,indexes_aux)
#     print('indexes finish')

#     return indexes,newcode



CO2_stats(dados_emission,'co2_emission_amazon_'+str(Ano)+'_definitive_box')
# CO_stats(dados_emission,'co_emission_amazon_'+str(Ano)+'_definitive_box')
# CH4_stats(dados_emission,'ch4_emission_amazon_'+str(Ano)+'_definitive_box')
# TPM_stats(dados_emission,'tpm_emission_amazon_'+str(Ano)+'_definitive_box')


# Indexes_2022,code_new = get_code_indexes(dados_emission)
# CO2_stats_v2(dados_emission,'CO2_emission_amazon_big_box_2022_v2',Indexes_2022,code_new)
# CO_stats_v2(dados_emission,'CO_emission_amazon_big_box_2022_v2',Indexes_2022,code_new)
# CH4_stats_v2(dados_emission,'CH4_emission_amazon_big_box_2022_v2',Indexes_2022,code_new)

# TPM_mapping(dados_emission,'TPM_emission_maping_amazon_big_box_2021')
# TPM_mapping_v2(dados_emission,'TPM_emission_maping_amazon_big_box_2022_flux')
# CO2_mapping(dados_emission,'CO2_emission_maping_amazon_big_box_2021_flux')
# CO_mapping(dados_emission,'CO_emission_maping_amazon_big_box_2021_flux_day')
# CH4_mapping(dados_emission,'CH4_emission_maping_amazon_big_box_2021_flux')
# N_mapping(dados_emission, 'N_fires_mapping_amazon_big_box_2021')
# Start_hours_mapping(dados_emission, 'start_hours_mapping_amazon_big_box_2022')

# CO_mapping(dados_emission,'CO_emission_maping_amazon_big_box_2022_flux_day')
# TPM_stats_v3(dados_emission,'TPM_emission_amazon_big_box_2021_v3')
# FRP_stats(dados_emission,'frp_cluster_cerrado_v7')
# Temp_stats(dados_emission,'temp_cluster_cerrado_v7')
# Area_stats(dados_emission,'area_cluster_cerrado_v7')
# AreaxFRP(dados_emission,'area_X_frp_cluster_cerrado_v7')
# TempxFRP(dados_emission,'temp_X_frp_cluster_cerrado_v7')


print('Done')

#=julian+TRUNCAR(hhmm/100,0)/24+mod(hhmm,100)/60/24