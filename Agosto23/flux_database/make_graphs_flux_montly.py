# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:07:50 2023

@author: thiag
"""



import pandas as pd
from matplotlib import pyplot
import glob
import numpy as np
import math as mt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from scipy.signal import chirp, find_peaks, peak_widths, find_peaks_cwt
from numba import jit
# from timezonefinder import TimezoneFinder
# from timezones import tz_utils

# tf = TimezoneFinder()


# datadir = '/home/thiagovg/Documentos/Projeto_Mestrado/Maio23/FRP_amazon/emission_rate_graphs_v2/output'
datadir = 'C:/Users/thiag/OneDrive/Documentos/Projeto_Mestrado/Agosto23/flux_database/output'

# data_FRP = sorted(glob.glob(datadir+'/*FRP.csv'))
# data_TEMP = sorted(glob.glob(datadir+'/*temp.csv'))
# data_DQF = sorted(glob.glob(datadir+'/*DQF.csv'))
# data_Area = sorted(glob.glob(datadir+'/*Area.csv'))
# data = sorted(glob.glob(datadir+'/*rate_test_cerrado_box_v7.csv'))
data = sorted(glob.glob(datadir+'/*montly.csv')) #south_box
# data = sorted(glob.glob(datadir+'/*big_box_v1_2021_2022.csv')) #big_box

# dados_FRP =  pd.read_csv(data_FRP[0])
# dados_TEMP =  pd.read_csv(data_TEMP[0])
# dados_DQF =  pd.read_csv(data_DQF[0])
# dados_Area = pd.read_csv(data_Area[0])
emission_data = pd.read_csv(data[0])
aux1 = 'year,month,central_lat,central_lon,mean_flux,TPM_flux,CO2_flux,CO_flux,CH4_flux'


header = 'year,month,central_lat,central_lon,mean_flux,TPM_flux,CO2_flux,CO_flux,CH4_flux'



def TPM_mapping_v1_2method(dados_emission, filename, year):
    header = list(dados_emission)
    
    # mean_flux,RE_flux,ME_flux,
    pos1 = header.index('code')
    column1 = dados_emission.iloc[:, pos1]
    code = np.array(column1.dropna(), dtype=pd.Series)

    pos3 = header.index('ME_flux')
    column3 = dados_emission.iloc[:, pos3]
    TPM = np.array(column3.dropna(), dtype=pd.Series)

    pos2 = header.index('central_lat')
    column2 = dados_emission.iloc[:, pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)

    pos4 = header.index('central_lon')
    column4 = dados_emission.iloc[:, pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)

    area = 27.75**2

    lats = np.unique(lat)

    lons = np.unique(lon)

    lats = lats[::-1]

    matrix_TPM = np.zeros((len(lats), len(lons)), dtype='float64')
    # print(matrix_TPM)
    print(matrix_TPM.shape)
    for k in range(0, len(lats)):
        Progress = 'Progress: {:.1f}lat/{:.1f}lats'.format(k, len(lats))
        print(Progress)
        for n in range(0, len(lons)):
            matrix_TPM[k, n] = np.sum(TPM[(lat == lats[k]) & (
                lon == lons[n]) & (TPM>0) ])
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    # X, Y = np.meshgrid(lats, lons)

    pyplot.clf()
    fig, ax = pyplot.subplots(figsize=(6, 3))

    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_TPM, cmap='inferno', extent=extend)

    # set aspect ratio to 8
    # ratio = 1
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(filename, fontsize=8)

    pyplot.colorbar(im, label="TPM emisson(kg/km^2) day")

    # ax.set_aspect(0.01)

    pyplot.show()

    fig.savefig(filename+'.png', dpi=200)
##########################################################################################################

    return

def TPM_mapping_v2_2method(dados_emission, filename, year):
    header = list(dados_emission)
    
    # mean_flux,RE_flux,ME_flux,
    # year,month,central_lat,central_lon,mean_flux,TPM_flux,CO2_flux,CO_flux,CH4_flux
    pos1 = header.index('year')
    column1 = dados_emission.iloc[:, pos1]
    year = np.array(column1.dropna(), dtype=pd.Series)

    # pos3 = header.index('month')
    # column3 = dados_emission.iloc[:, pos3]
    # month = np.array(column3.dropna(), dtype=pd.Series)

    pos2 = header.index('central_lat')
    column2 = dados_emission.iloc[:, pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)

    pos4 = header.index('central_lon')
    column4 = dados_emission.iloc[:, pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5 = header.index('TPM_flux')
    column5 = dados_emission.iloc[:, pos5]
    TPM_flux = np.array(column5.dropna(), dtype=pd.Series)


    lats = np.unique(lat)

    lons = np.unique(lon)
    
    # print(lats)

    lats = lats[::-1]

    matrix_TPM = np.zeros((len(lats), len(lons)), dtype='float64')
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    for k in range(0, len(lats)):
        Progress = 'Progress: {:.1f}lat/{:.1f}lats'.format(k, len(lats))
        print(Progress)
        for n in range(0, len(lons)):
            # print(lats[k])
            # print(lons[n])
            index = np.where((lat == lats[k]) & (lon == lons[n]))
            # print(TPM_flux[index])
            matrix_TPM[k, n] = TPM_flux[index]
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    # X, Y = np.meshgrid(lats, lons)

    pyplot.clf()
    fig, ax = pyplot.subplots(figsize=(6, 3))
    

    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_TPM, cmap='inferno', extent=extend)

    # set aspect ratio to 8
    # ratio = 1
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    Title = 'Ano: {}, Mês: {}'.format(np.unique(year)[0],8)
    ax.set_title(Title)
    
    

    cb = pyplot.colorbar(im, label='Fluxo mensal  TPM (kg/km^2.s)',format='%.0e')
    cb.ax.tick_params(labelsize=10)
    
    # cb = pyplot.colorbar(im, label=Label, format='%.0e')  # Formatar para notação científica sem casas decimais
    # cb.ax.tick_params(labelsize=10)  # Ajustar o tamanho dos números na barra de cores

    # ax.set_aspect(0.01)

    # pyplot.show()

    fig.savefig(filename+'.png', dpi=200)
##########################################################################################################

    return

def CO2_mapping_v2_2method(dados_emission, filename, year):
    header = list(dados_emission)
    
    # mean_flux,RE_flux,ME_flux,
    # year,month,central_lat,central_lon,mean_flux,TPM_flux,CO2_flux,CO_flux,CH4_flux
    pos1 = header.index('year')
    column1 = dados_emission.iloc[:, pos1]
    year = np.array(column1.dropna(), dtype=pd.Series)

    # pos3 = header.index('month')
    # column3 = dados_emission.iloc[:, pos3]
    # month = np.array(column3.dropna(), dtype=pd.Series)

    pos2 = header.index('central_lat')
    column2 = dados_emission.iloc[:, pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)

    pos4 = header.index('central_lon')
    column4 = dados_emission.iloc[:, pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5 = header.index('CO2_flux')
    column5 = dados_emission.iloc[:, pos5]
    CO2_flux = np.array(column5.dropna(), dtype=pd.Series)


    lats = np.unique(lat)

    lons = np.unique(lon)
    
    # print(lats)

    lats = lats[::-1]

    matrix_CO2 = np.zeros((len(lats), len(lons)), dtype='float64')
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    for k in range(0, len(lats)):
        Progress = 'Progress: {:.1f}lat/{:.1f}lats'.format(k, len(lats))
        print(Progress)
        for n in range(0, len(lons)):
            # print(lats[k])
            # print(lons[n])
            index = np.where((lat == lats[k]) & (lon == lons[n]))
            # print(CO2_flux[index])
            matrix_CO2[k, n] = CO2_flux[index]
    # print(matrix_TPM)
    # print(matrix_TPM.shape)
    # X, Y = np.meshgrid(lats, lons)

    pyplot.clf()
    fig, ax = pyplot.subplots(figsize=(6, 3))

    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_TPM)
    # pyplot.pcolormesh(lats, lons, matrix_TPM, vmin=np.min(matrix_TPM), vmax=np.max(matrix_TPM), shading='auto')
    im = ax.imshow(matrix_CO2, cmap='cividis', extent=extend)

    # set aspect ratio to 8
    # ratio = 1
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    Title = 'Ano: {}, Mês: {}'.format(np.unique(year)[0],8)
    ax.set_title(Title)
    
    

    cb = pyplot.colorbar(im, label='Fluxo mensal  CO2 (kg/km^2.s)',format='%.0e')
    cb.ax.tick_params(labelsize=10)

    # ax.set_aspect(0.01)

    # pyplot.show()

    fig.savefig(filename+'.png', dpi=200)
##########################################################################################################

    return

def CO_mapping_v2_2method(dados_emission, filename, year):
    header = list(dados_emission)
    
    # mean_flux,RE_flux,ME_flux,
    # year,month,central_lat,central_lon,mean_flux,CO_flux,CO2_flux,CO_flux,CH4_flux
    pos1 = header.index('year')
    column1 = dados_emission.iloc[:, pos1]
    year = np.array(column1.dropna(), dtype=pd.Series)

    # pos3 = header.index('month')
    # column3 = dados_emission.iloc[:, pos3]
    # month = np.array(column3.dropna(), dtype=pd.Series)

    pos2 = header.index('central_lat')
    column2 = dados_emission.iloc[:, pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)

    pos4 = header.index('central_lon')
    column4 = dados_emission.iloc[:, pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5 = header.index('CO_flux')
    column5 = dados_emission.iloc[:, pos5]
    CO_flux = np.array(column5.dropna(), dtype=pd.Series)


    lats = np.unique(lat)

    lons = np.unique(lon)
    
    # print(lats)

    lats = lats[::-1]

    matrix_CO = np.zeros((len(lats), len(lons)), dtype='float64')
    # print(matrix_CO)
    # print(matrix_CO.shape)
    for k in range(0, len(lats)):
        Progress = 'Progress: {:.1f}lat/{:.1f}lats'.format(k, len(lats))
        print(Progress)
        for n in range(0, len(lons)):
            # print(lats[k])
            # print(lons[n])
            index = np.where((lat == lats[k]) & (lon == lons[n]))
            # print(CO_flux[index])
            matrix_CO[k, n] = CO_flux[index]
    # print(matrix_CO)
    # print(matrix_CO.shape)
    # X, Y = np.meshgrid(lats, lons)

    pyplot.clf()
    fig, ax = pyplot.subplots(figsize=(6, 3))

    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_CO)
    # pyplot.pcolormesh(lats, lons, matrix_CO, vmin=np.min(matrix_CO), vmax=np.max(matrix_CO), shading='auto')
    im = ax.imshow(matrix_CO, cmap='viridis', extent=extend)

    # set aspect ratio to 8
    # ratio = 1
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    Title = 'Ano: {}, Mês: {}'.format(np.unique(year)[0],8)
    ax.set_title(Title)
    
    

    cb = pyplot.colorbar(im, label='Fluxo mensal  CO (kg/km^2.s)',format='%.0e')
    cb.ax.tick_params(labelsize=10)

    # ax.set_aspect(0.01)

    # pyplot.show()

    fig.savefig(filename+'.png', dpi=200)
##########################################################################################################

    return

def CH4_mapping_v2_2method(dados_emission, filename, year):
    header = list(dados_emission)
    
    # mean_flux,RE_flux,ME_flux,
    # year,month,central_lat,central_lon,mean_flux,CH4_flux,CO2_flux,CO_flux,CH4_flux
    pos1 = header.index('year')
    column1 = dados_emission.iloc[:, pos1]
    year = np.array(column1.dropna(), dtype=pd.Series)

    # pos3 = header.index('month')
    # column3 = dados_emission.iloc[:, pos3]
    # month = np.array(column3.dropna(), dtype=pd.Series)

    pos2 = header.index('central_lat')
    column2 = dados_emission.iloc[:, pos2]
    lat = np.array(column2.dropna(), dtype=pd.Series)

    pos4 = header.index('central_lon')
    column4 = dados_emission.iloc[:, pos4]
    lon = np.array(column4.dropna(), dtype=pd.Series)
    
    pos5 = header.index('CH4_flux')
    column5 = dados_emission.iloc[:, pos5]
    CH4_flux = np.array(column5.dropna(), dtype=pd.Series)


    lats = np.unique(lat)

    lons = np.unique(lon)
    
    # print(lats)

    lats = lats[::-1]

    matrix_CH4 = np.zeros((len(lats), len(lons)), dtype='float64')
    # print(matrix_CH4)
    # print(matrix_CH4.shape)
    for k in range(0, len(lats)):
        Progress = 'Progress: {:.1f}lat/{:.1f}lats'.format(k, len(lats))
        print(Progress)
        for n in range(0, len(lons)):
            # print(lats[k])
            # print(lons[n])
            index = np.where((lat == lats[k]) & (lon == lons[n]))
            # print(CH4_flux[index])
            matrix_CH4[k, n] = CH4_flux[index]
    # print(matrix_CH4)
    # print(matrix_CH4.shape)
    # X, Y = np.meshgrid(lats, lons)

    pyplot.clf()
    fig, ax = pyplot.subplots(figsize=(6, 3))

    extend = np.min(lons), np.max(lons),  np.min(lats), np.max(lats)
    # pyplot.pcolormesh(Y, X, matrix_CH4)
    # pyplot.pcolormesh(lats, lons, matrix_CH4, vmin=np.min(matrix_CH4), vmax=np.max(matrix_CH4), shading='auto')
    im = ax.imshow(matrix_CH4, cmap='plasma', extent=extend)

    # set aspect ratio to 8
    # ratio = 1
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    Title = 'Ano: {}, Mês: {}'.format(np.unique(year)[0],8)
    ax.set_title(Title)
    
    

    cb = pyplot.colorbar(im, label='Fluxo mensal  CH4 (kg/km^2.s)',format='%.0e')
    cb.ax.tick_params(labelsize=10)

    # ax.set_aspect(0.01)

    # pyplot.show()

    fig.savefig(filename+'.png', dpi=200)
##########################################################################################################

    return
# minlon,maxlon,minlat,maxlat=-57,-54,-9,-6 #Amazon small box
# minlon,maxlon,minlat,maxlat=-72,-48,-9,-6 #Amazon big box v2
minlon, maxlon, minlat, maxlat = -72, -48, -11, -3  # Amazon south box


# CO2_stats(dados_emission,'co2_emission_cerrado_box_v7')
# CO_stats(dados_emission,'co_emission_cerrado_box_v7')
# CH4_stats(dados_emission,'ch4_emission_cerrado_box_v7')
ano = 2020

#METODO 2
# TPM_mapping_v1_2method(emission_data,'definitive_box_TPM_emission_mapping_amazon_'+str(ano)+'_flux_v1_2_method_day',ano)
TPM_mapping_v2_2method(emission_data,'definitive_box_TPM_emission_mapping_amazon_'+str(ano)+'_flux_v2_2_method_day',ano)
CO2_mapping_v2_2method(emission_data,'definitive_box_CO2_emission_mapping_amazon_'+str(ano)+'_flux_v2_2_method_day',ano)
CO_mapping_v2_2method(emission_data,'definitive_box_CO_emission_mapping_amazon_'+str(ano)+'_flux_v2_2_method_day',ano)
CH4_mapping_v2_2method(emission_data,'definitive_box_CH4_emission_mapping_amazon_'+str(ano)+'_flux_v2_2_method_day',ano)
print('Done')

# =julian+TRUNCAR(hhmm/100,0)/24+mod(hhmm,100)/60/24
