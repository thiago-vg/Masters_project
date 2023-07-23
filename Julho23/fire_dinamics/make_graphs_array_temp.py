# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:37:31 2023

@author: thiag
"""

import pandas as pd
from matplotlib import pyplot
import glob
import numpy as np
import math as mt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from scipy.stats import norm
# import uncertainties.unumpy as unp
# import uncertainties as unc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from scipy.signal import chirp, find_peaks, peak_widths,find_peaks_cwt
# from timezonefinder import TimezoneFinder
# from timezones import tz_utils


datadir = 'C:/Users/thiag/OneDrive/Documentos/Projeto_Mestrado/Julho23/fire_dinamics/output' #Windows Version
# datadir = '/home/thiagovg//Documentos/Projeto_Mestrado/Maio23/FRP_amazon/fire_dinamics_graphs/output' #Ubuntu Version

# data_FRP = sorted(glob.glob(datadir+'/*FRP.csv'))
# data_TEMP = sorted(glob.glob(datadir+'/*temp.csv'))
# data_DQF = sorted(glob.glob(datadir+'/*DQF.csv'))
# data_Area = sorted(glob.glob(datadir+'/*Area.csv'))
# data = sorted(glob.glob(datadir+'/*rate_test_cerrado_box_v7.csv'))
data_temp = sorted(glob.glob(datadir+'/*array_T_data_test_v2.csv'))
data_frp = sorted(glob.glob(datadir+'/*array_P_data_test_v1.csv'))
# data = sorted(glob.glob(datadir+'/*fire_dinamics_amazon_small_box_full_v1.csv'))


# dados_FRP =  pd.read_csv(data_FRP[0])
# dados_TEMP =  pd.read_csv(data_TEMP[0])
# dados_DQF =  pd.read_csv(data_DQF[0])
# dados_Area = pd.read_csv(data_Area[0])
temperature_data = pd.read_csv(data_temp[0])
frp_data = pd.read_csv(data_frp[0])
# aux1 ='sat,year,julian,hhmm,code,central_lat,central_lon,FRP(MW),N_FRP\n'
aux1 ='array_temp,array_frp\n'


# tf = TimezoneFinder()

header=aux1

def plot_Temp(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=pd.Series)
    
    percentil_80 = np.percentile(Temp, 80)
    
    
    pyplot.clf()
    pyplot.title(filename)
    binwidth = 5
    pyplot.hist(Temp,bins=np.arange(min(Temp), 2000 + binwidth, binwidth))
    
    pyplot.axvline(percentil_80, color='r', linestyle='--', label='Percentil 80')
    pyplot.legend(loc='best')

    pyplot.savefig(filename+'.png',dpi=200)
    
    return 

def plot_Temp_log(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=pd.Series)
    

    percentil_80 = np.percentile(Temp, 80)
    
    pyplot.clf()
    pyplot.title(filename)
    binwidth = 5
    pyplot.hist(Temp,bins=np.arange(min(Temp), 2000 + binwidth, binwidth))
    pyplot.axvline(percentil_80, color='r', linestyle='--', label='Percentil 80')
    pyplot.xscale('log')
    pyplot.legend(loc='best')
    pyplot.savefig(filename+'.png',dpi=200)
    
    return 

def plot_Temp_log_curve_fit(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=float)
    

    # percentil_80 = np.percentile(Temp, 80)
    
    pyplot.clf()
    # pyplot.title(filename)
    # binwidth = 5
    # pyplot.hist(Temp,bins=np.arange(min(Temp), 2000 + binwidth, binwidth))
    # pyplot.axvline(percentil_80, color='r', linestyle='--', label='Percentil 80')
    # pyplot.xscale('log')
    # pyplot.legend(loc='best')
    
    
    parametros = lognorm.fit(Temp)

    # Extrair o parâmetro sigma
    sigma = parametros[2]
    Label = 'Sigma = {:.2f}'.format(sigma)
    #print(Label)
    # Plotar o histograma dos dados
    binwidth = 5
    pyplot.hist(Temp,bins=np.arange(min(Temp), 2000 + binwidth, binwidth),density=True, alpha=0.6, label=Label)
    # pyplot.hist(Temp, bins=30, )
    
    # Gerar valores para a curva ajustada
    x = np.linspace(min(Temp), max(Temp), 1000)
    curva_ajustada = lognorm.pdf(x, *parametros)
    
    # Plotar a curva ajustada
    pyplot.plot(x, curva_ajustada, 'r-', label='Curva Log-Normal')
    
    # Configurar o título e rótulos dos eixos
    pyplot.xscale('log')
    pyplot.title('Ajuste de Curva Log-Normal')
    pyplot.xlabel('Valores')
    pyplot.ylabel('Densidade')
    pyplot.legend()
    
    # Exibir o gráfico
    pyplot.savefig(filename+'.png',dpi=200)
    
    # Exibir o valor de sigma
    print('Valor de sigma:', sigma)
        
    return 

# Função Gaussiana para o ajuste
def gaussiana(x, amplitude, media, desvio_padrao):
    return amplitude * norm.pdf(x, loc=media, scale=desvio_padrao)

def plot_Temp_curve_fit_v2(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=float)
    
    pyplot.clf()
    # Gerar um conjunto de dados aleatórios (substitua o tamanho e os limites conforme necessário)
    dados = Temp
    
    # Criar um histograma dos dados
    num_bins = 200
    hist, bin_edges = np.histogram(dados, bins=num_bins,density=True, range=(min(Temp), 1300))
    
    # Calcular o centro de cada bin para usar como os valores de x
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ajustar a curva Gaussiana ao histograma
    parametros_iniciais = [np.max(hist), np.mean(dados), np.std(dados)]
    parametros_otimizados, _ = curve_fit(gaussiana, bin_centers, hist, p0=parametros_iniciais)
    
    # Parâmetros otimizados
    amplitude_otimizada, media_otimizada, sigma = parametros_otimizados
    
    # Gerar pontos para a curva ajustada
    x_curva = np.linspace(min(Temp), 1300, 10000)
    y_curva = gaussiana(x_curva, amplitude_otimizada, media_otimizada, sigma)
    
    # Plotar o histograma dos dados e a curva ajustada
    Label = 'Sigma = {:.2f}\n N_total = {:.1f}'.format(sigma,len(Temp))
    pyplot.hist(dados, bins=num_bins, range=(min(Temp), 1300),density=True, alpha=0.6, label=Label)
    pyplot.plot(x_curva, y_curva, 'r', label='Curva Ajustada')
    pyplot.xlabel('Temperatura (Kelvin)')
    pyplot.ylabel('Frequência')
    pyplot.title('Ajuste de Curva Gaussiana ao Histograma')
    pyplot.legend()
    pyplot.grid()
    # pyplot.show()
        
    # Exibir o gráfico
    pyplot.savefig(filename+'.png',dpi=200)
    
    # Exibir o valor de sigma
    # print('Valor de sigma:', sigma)
        
    return

def plot_Temp_log_curve_fit_v2(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=float)
    pyplot.clf()
    Temp = np.log(Temp)
    # Gerar um conjunto de dados aleatórios (substitua o tamanho e os limites conforme necessário)
    dados = Temp
    
    # Criar um histograma dos dados
    num_bins = 200
    hist, bin_edges = np.histogram(dados, bins=num_bins,density=True, range=(min(Temp), np.log(1300)))
    
    # Calcular o centro de cada bin para usar como os valores de x
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ajustar a curva Gaussiana ao histograma
    parametros_iniciais = [np.max(hist), np.mean(dados), np.std(dados)]
    parametros_otimizados, _ = curve_fit(gaussiana, bin_centers, hist, p0=parametros_iniciais)
    
    # Parâmetros otimizados
    amplitude_otimizada, media_otimizada, sigma = parametros_otimizados
    
    # Gerar pontos para a curva ajustada
    x_curva = np.linspace(min(Temp), np.log(1300), 10000)
    y_curva = gaussiana(x_curva, amplitude_otimizada, media_otimizada, sigma)
    
    # Plotar o histograma dos dados e a curva ajustada
    Label = 'Sigma = {:.2f}\nN_total = {:.0f}\nModa = {:.2f}'.format(sigma,len(Temp),media_otimizada)
    
    pyplot.hist(dados, bins=num_bins, range=(min(Temp), np.log(1300)),density=True, alpha=0.6, label=Label)
    pyplot.plot(x_curva, y_curva, 'r', label='Curva Ajustada')
    pyplot.axvline(media_otimizada, color='g', linestyle='--', label='Moda da gaussiana')
    pyplot.xlabel('log(T/T0)')
    pyplot.ylabel('Frequência Relativa')
    pyplot.title('Ajuste de Curva Gaussiana ao Histograma')
    pyplot.legend()
    pyplot.grid()
    # pyplot.show()
        
    # Exibir o gráfico
    pyplot.savefig(filename+'.png',dpi=200)
    
    # Exibir o valor de sigma
    # print('Valor de sigma:', sigma)
        
    return

# Função Gaussiana bimodal para o ajuste
def gaussiana_bimodal(x, amplitude1, media1, desvio_padrao1, amplitude2, media2, desvio_padrao2):
    return (amplitude1 * np.exp(-(x - media1) ** 2 / (2 * desvio_padrao1 ** 2)) +
            amplitude2 * np.exp(-(x - media2) ** 2 / (2 * desvio_padrao2 ** 2)))

def plot_Temp_log_bimodal_fit_v2(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=float)
    pyplot.clf()
    Temp = np.log(Temp)
    
    # Gerar um conjunto de dados aleatórios (substitua o tamanho e os limites conforme necessário)
    dados1 = Temp[Temp<6.7]
    dados2 = Temp[Temp>6.7]
    dados = Temp
    
    # Criar um histograma dos dados em frequência relativa
    num_bins = 200
    hist, bin_edges = np.histogram(dados, bins=num_bins,density=True, range=(min(Temp), np.log(2000)))
    
    # Calcular o centro de cada bin para usar como os valores de x
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ajustar a curva Gaussiana bimodal ao histograma
    parametros_iniciais = [np.max(hist), np.mean(dados1), np.std(dados1),
                           np.max(hist), np.mean(dados2), np.std(dados2)]
    parametros_otimizados, _ = curve_fit(gaussiana_bimodal, bin_centers, hist, p0=parametros_iniciais)
    
    # Parâmetros otimizados
    amplitude1_otimizada, media1_otimizada, desvio_padrao1_otimizado, amplitude2_otimizada, media2_otimizada, desvio_padrao2_otimizado = parametros_otimizados
    
    # Imprimir os valores de sigma das curvas Gaussianas
    print("Valor de sigma da primeira curva gaussiana ajustada:", desvio_padrao1_otimizado)
    print("Valor de sigma da segunda curva gaussiana ajustada:", desvio_padrao2_otimizado)
    
    # Gerar pontos para a curva bimodal ajustada
    x_curva = np.linspace(min(Temp), np.log(2000), 10000)
    y_curva = gaussiana_bimodal(x_curva, amplitude1_otimizada, media1_otimizada, desvio_padrao1_otimizado,
                                amplitude2_otimizada, media2_otimizada, desvio_padrao2_otimizado)
    
    Label = 'Sigma = {:.2f}\nN_total = {:.1f}\nModa = {:.1f}'.format(desvio_padrao1_otimizado,len(Temp),media1_otimizada)
    # Plotar o histograma dos dados em frequência relativa e a curva bimodal ajustada
    pyplot.hist(dados, bins=num_bins, range=(min(Temp), np.log(2000)), density=True, alpha=0.6)
    
    # Plotar a linha tracejada no valor centralizado entre as duas curvas
    # pyplot.axvline(media1_otimizada, color='g', linestyle='--', label='Média da primeira gaussiana')
    # pyplot.axvline(media2_otimizada, color='g', linestyle='--', label='Média da segunda gaussiana')
    pyplot.plot(x_curva, y_curva, 'r', label='Curva Bimodal Ajustada')
    
    
    # Plotar as curvas individuais
    # x_curva_individual = np.linspace(400, 1400, 100)
    y_curva_individual_1 = gaussiana(x_curva, amplitude1_otimizada, media1_otimizada, desvio_padrao1_otimizado)
    y_curva_individual_2 = gaussiana(x_curva, amplitude2_otimizada, media2_otimizada, desvio_padrao2_otimizado)
    
    pyplot.plot(x_curva, y_curva_individual_1, 'b--', label='Curva Individual 1')
    pyplot.plot(x_curva, y_curva_individual_2, 'g--', label='Curva Individual 2')
    
    
    
    pyplot.xlabel('Temperatura (Kelvin)')
    pyplot.ylabel('Frequência Relativa')
    pyplot.title('Ajuste de Curva Gaussiana Bimodal ao Histograma')
    pyplot.legend()
    pyplot.grid()
    # pyplot.show()
        
    # Exibir o gráfico
    pyplot.savefig(filename+'.png',dpi=200)
    
    # Exibir o valor de sigma
    # print('Valor de sigma:', sigma)
        
    return

   # Função Gaussiana 1 individual
def gaussiana_1(x, amplitude1, media1, desvio_padrao1):
    return amplitude1 * np.exp(-(x - media1) ** 2 / (2 * desvio_padrao1 ** 2))

# Função Gaussiana 2 individual
def gaussiana_2(x, amplitude2, media2, desvio_padrao2):
    return amplitude2 * np.exp(-(x - media2) ** 2 / (2 * desvio_padrao2 ** 2))

def plot_Temp_log_bimodal_fit_v3(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=float)
    pyplot.clf()
    Temp = np.log(Temp)
    
    # Gerar um conjunto de dados aleatórios (substitua o tamanho e os limites conforme necessário)
    dados1 = Temp[Temp<=6.7]
    dados2 = Temp[Temp>6.7]
    # print(min(dados2))
    # dados1 = Temp[Temp<800]
    # dados2 = Temp[Temp>800]
    dados = np.concatenate([dados1, dados2])
    
    # Criar um histograma dos dados em frequência relativa
    num_bins = 200
    hist, bin_edges = np.histogram(dados, bins=num_bins, range=(min(dados), max(dados)), density=True)  # density=True calcula a frequência relativa
    
    # Calcular o centro de cada bin para usar como os valores de x
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ajustar a curva Gaussiana bimodal ao histograma
    parametros_iniciais = [np.max(hist), 6.28, 0.1,
                           np.max(hist), 7.0, 0.001]
    parametros_otimizados, _ = curve_fit(gaussiana_bimodal, bin_centers, hist, p0=parametros_iniciais)
    
    # Parâmetros otimizados
    amplitude1_otimizada, media1_otimizada, desvio_padrao1_otimizado, amplitude2_otimizada, media2_otimizada, desvio_padrao2_otimizado = parametros_otimizados
    
    # # Imprimir os valores de sigma das curvas Gaussianas
    # print("Valor de sigma da primeira curva gaussiana ajustada:", desvio_padrao1_otimizado)
    # print("Valor de sigma da segunda curva gaussiana ajustada:", desvio_padrao2_otimizado)
    
    # # Calcular a moda de cada Gaussiana ajustada
    # moda_gaussiana1 = media1_otimizada
    # moda_gaussiana2 = media2_otimizada
    # print("Moda da primeira Gaussiana ajustada:", moda_gaussiana1)
    # print("Moda da segunda Gaussiana ajustada:", moda_gaussiana2)
    
    
    # Gerar pontos para a curva bimodal ajustada
    x_curva = np.linspace(min(dados), max(dados), 1000)
    y_curva = gaussiana_bimodal(x_curva, amplitude1_otimizada, media1_otimizada, desvio_padrao1_otimizado,
                                amplitude2_otimizada, media2_otimizada, desvio_padrao2_otimizado)
    
    # Plotar o histograma dos dados em frequência relativa e a curva bimodal ajustada
    Label = 'N_total = {:.0f}\nSigma_1={:.4f},Sigma_2={:.4f}\nModa_1={:.2f},Moda_2={:.2f}'.format(len(Temp),desvio_padrao1_otimizado,desvio_padrao2_otimizado,media1_otimizada,media2_otimizada)
    pyplot.hist(dados, bins=num_bins, range=(min(dados), max(dados)), density=True, alpha=0.6, label=Label)
    pyplot.plot(x_curva, y_curva, 'r', label='Curva Bimodal Ajustada')
    
    # Plotar as curvas individuais
    x_curva_individual = np.linspace(min(dados), max(dados), 1000)
    y_curva_individual_1 = gaussiana_1(x_curva_individual, amplitude1_otimizada, media1_otimizada, desvio_padrao1_otimizado)
    y_curva_individual_2 = gaussiana_2(x_curva_individual, amplitude2_otimizada, media2_otimizada, desvio_padrao2_otimizado)
    
    pyplot.plot(x_curva_individual, y_curva_individual_1, 'b--', label='Curva Individual 1')
    pyplot.plot(x_curva_individual, y_curva_individual_2, 'g--', label='Curva Individual 2')
    
    pyplot.xlabel('ln(T/T0)')
    pyplot.ylabel('Frequência Relativa')
    pyplot.title('Ajuste de Curva Gaussiana Bimodal ao Histograma')
    pyplot.legend()
    pyplot.grid()
    # pyplot.show()
    # pyplot.show()
        
    # Exibir o gráfico
    pyplot.savefig(filename+'.png',dpi=200)
    
    # Exibir o valor de sigma
    # print('Valor de sigma:', sigma)
        
    return

#x, y inputs can be lists or 1D numpy arrays

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def plot_Temp_log_bimodal_fit_v4(Temp_data,filename,):
    header = list(Temp_data)
      
    pos1=header.index('array_temp')
    column1=Temp_data.iloc[:,pos1]
    Temp = np.array(column1.dropna(), dtype=float)
    pyplot.clf()
    # Temp = np.log(Temp)
    
    #data generation
    dados1 = Temp[Temp<800]
    dados2 = Temp[Temp>800]
    data=np.concatenate([dados1, dados2])
    y,x,_=pyplot.hist(data, 100, alpha=.3,density=True ,label='data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    
    
    
    expected = (np.mean(dados1), 100, np.std(dados1), np.mean(dados1), 40, np.std(dados2))
    # parametros_iniciais = [np.max(hist), np.mean(dados1), np.std(dados1),
    #                        np.max(hist), np.mean(dados1), np.std(dados2)]
    params, cov = curve_fit(bimodal, x, y, expected)
    sigma=np.sqrt(np.diag(cov))
    x_fit = np.linspace(x.min(), x.max(), 1000)
    #plot combined...
    pyplot.plot(x_fit, bimodal(x_fit, *params), color='red', lw=3, label='model')
    #...and individual Gauss curves
    pyplot.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
    pyplot.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='distribution 2')
    #and the original data points if no histogram has been created before
    #pyplot.scatter(x, y, marker="X", color="black", label="original data")
    pyplot.legend()
    pyplot.xlabel('T/T0')
    pyplot.ylabel('Frequência Relativa')
    pyplot.title('Ajuste de Curva Gaussiana Bimodal ao Histograma')
    print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=bimodal.__code__.co_varnames[1:]))
    # pyplot.show() 
    pyplot.savefig(filename+'.png',dpi=200)
    
    # Exibir o valor de sigma
    # print('Valor de sigma:', sigma)
        
    return

def plot_FRP(FRP_data,filename,):
    header = list(FRP_data)
      
    pos1=header.index('array_frp')
    column1=FRP_data.iloc[:,pos1]
    FRP = np.array(column1.dropna(), dtype=pd.Series)
    

    
    
    pyplot.clf()
    pyplot.title(filename)
    binwidth = 5
    pyplot.hist(FRP,bins=np.arange(min(FRP), 2000 + binwidth, binwidth))
    pyplot.savefig(filename+'.png',dpi=200)
    
    return 

def plot_FRP_log(FRP_data,filename,):
    header = list(FRP_data)
      
    pos1=header.index('array_frp')
    column1=FRP_data.iloc[:,pos1]
    FRP = np.array(column1.dropna(), dtype=pd.Series)
    

    
    
    
    pyplot.clf()
    pyplot.title(filename)
    binwidth = 5
    pyplot.hist(FRP,bins=np.arange(min(FRP), max(FRP) + binwidth, binwidth))
    # pyplot.hist(FRP)
    pyplot.xscale('log')
    pyplot.savefig(filename+'.png',dpi=200)
    
    return 


def plot_TEMPXFRP(Temp_data,FRP_data,filename):
    header = list(FRP_data)
    header_temp = list(Temp_data)
      
    pos1=header.index('array_frp')
    column1=FRP_data.iloc[:,pos1]
    FRP = np.array(column1.dropna(), dtype=float)
    
    pos2=header_temp.index('array_temp')
    column2=Temp_data.iloc[:,pos2]
    Temp = np.array(column2.dropna(), dtype=float)
    
    # Definir o tamanho comum
    tamanho_comum = len(Temp)
    print(FRP.dtype)  # Saída: int64
    print(Temp.dtype)  # Saída: <U1 (string)

    # Interpolar os dados para o tamanho comum
    FRP_interpolados = np.interp(np.linspace(0, 1, tamanho_comum), np.linspace(0, 1, len(FRP)), FRP)
    Temp_interpolados = np.interp(np.linspace(0, 1, tamanho_comum), np.linspace(0, 1, len(Temp)), Temp)

    
    
    pyplot.clf()
    pyplot.title(filename)
    
    pyplot.plot(Temp_interpolados,FRP_interpolados,'.',markersize=3)
    pyplot.xlabel('Temp(K)')
    pyplot.ylabel('FRP')
    # binwidth = 5
    # pyplot.hist(FRP,bins=np.arange(min(FRP), max(FRP) + binwidth, binwidth))
    # # pyplot.hist(FRP)
    # pyplot.xscale('log')
    pyplot.savefig(filename+'.png',dpi=200)
    
    return 



# plot_Temp(temperature_data,'Plot_array_temp_test_V2_full')
# plot_Temp_log(temperature_data,'Plot_array_temp_log_test_V2_full')
# plot_Temp_log_curve_fit(temperature_data,'Teste_curve_fit')
# plot_Temp_curve_fit_v2(temperature_data,'Teste_curve_fit_v2')
# plot_Temp_log_curve_fit_v2(temperature_data,'Teste_log_curve_fit_v2')
# plot_Temp_log_bimodal_fit_v2(temperature_data,'Teste_bimodal_fit_v2')
plot_Temp_log_bimodal_fit_v3(temperature_data,'Teste_bimodal_fit_v3_log')
# plot_Temp_log_bimodal_fit_v4(temperature_data,'Teste_bimodal_fit_v4')
# plot_FRP(frp_data,'Plot_array_FRP_test_full')
# plot_FRP_log(frp_data,'Plot_array_FRP_log_test_full')
# plot_TEMPXFRP(temperature_data,frp_data, 'TempxFRP_test')

print('Done')