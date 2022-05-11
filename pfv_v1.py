import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats
from scipy import signal
from numpy import linalg as LA
import sympy
import math
import matplotlib.pyplot as plt
import os
import pathlib

# Full size page
st.set_page_config(layout="wide")

# Fonction du nouveau filtre 
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, padlen=len(data)-1)
    return y

# Fonction calcul régression linéaire
def linear_regression(x, y):     
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    
    B0 = y_mean - (B1*x_mean)
    
    reg_line = 'y = {} + {}β'.format(round(B0,5), round(B1, 3))
    
    return (B0, B1, reg_line)

def corr_coef(x, y):
    N = len(x)
    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R

# Constante 
FC = 20
Fr = 1000
FreqAcquis = 1 / Fr
Inertie = 13.8
Resistance = 0.0001
Pignons = 14
Volant = 0.2571
Plateau = 52
CoefVitAngulairePedalier = Pignons/(Volant*Plateau)
df_final = pd.DataFrame(columns=("Nom","VBA - a","VBA - b","VBA - Pmax","Python - a","Python - b","Python - Pmax","delta - a","delta - b","delta - Pmax"))


st.title("Analyse d'un fichier de sprint - Profil Force-Vitesse")
st.markdown('*version 1.0 - création Corentin Casali*')
st.markdown("Application permettant le traitement d'un fichier .xls déjà traité avec une macro VBA pour la comparaison entre l'utilisation de cette même macro VBA sur la détection et la prédiction de valeur de Pmax par rapport à un script python.")

uploaded_file = st.file_uploader("Choisir un fichier")
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, sheet_name = "DATA_CALC",header=1, engine = 'xlrd')
    data2 = pd.read_excel(uploaded_file, sheet_name = "DEMI_CYCLES", engine = 'xlrd')
    st.markdown("## Lecture du fichier :")
    st.write(data)

    # Filter requirements.
    T = len(data)/1000       # Sample Period
    Fs = 1000.0      # sample rate, Hz
    cutoff = 20      # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * Fs      # Frequence echantillonnage / 2
    order = 4
    n = int(T * Fs)

    # Récupération des valeurs du fichier
    temps = data["Temps"]
    force = data["FAcquis(N)"]
    deplacement = data["Depl Acquis(m).1"]
    vitesse = data["Vit(m.s-1).1"]
    acc = data["Acc(m.s-2).1"]
    ForceInertie = acc * Inertie +Resistance
    ForceTotale = force + ForceInertie
    Puissance = ForceTotale * vitesse 
    VitesseAngulaire = vitesse * CoefVitAngulairePedalier

    dataCalc = pd.DataFrame(np.transpose([temps,deplacement,vitesse,acc,ForceTotale,Puissance,VitesseAngulaire]))
    dataCalc.rename(columns={
        0:"Temps",
        1:"Déplacement",
        2:"Vitesse",
        3:"Accélération",
        4:"Force",
        5:"Puissance",
        6:"VitesseAngulaire"}, inplace = True)

    # Application du filtre : 
    puissance_filt = butter_lowpass_filter(Puissance, cutoff, Fs, order)

    # Détection des points en moyenne
    x = puissance_filt
    peaks, _ = signal.find_peaks(x, distance = 100,prominence=100)
    vallees, _ = signal.find_peaks(-x, distance = 100,prominence=100, height=-x.mean()-200)

    indexVal = []
    for i in range (0,len(vallees)):
        indexVal.append(dataCalc.iloc[vallees[i]-50:vallees[i]+50]["Puissance"].idxmin())

    indexPic = []
    for i in range (0,len(peaks)):
        indexPic.append(dataCalc.iloc[peaks[i]-50:peaks[i]+50]["Puissance"].idxmax())

    df_indexVal = pd.DataFrame(indexVal, columns = ['index'])
    df_indexVal['origineV'] = 'vallee'
    df_indexVal.set_index('index',inplace=True)
    df_indexPic = pd.DataFrame(indexPic, columns = ['index'])
    df_indexPic['origineP'] = 'pic'
    df_indexPic.set_index('index',inplace=True)

    resultat_index = pd.concat([df_indexPic,df_indexVal], axis=1)
    resultat_index['origine_final'] = resultat_index['origineP'].fillna(resultat_index['origineV'])
    resultat_index.reset_index(inplace=True)
    resultat_index.drop(["origineP","origineV"], axis=1, inplace=True)


    # Reset index dataCalc
    dataCalc_index_reset = dataCalc.reset_index()
    resultat_index = pd.merge(resultat_index,dataCalc_index_reset, on='index', how='inner')

    for i in range(0,len(resultat_index)-1):
        # Détection des doubles pics
        if ((resultat_index.iloc[i]['origine_final'] == 'pic') and (resultat_index.iloc[i+1]['origine_final'] == 'pic')):
            resultat_index.at[i,'doublepic'] = 1
        else:
            resultat_index.at[i,'doublepic'] = 0
        # Détection des doubles vallées
        if ((resultat_index.iloc[i]['origine_final'] == 'vallee') and (resultat_index.iloc[i+1]['origine_final'] == 'vallee')):
            resultat_index.at[i,'doubleval'] = 1
        else:
            resultat_index.at[i,'doubleval'] = 0

    resultat_index.fillna(0, inplace=True)

    # Erreur dans le boucle, le .at(i+1) ne fonctionne pas
    
    for i in range(0,len(resultat_index)-1):
        # choix du double pic à supprimer
        if resultat_index.iloc[i]['doublepic'] == 1:
            if resultat_index.iloc[i]['Puissance'] < resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doublepic_2'] = 1
                resultat_index.at[i+1,'doublepic_2'] = 0
            elif resultat_index.iloc[i]['Puissance'] > resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doublepic_2'] = 0
                resultat_index.at[i+1,'doublepic_2'] = 1
        else:
            resultat_index.at[i+1,'doublepic_2'] = 0 

        # # choix du double vallee à supprimer
        if resultat_index.iloc[i]['doubleval'] == 1:
            if resultat_index.iloc[i]['Puissance'] < resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doubleval_2'] = 0
                resultat_index.at[i+1,'doubleval_2'] = 1
            elif resultat_index.iloc[i]['Puissance'] > resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doubleval_2'] = 1
                resultat_index.at[i+1,'doubleval_2'] = 0
        else:
            resultat_index.at[i+1,'doubleval_2'] = 0 

    resultat_index.fillna(0, inplace=True)

    resultat_index.drop(resultat_index[resultat_index["doublepic_2"]==1].index, inplace=True,errors='ignore')
    resultat_index.drop(resultat_index[resultat_index["doubleval_2"]==1].index, inplace=True,errors='ignore')
    # Attribution des données peak et val du filtre dans des dataframes
    dataPeak = dataCalc.iloc[indexPic].copy()
    dataVal = dataCalc.iloc[indexVal].copy()

    # Récupération des vrai valeurs de puissance et d'index avec
    # Premier pic ou premiere vallées > 20 Watts

    # Suppression des valeurs ne dépassant pas 20 Watts et inférieur à 2 secondes
    dataVal.drop(index=dataVal.loc[(dataVal["Puissance"]<20)&(dataVal["Temps"]<2)].index.values,inplace=True)

    # Ajout de la première ligne
    # dataVal = pd.concat([pd.DataFrame(dataCalc.iloc[0]).transpose(),dataVal], ignore_index = False, axis = 0)
    # newIndexVal = dataVal.index.values
    newIndexVal = np.array(resultat_index.loc[resultat_index["origine_final"]=="vallee"]['index'])
    newIndexVal = np.insert(newIndexVal,0,0)
    newIndexPic = np.array(resultat_index.loc[resultat_index["origine_final"]=="pic"]['index'])

    # ## Calcul des DEMI_CYCLES
    # Calcul du temps, déplacement, vitesse, acceleration, force, puissance et vitesse angulaire
    DC_temps = []
    for i in range (0,len(newIndexVal)-1):
        DC_temps.append((dataCalc["Temps"].iloc[newIndexVal[i+1]]+dataCalc["Temps"].iloc[newIndexVal[i]])/2)

    DC_puissance = []
    for i in range (0,len(newIndexVal)-1):
        DC_puissance.append(np.mean(dataCalc["Puissance"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    DC_vitesse = []
    for i in range (0,len(newIndexVal)-1):
        DC_vitesse.append(np.mean(dataCalc["Vitesse"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    DC_force = []
    for i in range (0,len(newIndexVal)-1):
        DC_force.append(np.mean(dataCalc["Force"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    DC_vitAng = []
    for i in range (0,len(newIndexVal)-1):
        DC_vitAng.append(np.mean(dataCalc["VitesseAngulaire"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    ## DATAFRAME DEMI_CYCLE
    DEMI_CYCLE = pd.DataFrame(np.transpose([DC_temps,DC_vitesse,DC_force, DC_puissance,DC_vitAng]))
    DEMI_CYCLE.rename(columns={
        0:"Temps",
        1:"Vitesse",
        2:"Force",
        3:"Puissance",
        4:"VitAng"}, inplace = True)

    # Récupération des indices pic-vallées de rodolphe
    dataPicVallee = pd.read_excel(uploaded_file,sheet_name = "PIC_VALLEES")
    indexValRodolphe = np.array(dataPicVallee["Indice Val"])
    indexPicRodolphe = np.array(dataPicVallee["Indice Pic"][:-1])


    # --------2 EME MÉTHODE--------
    # Calcul de Vmax (on ne compte pas le dernier cycle)
    Vmax = DEMI_CYCLE["Vitesse"][:-1].max()
    # Fixation du critère de recherche à 0.99
    CritereRecherche = 0.95
    nbValeurMax = DEMI_CYCLE[DEMI_CYCLE["Vitesse"]>=CritereRecherche*Vmax].index[0]-1
    # print("Critère de recherche de :", round(CritereRecherche*Vmax,3))
    # print("Nombre de valeurs choisies :",nbValeurMax)

    # Calcul de régression
    x = DEMI_CYCLE["VitAng"][2:nbValeurMax]
    y = DEMI_CYCLE["Force"][2:nbValeurMax]
    B0, B1, reg_line = linear_regression(x, y)

    # Calcul des différents paramètres
    F0 = round(B0,3)
    V0 = round(-B0/B1,3)
    Pmax = round(V0*F0/4,3)

    # Récupération des données de Rodolphe
    aRodolphe = data2.iloc[1,9]
    bRodolphe = data2.iloc[1,10]
    pmaxRodolphe = data2.iloc[1,15]

    
    df_final.loc[0]=[uploaded_file.name[:-4],aRodolphe,bRodolphe,pmaxRodolphe,round(B1,3),round(B0,5),Pmax,(B1-aRodolphe),(B0-bRodolphe),(Pmax-pmaxRodolphe)]

    # Graphique Puissance
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=Puissance,
        line_color = "black",
        name = "Puissance"
    ))

    fig.add_trace(go.Scatter(
        x = newIndexPic,
        y = Puissance[newIndexPic],
        mode = "markers",
        marker = dict(
            size = 10,
            symbol = 'x-thin',
            line=dict(
                color = "Blue",
                width = 2
        )), name = "Pic"
    ))

    fig.add_trace(go.Scatter(
        x = newIndexVal,
        y = Puissance[newIndexVal],
        mode = "markers",
        marker = dict(
            size = 10,
            symbol = 'x-thin',
            line=dict(
                color = "Red",
                width = 2
        )), name = "Vallée"
    ))

    fig.update_layout(
        title="Graphique Puissance - Python Corentin",
        xaxis_title="Temps",
        yaxis_title="Puissance (W)",
        font=dict(
            family="Raleway, monospace",
            size=16,
            color="Black"
        )
    )
    st.markdown("# Graphique Puissance :")
    st.markdown("Les graphiques représentent la puissance au cours du temps avec la détection de pic/vallée pour les 2 méthodes.\
        Attention des différences dans les détections peuvent arriver entre les deux méthodes, l'application a pour but de visualiser ses différences.")
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique Puissance - Rodolphe
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=Puissance,
        line_color = "Grey",
        name = "Puissance"
    ))

    fig.add_trace(go.Scatter(
        x = indexPicRodolphe,
        y = Puissance[indexPicRodolphe],
        mode = "markers",
        marker = dict(
            size = 10,
            symbol = 'x-thin',
            line=dict(
                color = "Blue",
                width = 2
        )), name = "Pic"
    ))

    fig.add_trace(go.Scatter(
        x = indexValRodolphe,
        y = Puissance[indexValRodolphe],
        mode = "markers",
        marker = dict(
            size = 10,
            symbol = 'x-thin',
            line=dict(
                color = "Red",
                width = 2
        )), name = "Vallée"
    ))

    fig.update_layout(
        title="Graphique Puissance - VBA Rodolphe",
        xaxis_title="Temps",
        yaxis_title="Puissance (W)",
        font=dict(
            family="Raleway, monospace",
            size=16,
            color="Black"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Visualisation de la relation force/vitesse")
    # Relation force-Vitesse
    fig = px.scatter(
        x = DEMI_CYCLE["VitAng"][2:nbValeurMax+i],
        y = DEMI_CYCLE["Force"][2:nbValeurMax+i],
        trendline = 'ols',
        width = 1200)

    fig.update_yaxes(rangemode="tozero")

    fig.update_traces(
        marker=dict(
            size=10,
            color = "Grey",
            line=dict(width = 2,
            color = "Black")
        )
    )

    fig.update_layout(
        title="Relation force-vitesse",
        xaxis_title="Vitesse (rad.s-1)",
        yaxis_title="Force (N)",
        font=dict(
            family="Raleway, monospace",
            size=16,
            color="Black"
        )
    )

    st.plotly_chart(fig, use_container_width=False)
    st.markdown("## Comparaison des données entre les 2 méthodes")
    st.write("Tableau présentant un comparatif entre les différentes données récupérées à travers les 2 méthodes. Le delta exprime ici la différence entre les données récupérée par la méthode VBA par rapport à celle de Python")
    st.write(df_final)

