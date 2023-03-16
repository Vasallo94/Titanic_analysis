# --------------------LIBRERÍAS----------------------------#
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly_express as px
import base64

# --------------------CONFIGURACIÓN DE LA PÁGINA----------------------------#
# layout="centered" or "wide"
st.set_page_config(page_title="Titanic: preprocesamiento",
                   layout="wide", page_icon="min⛏️")
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title('Preprocesamiento de los datos')

# ----------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------#
# Leemos el csv con los datos del Titanic y creamos el dataframe
data = pd.read_csv(
    'data\titanic.csv')
titanic = pd.DataFrame(data=data)

st.dataframe(titanic)
nulos = sns.heatmap(titanic.isnull(), cmap='viridis')

st.markdown("""<h2 style='text-align: justify; color: white;'>Aquí podemos ver la representación de los valores nulos dentro del dataset</h2>""", unsafe_allow_html=True)
st.pyplot(nulos.get_figure())
# Corregimos los nombres de las columnas
titanic = titanic.rename(columns=lambda x: x.strip().replace(' ', '_').upper())

# Cambiamos el nombre de las ciudades de salida de los pasajeros
titanic['EMBARKED'] = titanic['EMBARKED'].replace(
    {'S': 'Southampton', 'C': 'Cherburgo', 'Q': 'Queenstown'})

# Cambiamos los valores de la columnas PCLASS a Primera, Segunda y Tercera y los hacemos string
titanic["PCLASS"] = titanic["PCLASS"].replace(
    {1: "Primera", 2: "Segunda", 3: "Tercera"}).astype(str)

# Cambiamos los valores nulos de la columna CABIN por Unknown
titanic['CABIN'] = titanic['CABIN'].replace({np.nan: 'Unknown'})

# Ordenamos la columnas PCLASS por Primera, Segunda y Tercera, porque somos clasistas hasta en el análisis de datos.
pclass_order = ['Primera', 'Segunda', 'Tercera']
titanic['PCLASS'] = pd.Categorical(
    titanic['PCLASS'], categories=pclass_order, ordered=True)

# He buscado de dónde salió Rose Amélie Icard que era el dato que faltaba
titanic['EMBARKED'] = titanic['EMBARKED'].replace({np.nan: 'Southampton'})

# Cambiamos los valores de la columna SURVIVED a strings
titanic["SURVIVED"] = titanic["SURVIVED"].replace(
    {1: "Alive", 0: "Dead"}).astype(str)

# Corregimos con la moda la columna de la edad ya que el porcentaje de valores nulos es menor que el 20%
titanic['AGE'].fillna(titanic['AGE'].mode()[0], inplace=True)

st.markdown("""<h3 style='text-align: justify; color: white;'>En el tratamiento de datos he corregido la columna CABIN y los valores nulos los hemos dejado como Unknown. Se ha corregido por la moda la edad de los pasajeros, se han cambiado los nombres las ciudades de partida de los pasajeros, entre otras cosas. </h3>""", unsafe_allow_html=True)

no_nulos = sns.heatmap(titanic.isnull(), cmap='viridis', cbar=False)
st.pyplot(no_nulos.get_figure())


titanic.to_csv(
    '\data\titanic_proces.csv', index=False)

csv = titanic.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV</a>'
st.markdown(href, unsafe_allow_html=True)
