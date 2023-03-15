# --------------------LIBRERÍAS----------------------------#
import streamlit as st
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go


# --------------------CONFIGURACIÓN DE LA PÁGINA----------------------------#
# layout="centered" or "wide"
st.set_page_config(page_title="Titanic: análisis de datos",
                   layout="wide", page_icon="🚢")
st.set_option("deprecation.showPyplotGlobalUse", False)

# --------------------LOGO+CREACIÓN DE COLUMNA----------------------------#
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.title("Titanic: Análisis de datos")
# with col2:
#     st.subheader("")
# with col3:
#     st.markdown(
#         "![Titanic Leonardo Dicaprio GIF](https://media.giphy.com/media/ZiHgApcM5ij1S/giphy.gif)")


# ----------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------#
# Leemos el csv con los datos del Titanic y creamos el dataframe
data = pd.read_csv('data/titanic.csv')
titan = pd.DataFrame(data=data)

# Corregimos los nombres de las columnas
titan = titan.rename(columns=lambda x: x.strip().replace(' ', '_').upper())

# Cambiamos el nombre de las ciudades de salida de los pasajeros
titan['EMBARKED'] = titan['EMBARKED'].replace(
    {'S': 'Southampton', 'C': 'Cherburgo', 'Q': 'Queenstown'})

# Cambiamos los valores de la columnas PCLASS a Primera, Segunda y Tercera y los hacemos string
titan["PCLASS"] = titan["PCLASS"].replace(
    {1: "Primera", 2: "Segunda", 3: "Tercera"}).astype(str)

# Cambiamos los valores nulos de la columna CABIN por Unknown
titan['CABIN'] = titan['CABIN'].replace({np.nan: 'Unknown'})

# Ordenamos la columnas PCLASS por Primera, Segunda y Tercera, porque somos clasistas hasta en el análisis de datos.
pclass_order = ['Primera', 'Segunda', 'Tercera']
titan['PCLASS'] = pd.Categorical(
    titan['PCLASS'], categories=pclass_order, ordered=True)

# He buscado de dónde salió Rose Amélie Icard que era el dato que faltaba
titan['EMBARKED'] = titan['EMBARKED'].replace({np.nan: 'Southampton'})

# Cambiamos los valores de la columna SURVIVED a strings
titan["SURVIVED"] = titan["SURVIVED"].replace(
    {1: "Alive", 0: "Dead"}).astype(str)

# Corregimos con la moda la columna de la edad ya que el porcentaje de valores nulos es menor que el 20%
titan['AGE'].fillna(titan['AGE'].mode()[0], inplace=True)

# --------------------TITLE----------------------------#
st.title("Titanic: my first data analysis")


# ----------------------------SIDEBAR---------------------------------#

columns = titan.columns.tolist()

# Create a list with all the columns of the dataframe
columns = titan.columns.tolist()
st.sidebar.markdown(
    "![Titanic Leonardo Dicaprio GIF](https://media.giphy.com/media/ZiHgApcM5ij1S/giphy.gif)")
# Create the sidebar to select the columns to display
selected_columns = st.sidebar.multiselect(
    "Select the columns you want to display", columns, default=columns)

# Select the columns of the dataframe to display
titan_selected = titan.loc[:, selected_columns]


# --------------------------------MAIN PAGE---------------------------#
# Show the selected dataframe on the main page
st.dataframe(titan_selected)
st.markdown("""---""")
st.markdown(
    "<center><h2><l style='color:white; font-size: 30px;'>Empezamos las gráficas</h2></center>",
    unsafe_allow_html=True,
)

edad_precio = px.scatter(titan, x='AGE', y='FARE', color='SURVIVED', size_max=15, symbol="SEX", template='plotly_dark',
                         title="Relación entre la edad, el precio del pasaje y la supervivencia de los pasajeros", labels={"AGE": "Edad", "SURVIVED": "Superviviente", 'FARE': 'Precio pasaje'})
st.plotly_chart(edad_precio, use_container_width=True)


hist_edad = fig = px.histogram(titan, x="AGE", color='SURVIVED', barmode='group', nbins=20,  template='plotly_dark', labels={
                               "AGE": "Edad", "SURVIVED": "Superviviente"}, title="Distribución de la edad de los pasajeros")
st.plotly_chart(hist_edad, use_container_width=True)

hist_edad_genero = px.histogram(titan, x="AGE", color="SEX", nbins=20,
                                barmode='group', labels={'AGE': 'Edad'},
                                template="plotly_dark")
fig.update_layout(title="Distribución de edades por género",
                  yaxis_title="Número de pasajeros")
st.plotly_chart(hist_edad_genero, use_container_width=True)

# Convertir los datos en long-form
titan_long = pd.melt(titan, id_vars=["SURVIVED"], value_vars=[
                     "SEX", "AGE"], var_name="variable")
# Va representar la supervivencia frente al sexo y la edad
sexo_edad_superv = px.histogram(titan_long, x="value", color="SURVIVED", nbins=20, opacity=0.7,
                                marginal="rug", barmode="overlay", labels={"value": "Sexo/Edad", "count": "Cantidad de pasajeros"},
                                facet_col="variable", category_orders={"variable": ["SEX", "AGE"]}, template="plotly_dark")
# Establecer el título y las etiquetas de los ejes
fig.update_layout(title="Distribución de sexo y edad por Supervivencia",
                  xaxis_title="Sexo/Edad",
                  yaxis_title="Cantidad de pasajeros")
st.plotly_chart(sexo_edad_superv, use_container_width=True)


# Distribución de los pasajeros por género
gender_counts = titan.groupby('SEX')['PASSENGERID'].count().reset_index()
gender_counts = px.pie(gender_counts, values='PASSENGERID', names='SEX',
                       title='Distribución de género de los pasajeros', template='plotly_dark')
st.plotly_chart(gender_counts, use_container_width=True)


# Distribución de supervivencia de los pasajeros
survival_counts = titan.groupby(
    'SURVIVED')['PASSENGERID'].count().reset_index()
survival_counts = px.pie(survival_counts, values='PASSENGERID', names='SURVIVED',
                         title='Distribución de supervivencia de los pasajeros', template='plotly_dark')
st.plotly_chart(survival_counts,  use_container_width=True)


# Número de pasajeros por tipo de pasaje
passenger_class = titan.groupby(
    ['PCLASS', 'SURVIVED']).size().reset_index(name='count')
passenger_class['SURVIVED'] = passenger_class['SURVIVED'].map(
    {'Dead': 'No', 'Alive': 'Sí'})

passenger_class = px.bar(passenger_class, x="PCLASS", y="count", color="SURVIVED", template='plotly_dark', labels={
    "PCLASS": "Clase de pasajero", "count": "Número de pasajeros", "SURVIVED": "Sobreviviente"}, title="Número de pasajeros por tipo de pasaje")
st.plotly_chart(passenger_class,  use_container_width=True)


# Distribución de las tarifas por clase de pasajero
tarifas = px.box(titan, x="PCLASS", y="FARE", points="all", labels={
                 'PCLASS': 'Clase de pasajero', 'FARE': 'Tarifa del pasaje'}, color='PCLASS', template="plotly_dark", category_orders={'PCLASS': pclass_order})
tarifas.update_layout(title="Distribución de tarifas por clase de pasajero")
st.plotly_chart(tarifas,  use_container_width=True)


# Matriz de comparación de los datos
# Creamos un diccionario de colores para asignar a cada valor de la columna "SURVIVED"
colors = {'Dead': 'red', 'Alive': 'blue'}
compare = go.Figure(
    data=go.Splom(
        dimensions=[
            dict(
                label="SURVIVED",
                values=titan["SURVIVED"],
            ),
            dict(label="PCLASS", values=titan["PCLASS"]),
            dict(label="SEX", values=titan["SEX"]),
            dict(label="AGE", values=titan["AGE"]),
            dict(label="SIBSP", values=titan["SIBSP"]),
            dict(label="PARCH", values=titan["PARCH"]),
            dict(label="FARE", values=titan["FARE"]),
        ],
        showupperhalf=False,
        marker=dict(showscale=False, line_color="white",
                    color=titan["SURVIVED"].map(colors)),
    )
)
# Añadimos el título y ajustamos el tamaño
compare.update_layout(
    title="Matriz de comparación para los datos del Titanic",
    autosize=False,
    width=900,
    height=900,
    template="plotly_dark",
)
st.plotly_chart(compare,  use_container_width=True)


# Matriz de correlación
corr = titan.corr(numeric_only=True)
correlation = px.imshow(
    corr, color_continuous_scale='viridis_r', zmin=-1, zmax=1)
correlation.update_layout(
    title="Correlación entre variables", template="plotly_dark")
st.plotly_chart(correlation,  use_container_width=True)


# Histograma de dónde embarcaron los pasajeros
EMBARKED_counts = titan.groupby(
    'EMBARKED')['PASSENGERID'].count().reset_index()
# Especificar el orden deseado de las ciudades de embarque
order = ['Southampton', 'Cherburgo', 'Queenstown']
embark = px.bar(EMBARKED_counts, x='EMBARKED', y='PASSENGERID', labels={
                'EMBARKED': 'Ciudad de embarque', 'PASSENGERID': 'Cantidad de pasajeros'}, template='plotly_dark', category_orders={'EMBARKED': order})
st.plotly_chart(embark,  use_container_width=True)


# Porcentaje de supervivientes dependiendo de su lugar de procedencia
EMBARKED_survival_counts = titan.groupby(['EMBARKED', 'SURVIVED'])[
    'PASSENGERID'].count().reset_index()

EMBARKED_survival_counts['Total'] = EMBARKED_survival_counts.groupby(
    ['EMBARKED'])['PASSENGERID'].transform('sum')
EMBARKED_survival_counts['Proportion'] = EMBARKED_survival_counts['PASSENGERID'] / \
    EMBARKED_survival_counts['Total']

order = ['Southampton', 'Cherburgo', 'Queenstown']

surv_embark = go.Figure()
surv_embark.add_trace(go.Bar(x=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED'] == 'Dead']['EMBARKED'],
                             y=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                        == 'Dead']['Proportion'],
                             name='No sobrevivió',
                             text=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                           == 'Dead']['PASSENGERID'],
                             hovertemplate='Pasajeros: %{text}<br>Porcentaje: %{y:.2%}<extra></extra>',
                             marker=dict(color='#1f77b4')))
surv_embark.add_trace(go.Bar(x=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED'] == 'Alive']['EMBARKED'],
                             y=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                        == 'Alive']['Proportion'],
                             name='Sobrevivió',
                             text=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                           == 'Alive']['PASSENGERID'],
                             hovertemplate='Pasajeros: %{text}<br>Porcentaje: %{y:.2%}<extra></extra>',
                             marker=dict(color='#ff7f0e')))
surv_embark.update_layout(barmode='stack',
                          xaxis_title='Ciudad de embarque',
                          yaxis_title='Proporción de pasajeros',
                          legend_title='Supervivencia',
                          title='Porcentaje de supervivientes y fallecidos por ciudad de embarque',
                          template='plotly_dark',
                          xaxis={'categoryorder': 'array', 'categoryarray': order})
st.plotly_chart(surv_embark,  use_container_width=True)
