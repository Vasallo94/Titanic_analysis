# --------------------LIBRERÍAS----------------------------#
import streamlit as st
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import re
from plotly.subplots import make_subplots


def main():
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
    data = pd.read_csv('data/titanic_proces.csv')
    titan = pd.DataFrame(data=data)
    
    st.image('img/00-Profile_H2.webp',use_container_width=True)

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
    colores = ['#972a27', '#00be4c']
    # --------------------------------MAIN PAGE---------------------------#
    # Show the selected dataframe on the main page
    st.dataframe(titan_selected)
    st.markdown("""---""")
    st.markdown(
        "<center><h2><l style='color:white; font-size: 30px;'>Visualización y estudio de los datos</h2></l></center>",
        unsafe_allow_html=True,
    )

    st.title("Selección de temas a estudiar")

    menu = ["Correlaciones entre los datos",
            "Conociendo a los pasajeros", "Sobre el precio de los pasajes", "Ruta y ciudades de embarque", "In memoriam..."]
    choice = st.sidebar.selectbox("Seleccione una pestaña", menu)

    if choice == "Correlaciones entre los datos":
        # Contenido de la Correlaciones entre los datos
        st.subheader("Correlaciones entre los datos del data set del Titanic")
        st.write(
            "Una manera visual de hacernos una primera idea de los datos y cómo se correlacionan entre sí.")
        # Matriz de comparación de los datos
        # Creamos un diccionario de colores para asignar a cada valor de la columna "SURVIVED"

        colors = {'Dead': '#972a27', 'Alive': '#00be4c'}
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

    elif choice == "Conociendo a los pasajeros":
        # Contenido de la Conociendo a los pasajeros
        st.subheader("Conociendo a los pasajeros")
        st.write(
            "Un estudio de los pasajeros del Titanic atendiendo a diferentes factores y viendo si hay una relación con la supervivencia de los mismos.")
        colores = ['#972a27', '#00be4c']

        hist_edad = px.histogram(titan, x="AGE", color='SURVIVED', barmode='group', nbins=40,  template='plotly_dark', labels={
            "AGE": "Edad", "SURVIVED": "Superviviente"}, color_discrete_sequence=colores, title="Distribución de la edad de los pasajeros")
        hist_edad.update_yaxes(title_text='Número de pasajeros')
        st.plotly_chart(hist_edad, use_container_width=True)

        # Distribución de edad-sexo y supervivencia
        sexo_edad_superv = px.histogram(titan, x='AGE', color='SURVIVED', barmode='overlay', nbins=40,
                                        marginal='box', opacity=0.7, template='plotly_dark', facet_col='SEX', color_discrete_sequence=colores,
                                        category_orders={'SEX': ['male', 'female'], 'SURVIVED': ['Dead', 'Alive']})

        sexo_edad_superv.update_layout(
            title='Distribución de edades por género y supervivencia',
            xaxis_title='Edad',
            yaxis_title='Número de pasajeros'
        )
        st.plotly_chart(sexo_edad_superv, use_container_width=True)

        # Create subplots with two rows and one column
        donuts = make_subplots(rows=1, cols=2, specs=[
            [{'type': 'pie'}, {'type': 'pie'}]])

        # Group by gender and count the number of passengers
        gender_counts = titan.groupby(
            'SEX')['PASSENGERID'].count().reset_index()
        # Group by survival and count the number of passengers
        survival_counts = titan.groupby(
            'SURVIVED')['PASSENGERID'].count().reset_index()

        # Add the first pie chart to the first row
        donuts.add_trace(
            go.Pie(
                labels=gender_counts['SEX'],
                values=gender_counts['PASSENGERID'],
                hole=0.6,
                name='Género',
                textinfo='percent+label',
                hoverinfo='label+percent',
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Add the second pie chart to the second row
        donuts.add_trace(
            go.Pie(
                labels=survival_counts['SURVIVED'],
                values=survival_counts['PASSENGERID'],
                hole=0.6,
                name='Supervivencia',
                marker=dict(colors=colores),
                textinfo='percent+label',
                hoverinfo='label+percent',
                showlegend=True,
            ),
            row=1, col=2,
        )

        donuts.update_traces(
            textfont=dict(color='white', size=25),
            hovertemplate=None,
            textposition='inside',
        )

        # Set the title and layout of the figure
        donuts.update_layout(
            title='Distribución de género y supervivencia de los pasajeros',
            height=800,
            showlegend=True, template='plotly_dark'
        )
        st.plotly_chart(donuts, use_container_width=True)

        # Filtrar los pasajeros menores de 18 años
        kids = titan[titan['AGE'] < 18]
        # Crear un histograma con la cantidad de niños por años cumplidos y si murieron o no
        kid = px.histogram(kids, x='AGE', color='SURVIVED', nbins=18,
                           labels={'AGE': 'Edad (años)',
                                   'count': 'Cantidad de niños'},
                           color_discrete_sequence=colores)
        kid.update_layout(title='Cantidad de niños por edad y si murieron o no',
                          xaxis_title_text='Edad (años)', yaxis_title_text='Cantidad de niños',
                          template='plotly_dark')
        st.plotly_chart(kid,  use_container_width=True)

    elif choice == "Sobre el precio de los pasajes":
        # Contenido de la Sobre el precio de los pasajes
        st.subheader(
            "Sobre el precio de los pasajes")
        st.write(
            "Un estudio sobre las clases sociales de los pasajeros dentro del Titanic.")
        # Número de pasajeros por tipo de pasaje
        symbols = ['cross', 'diamond']
        edad_precio = px.scatter(titan, x='AGE', y='FARE', color='SURVIVED', size_max=25, symbol="SEX", symbol_sequence=symbols, color_discrete_sequence=colores, template='plotly_dark',
                                 title="Relación entre la edad, el precio del pasaje y la supervivencia de los pasajeros", labels={"AGE": "Edad", "SURVIVED": "Superviviente", 'FARE': 'Precio pasaje'}, symbol_map={})
        st.plotly_chart(edad_precio, use_container_width=True)

        passenger_class = titan.groupby(
            ['PCLASS', 'SURVIVED']).size().reset_index(name='count')
        passenger_class['SURVIVED'] = passenger_class['SURVIVED'].map(
            {'Dead': 'No', 'Alive': 'Sí'})

        biglietti = px.bar(passenger_class, barmode='group', x="PCLASS", y="count", color="SURVIVED", color_discrete_sequence=colores, template='plotly_dark', labels={
            "PCLASS": "Clase de pasajero", "count": "Número de pasajeros", "SURVIVED": "Superviviente"}, title="Número de pasajeros por tipo de pasaje")
        st.plotly_chart(biglietti, use_container_width=True)

        # Distribución de las tarifas por clase de pasajero
        pclass_order = ['Primera', 'Segunda', 'Tercera']
        titan['PCLASS'] = pd.Categorical(
            titan['PCLASS'], categories=pclass_order, ordered=True)
        tarifas = px.box(titan, x="PCLASS", y="FARE", points="all", labels={
            'PCLASS': 'Clase de pasajero', 'FARE': 'Tarifa del pasaje'}, color='PCLASS', template="plotly_dark", category_orders={'PCLASS': pclass_order})
        tarifas.update_layout(
            title="Distribución de tarifas por clase de pasajero")
        st.plotly_chart(tarifas, use_container_width=True)

        d3 = go.Figure(data=[go.Scatter3d(
            x=titan['AGE'],
            y=titan['PCLASS'],
            z=[1 if s == 'Alive' else 0 for s in titan['SURVIVED']],
            mode='markers',
            marker=dict(
                size=5,
                color=[0 if s == 'Dead' else 1 for s in titan['SURVIVED']],
                colorscale=[[0, '#972a27'], [1, '#00be4c']],
                opacity=0.8
            )
        )])

        d3.update_layout(
            scene=dict(
                xaxis_title='Age',
                yaxis_title='Pclass',
                zaxis_title='Survived',
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=10
                ),
                yaxis=dict(
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                zaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
                aspectratio=dict(x=1, y=1, z=1),
                camera_eye=dict(x=2, y=2, z=2),
            ),
            title='Edad, clase del billete y supervivencia',
            template='plotly_dark'
        )

        st.plotly_chart(d3, use_container_width=True)

    elif choice == "Ruta y ciudades de embarque":
        # Contenido de la Ruta y ciudades de embarque
        st.subheader("Ruta y ciudades de embarque")
        st.write(
            "Aquí se puede agregar el contenido de la Ruta y ciudades de embarque.")
        # Ciudades en lla ruta del titanic
        order = ['Southampton', 'Cherburgo', 'Queenstown']

        EMBARKED_counts = titan.groupby(
            'EMBARKED')['PASSENGERID'].count().reset_index()
        # Especificar el orden deseado de las ciudades de embarque
        order = ['Southampton', 'Cherburgo', 'Queenstown']
        embark = px.bar(EMBARKED_counts, x='EMBARKED', y='PASSENGERID', labels={
                        'EMBARKED': 'Ciudad de embarque', 'PASSENGERID': 'Cantidad de pasajeros'}, template='plotly_dark', category_orders={'EMBARKED': order})
        st.plotly_chart(embark,  use_container_width=True)

        EMBARKED_survival_counts = titan.groupby(['EMBARKED', 'SURVIVED'])[
            'PASSENGERID'].count().reset_index()

        EMBARKED_survival_counts['Total'] = EMBARKED_survival_counts.groupby(
            ['EMBARKED'])['PASSENGERID'].transform('sum')
        EMBARKED_survival_counts['Proportion'] = EMBARKED_survival_counts['PASSENGERID'] / \
            EMBARKED_survival_counts['Total']

        surv_embark = go.Figure()
        surv_embark.add_trace(go.Bar(x=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED'] == 'Dead']['EMBARKED'],
                                     y=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                                == 'Dead']['Proportion'],
                                     name='No sobrevivió',
                                     text=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                                   == 'Dead']['PASSENGERID'],
                                     hovertemplate='Pasajeros: %{text}<br>Porcentaje: %{y:.2%}<extra></extra>',
                                     marker=dict(color='#972a27')))
        surv_embark.add_trace(go.Bar(x=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED'] == 'Alive']['EMBARKED'],
                                     y=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                                == 'Alive']['Proportion'],
                                     name='Sobrevivió',
                                     text=EMBARKED_survival_counts[EMBARKED_survival_counts['SURVIVED']
                                                                   == 'Alive']['PASSENGERID'],
                                     hovertemplate='Pasajeros: %{text}<br>Porcentaje: %{y:.2%}<extra></extra>',
                                     marker=dict(color='#00be4c')))
        surv_embark.update_layout(barmode='group',
                                  xaxis_title='Ciudad de embarque',
                                  yaxis_title='Proporción de pasajeros',
                                  legend_title='Supervivencia',
                                  title='Porcentaje de supervivientes y fallecidos por ciudad de embarque',
                                  template='plotly_dark',
                                  xaxis={'categoryorder': 'array', 'categoryarray': order})

        st.plotly_chart(surv_embark,  use_container_width=True)

    elif choice == "In memoriam...":
        # Contenido de la Sobre el precio de los pasajes a los pasajeros
        st.subheader("In memoriam...")
        # st.write(
        #     "Aquí se puede agregar el contenido de la Sobre el precio de los pasajes a los pasajeros.")
        # Títulos y nubes de palabras
        st.image('img/wordcloud.png', use_column_width=True)


if __name__ == '__main__':
    main()
