import streamlit as st
import pandas as pd
import sys

def app():
    # sys.path.append('d:/Cositas/Proyecto_UH/APP')
    import functions as f

    # st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>📅 Análisis por Temporada</h1>", unsafe_allow_html=True)
    st.write("Explora los resultados y estadísticas completas de una temporada.")
    # Añade gráficos o tablas aquí.
    season = st.selectbox("Selecciona la temporada", [2024, 2023, 2022, 2021])

    st.markdown(f"<h2 style='text-align: center;'>Resumen de la Temporada {season}</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Clasificación de Pilotos</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        st.image(
            fr'APP/images/points_heatmaps/{season}_drivers_points_heatmap.png', 
                caption=f'Resumen de la Temporada {season}', 
                width=1000)

        st.markdown("<h3 style='text-align: center;'>Clasificación de Constructores</h3>", unsafe_allow_html=True)
        st.image(
            fr'APP/images/points_heatmaps/{season}_teams_points_heatmap.png', 
            caption=f'Resumen de la Temporada {season}', 
            width=1000
        )

    st.markdown("<h3 style='text-align: center;'>Comparativa del Ritmo por Piloto</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.1,8,1.4])
    with col2:
        fig_pace_drv = f.plot_year_pace_driver(season)
        st.plotly_chart(fig_pace_drv)

    
    with col3:
        st.text("")
        st.text("")
        st.text("")
        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Esta gráfica muestra el ritmo medio de cada piloto a lo largo de las distintas carreras. <br>
                El ritmo se calcula como la diferencia de tiempo de un piloto con el tiempo medio en esa vuelta. <br>
                Lo normales estar en un rango de -0.5 a 0.5 segundos. Fuera de este rango se considera un ritmo bueno (<-0.5) o malo (>+0.5) . <br>
                Ritmos por encima de 1 segundo son muy malos y por debajo de -1 segundo son muy buenos.
    </div>
""", unsafe_allow_html=True)
        


    st.text("")
    st.text("")

    st.markdown("<h3 style='text-align: center;'>Comparativa del Ritmo por Constructores</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.1,8,1.4])

    with col2:


        st.text("")
        st.text("")

        fig_pace_team = f.plot_year_pace_team(season)
        
        st.plotly_chart(fig_pace_team)
    
    with col3:

        st.text("")
        st.text("")
        st.text("")
        st.text("")

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Esta gráfica muestra el ritmo medio de cada equipo a lo largo de las distintas carreras. <br>
                El ritmo se calcula como la diferencia de tiempo de un equipo (media de los pilotos) con el tiempo medio en esa vuelta.<br>
                Lo normales estar en un rango de -0.5 a 0.5 segundos. Fuera de este rango se considera un ritmo bueno (<-0.5) o malo (>+0.5) . <br>
                Ritmos por encima de 1 segundo son muy malos y por debajo de -1 segundo son muy buenos.
    </div>
""", unsafe_allow_html=True)

        st.write("")
        st.write("")


    st.markdown("<h3 style='text-align: center;'>Comparativa de Pilotos</h3>", unsafe_allow_html=True)

    year_results= pd.read_csv(fr'APP/data/bueno/{season}/HtH/{season}_results.csv')

    drivers = list(year_results['driverCode'].value_counts().index[:19])
    driver_1 = st.selectbox("Selecciona el primer piloto", drivers)
    driver_2 = st.selectbox("Selecciona el segundo piloto", drivers)

    drivers_to_comp = [driver_1, driver_2]
    if driver_1 == driver_2:
        st.write("Selecciona dos pilotos diferentes para comparar.")
    else:
        comparisons_head_to_head = f.compare_results_pair(season, drivers_to_comp)
        fig_hth = f.plot_comparisons(season, comparisons_head_to_head)
        st.pyplot(fig_hth)

