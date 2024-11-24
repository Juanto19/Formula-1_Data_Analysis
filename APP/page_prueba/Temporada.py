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
    season = st.selectbox("Selecciona la temporada", [2023, 2022, 2021])

    st.markdown(f"<h2 style='text-align: center;'>Resumen de la Temporada {season}</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Clasificación de Pilotos</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        st.image(
            fr'.\APP\images\points_heatmaps\{season}_drivers_points_heatmap.png', 
                caption='Resumen de la Temporada 2023', 
                width=1000)

        st.markdown("<h3 style='text-align: center;'>Clasificación de Constructores</h3>", unsafe_allow_html=True)
        st.image(
            fr'.\APP\images\points_heatmaps\{season}_teams_points_heatmap.png', 
            caption='Resumen de la Temporada 2023', 
            width=1000
        )

        st.markdown("<h3 style='text-align: center;'>Comparativa del Ritmo por Piloto</h3>", unsafe_allow_html=True)
        fig_pace_drv = f.plot_year_pace_driver(season)
        st.plotly_chart(fig_pace_drv)

        st.markdown("<h3 style='text-align: center;'>Comparativa del Ritmo por Constructores</h3>", unsafe_allow_html=True)
        fig_pace_team = f.plot_year_pace_team(season)
        
        st.plotly_chart(fig_pace_team)

        st.markdown("<h3 style='text-align: center;'>Comparativa de Pilotos</h3>", unsafe_allow_html=True)

    year_results= pd.read_csv(fr'.\data\bueno\{season}\HtH\{season}_results.csv')
    # year_sprint_results = pd.read_csv(fr'.\data\bueno\{season}\HtH\{season}_sprint_results.csv')
    # year_q_results = pd.read_csv(fr'.\data\bueno\{season}\HtH\{season}_q_results.csv')

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

