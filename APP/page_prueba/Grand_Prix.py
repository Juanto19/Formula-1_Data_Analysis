import streamlit as st
import pandas as pd

def app():
    import functions as f
    st.title("📊 Análisis de un Gran Premio")
    st.write("Selecciona un Gran Premio para ver el análisis detallado.")
    # Puedes añadir widgets como selectboxes aquí.

    st.title("🏁 Resultados de un Gran Premio")

    # Dropdown para seleccionar el año
    year = st.selectbox(
        "Selecciona un Año:",
        ["2023"] # , "2022", "2021"
    )

    # Dropdown para seleccionar el Gran Premio basado en el año seleccionado
    if year == "2023":
        gran_premio = st.selectbox(
            "Selecciona un Gran Premio:",
            [
                'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 
                'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 
                'Spanish Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix', 
                'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 
                'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 
                'Japanese Grand Prix', 'Qatar Grand Prix', 'United States Grand Prix', 
                'Mexico City Grand Prix', 'São Paulo Grand Prix', 'Las Vegas Grand Prix', 
                'Abu Dhabi Grand Prix'
            ]
        )
    # elif year == "2022":
    #     gran_premio = st.selectbox(
    #         "Selecciona un Gran Premio:",
    #         ["Bahrain GP 2022", "Monaco GP 2022", "Spa GP 2022", "Silverstone GP 2022"]
    #     )
    # elif year == "2021":
    #     gran_premio = st.selectbox(
    #         "Selecciona un Gran Premio:",
    #         ["Bahrain GP 2021", "Monaco GP 2021", "Spa GP 2021", "Silverstone GP 2021"]
    #     )

    # Información del GP seleccionado
    col1, col2 = st.columns([1,3])
    if gran_premio == "Bahrain Grand Prix":
        st.image(f".\APP\images\circuits\{gran_premio.lower().replace(' ', '_')}_track.png", caption="Trazado del Circuito")
        with col1:
            st.markdown(f"<h2 style='text-align: center;'>📍 Información del {gran_premio}</h2>", unsafe_allow_html=True)
            st.write("""
            - **Circuito**: Bahrain International Circuit  
            - **Fecha**: 5 de Marzo de 2023  
            - **Vueltas**: 57  
            """)
        data = pd.read_csv(r".\data\bueno\2023\results_info\Brasil_results.csv")
        with col2:
            st.markdown(f"<h2 style='text-align: center;'>📍 Información del {gran_premio}</h2>", unsafe_allow_html=True)
            st.table(data)
    

    # Gráfica (placeholder)
    st.markdown("<h2 style='text-align: center;'>📊 Gráficas de Rendimiento</h2>", unsafe_allow_html=True)
    # st.write("Aquí se incluirán las gráficas, como tiempos por vuelta o posiciones en pista.")

    st.markdown("<h3 style='text-align: center;'>⏱️ Clasificación</h3>", unsafe_allow_html=True)

    fig_tiempos_quali = f.plot_qualifying_times(year, gran_premio)
    st.pyplot(fig_tiempos_quali)

    

    st.markdown("<h3 style='text-align: center;'>🏁 Carrera</h3>", unsafe_allow_html=True)
    fig_pos = f.plot_position_changes(year, gran_premio)
    st.pyplot(fig_pos)

    fig_tiempo = f.plot_laptimes_race(year, gran_premio)
    st.plotly_chart(fig_tiempo)

    fig_pitstop = f.plot_pitstop_estrategy(year, gran_premio)
    st.pyplot(fig_pitstop)





