import streamlit as st
import pandas as pd

def app():
    import APP.plotting as f
    st.markdown("<h1 style='text-align: center;'>📊 Análisis de un Gran Premio</h1>", unsafe_allow_html=True)
    st.write("Selecciona un Gran Premio para ver el análisis detallado.")
    # Puedes añadir widgets como selectboxes aquí.


    # Dropdown para seleccionar el año
    year = st.selectbox(
        "Selecciona un Año:",
        [2024, 2023] # , "2022", "2021"
    )
    circuits_info = pd.read_csv(rf"APP/data/bueno/2023/circuits_info/circuits_2023_info.csv")

    circuit_data = circuits_info.rename(columns={
        "Length (km)": "Longitud (km)",
        "Turns": "Curvas",
        "Laps": "Vueltas",
        "Turns/km": "Curvas/km",
        "Mean Speed (km/h)": "Velocidad Media (km/h)", 

    })

    # Dropdown para seleccionar el Gran Premio basado en el año seleccionado
    if year == 2023 or year == 2024:
        gran_premio = st.selectbox(
            "Selecciona un Gran Premio:",
            [
                '','Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 
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
    if gran_premio != "":
        st.markdown(f"<h2 style='text-align: center;'>📍 Información del {gran_premio} - {year}</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        with col1:
            st.write(f"""
            - **Circuito**: {circuit_data[circuit_data["EventName"] == gran_premio].iloc[0]["Circuit"]}    
            - **Localización**: {circuit_data[circuit_data["EventName"] == gran_premio].iloc[0]["Location"]}
            - **Fecha**: {circuit_data[circuit_data["EventName"] == gran_premio].iloc[0]["Date"]}  
            - **Vueltas**: {circuit_data[circuit_data["EventName"] == gran_premio].iloc[0]["Vueltas"]}  
            """)
            st.image(f"APP/images/circuits/{gran_premio}_track.png", caption="Trazado del Circuito")

        data = pd.read_csv(rf"APP/data/bueno/{year}/results_info/{gran_premio}_results.csv")
        data = data.rename(columns={
            "Position": "Posición",
            "Driver": "Piloto",
            "Laps": "Vueltas",
            "Time": "Tiempo",
            "Points": "Puntos",
            'Status': 'Resultado'
        })
        with col2:
            st.table(data.set_index('Posición'))
    

        # Gráfica (placeholder)
        st.markdown("<h2 style='text-align: center;'>📊 Gráficas de Rendimiento</h2>", unsafe_allow_html=True)
        # st.write("Aquí se incluirán las gráficas, como tiempos por vuelta o posiciones en pista.")

        st.markdown("<h3 style='text-align: center;'>⏱️ Clasificación</h3>", unsafe_allow_html=True)

        fig_tiempos_quali = f.plot_qualifying_times(year, gran_premio)
        st.pyplot(fig_tiempos_quali)

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Se muestra la diferencia con respecto al tiempo de cualificación. <br>
    </div>
""", unsafe_allow_html=True)
        
        fig_telemetry = f.plot_overlap_telemetries(year, gran_premio)
        st.plotly_chart(fig_telemetry)

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Información de la telemetría (velocidad, acelerador y freno) durante la vuelta de clasificación de cada piloto. <br>
    </div>
""", unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center;'>🏁 Carrera</h3>", unsafe_allow_html=True)
        fig_pos = f.plot_position_changes(year, gran_premio)
        st.pyplot(fig_pos)

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Avance de las posiciones en cada vuelta de la carrera. <br>
    </div>
""", unsafe_allow_html=True)
        fig_tiempo = f.plot_laptimes_race(year, gran_premio)
        st.plotly_chart(fig_tiempo)

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Comparación de los tiempos de cada vuelta a lo largo de la carrera. <br>
    </div>
""", unsafe_allow_html=True)

        fig_dist = f.plot_relative_distances(year, gran_premio)
        st.plotly_chart(fig_dist)

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Distancia (en segundos) con respecto al líder de la carrera en cada vuelta. <br>
    </div>
""", unsafe_allow_html=True)

        fig_pitstop = f.plot_pitstop_estrategy(year, gran_premio)
        st.pyplot(fig_pitstop)

        st.markdown("""
    <div style="
        background-color: #fffcce; 
        border-left: 6px solid #d4c80a; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 16px;">
        <strong>Info:</strong> Estrategia de neumáticos de cada piloto. <br>
    </div>
""", unsafe_allow_html=True)





