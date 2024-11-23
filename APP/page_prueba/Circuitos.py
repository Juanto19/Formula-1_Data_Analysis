import streamlit as st
import pandas as pd
import plotly.express as px
import sys


def app():
    sys.path.append(r'D:\Cositas\Proyecto_UH\APP')
    import functions as f


    # Cargar datos del circuito (ejemplo)
    def load_circuit_data():
        data = {
            "Nombre": ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 
                       'Australian Grand Prix', 'Azerbaijan Grand Prix', 
                       'Miami Grand Prix', 'Monaco Grand Prix', 
                       'Spanish Grand Prix', 'Canadian Grand Prix', 
                       'Austrian Grand Prix', 'British Grand Prix', 
                       'Hungarian Grand Prix', 'Belgian Grand Prix', 
                       'Dutch Grand Prix', 'Italian Grand Prix', 
                       'Singapore Grand Prix', 'Japanese Grand Prix', 
                       'Qatar Grand Prix', 'United States Grand Prix', 
                       'Mexico City Grand Prix', 'São Paulo Grand Prix', 
                       'Las Vegas Grand Prix', 'Abu Dhabi Grand Prix'],
            "Longitud (km)": [
                5.412, 6.174, 5.278, 6.003, 5.412, 3.337, 4.675, 4.361, 4.318, 5.891, 
                4.381, 7.004, 4.259, 5.793, 5.063, 5.807, 5.419, 5.513, 4.304, 4.309, 
                6.120, 5.281
            ],
            "Curvas": [
                15, 27, 16, 20, 19, 19, 16, 14, 10, 18, 14, 19, 14, 11, 23, 18, 16, 
                20, 17, 15, 17, 21
            ],
            "Velocidad Media (km/h)": [
                207, 252, 230, 205, 225, 160, 210, 212, 240, 228, 200, 238, 205, 
                264, 180, 220, 210, 230, 208, 215, 225, 222
            ],
            "Elevación Máxima (m)": [
                30, 20, 35, 22, 5, 62, 80, 20, 18, 15, 25, 469, 3, 183, 45, 120, 
                15, 28, 2, 5, 10, 12
            ],
            "Elevación Mínima (m)": [
                15, 10, 20, 10, 2, 20, 60, 5, 10, 5, 10, 220, 2, 177, 5, 50, 5, 
                10, 1, 1, 5, 8]
        }
        return pd.DataFrame(data)

    # Cargar los datos
    circuit_data = load_circuit_data()

    # Título principal
    st.title("🏎️ Información de Circuitos de F1")

    # Selección de circuito
    selected_circuit = st.selectbox(
        "Selecciona un circuito para obtener más detalles:",
        circuit_data["Nombre"]
    )

    # Filtrar los datos del circuito seleccionado
    circuit_details = circuit_data[circuit_data["Nombre"] == selected_circuit].iloc[0]

    # Mostrar información general
    st.header(f"Detalles del Circuito: {selected_circuit}")
    col1, col2 = st.columns(2)
    with col1:  
        st.image(f"D:\Cositas\Proyecto_UH\APP\images\circuits\{selected_circuit.lower().replace(' ', '_')}_speed_track.png", caption="Trazado del Circuito")
    with col2:
        st.metric("Longitud", f"{circuit_details['Longitud (km)']} km")
        st.metric("Curvas", circuit_details["Curvas"])
        st.metric("Velocidad Media", f"{circuit_details['Velocidad Media (km/h)']} km/h")
        st.metric("Elevación Máxima", f"{circuit_details['Elevación Máxima (m)']} m")
        st.metric("Elevación Mínima", f"{circuit_details['Elevación Mínima (m)']} m")


    #Mostrar graficas
    circuits_info = pd.read_csv(r"D:\Cositas\Proyecto_UH\data\bueno\2023\circuits_info\circuits_2023_info.csv")

    col2_1, col2_2, col2_3 = st.columns([1,4,1])

    with col2_2:
        fig = f.plot_length_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_mean_speed_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_number_of_laps_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_number_of_turns_circuit(circuits_info, selected_circuit)
        st.pyplot(fig, width=70, height=50)

        fig = f.plot_turns_per_km_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

