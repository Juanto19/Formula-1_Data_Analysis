import streamlit as st
import pandas as pd
import plotly.express as px
import sys


def app():
    # sys.path.append(r'C:\Users\Usuario\Documents\GitHub\Formula-1_Data_Analysis\APP\images\circuits')
    import functions as f

    # Título principal
    st.title("🏎️ Información de Circuitos de F1")

    # Cargar los datos
    circuit_data = pd.read_csv(r".\data\bueno\2023\circuits_info\circuits_2023_info.csv")


    # Renombrar columnas
    circuit_data = circuit_data.rename(columns={
        "Length (km)": "Longitud (km)",
        "Turns": "Curvas",
        "Laps": "Vueltas",
        "Turns/km": "Curvas/km",
        "Mean Speed (km/h)": "Velocidad Media (km/h)"
    })

    # Selección de circuito
    selected_circuit = st.selectbox(
        "Selecciona un circuito para obtener más detalles:",
        circuit_data["EventName"]
    )

    # Filtrar los datos del circuito seleccionado
    circuit_details = circuit_data[circuit_data["EventName"] == selected_circuit].iloc[0]

    # Mostrar información general
    st.header(f"Detalles del Circuito: {selected_circuit}")
    col1, col2 = st.columns(2)
    with col1:  
        st.image(f".\APP\images\circuits\{selected_circuit.lower().replace(' ', '_')}_speed_track.png", caption="Trazado del Circuito")
    with col2:
        st.metric("Longitud", f"{circuit_details['Longitud (km)']} km")
        st.metric("Vueltas", circuit_details["Vueltas"])
        st.metric("Curvas", circuit_details["Curvas"])
        st.metric("Velocidad Media", f"{circuit_details['Velocidad Media (km/h)']} km/h")
        st.metric("Curvas/km", circuit_details["Curvas/km"])

    #Mostrar graficas
    circuits_info = pd.read_csv(r".\data\bueno\2023\circuits_info\circuits_2023_info.csv")

    col2_1, col2_2, col2_3 = st.columns([1,4,1])

    with col2_2:
        fig = f.plot_length_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_number_of_laps_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_mean_speed_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_number_of_turns_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

        fig = f.plot_turns_per_km_circuit(circuits_info, selected_circuit)
        st.pyplot(fig)

