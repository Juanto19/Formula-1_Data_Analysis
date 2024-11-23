import streamlit as st


def app():
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
    st.subheader(f"📍 Información del {gran_premio}")
    if gran_premio == "Bahrain Grand Prix":
        st.write("""
        - **Circuito**: Bahrain International Circuit  
        - **Fecha**: 5 de Marzo de 2023  
        - **Vueltas**: 57  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "Saudi Arabian Grand Prix":
        st.write("""
        - **Circuito**: Jeddah Street Circuit  
        - **Fecha**: 19 de Marzo de 2023  
        - **Vueltas**: 50  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "Australian Grand Prix":
        st.write("""
        - **Circuito**: Albert Park Circuit  
        - **Fecha**: 2 de Abril de 2023  
        - **Vueltas**: 58  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "Azerbaijan Grand Prix":
        st.write("""
        - **Circuito**: Baku City Circuit  
        - **Fecha**: 30 de Abril de 2023  
        - **Vueltas**: 51  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "Miami Grand Prix":
        st.write("""
        - **Circuito**: Miami International Autodrome  
        - **Fecha**: 7 de Mayo de 2023  
        - **Vueltas**: 57  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "Monaco Grand Prix":
        st.write("""
        - **Circuito**: Circuit de Monaco  
        - **Fecha**: 28 de Mayo de 2023  
        - **Vueltas**: 78  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "Spanish Grand Prix":
        st.write("""
        - **Circuito**: Circuit de Barcelona-Catalunya  
        - **Fecha**: 4 de Junio de 2023  
        - **Vueltas**: 66  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "Canadian Grand Prix":
        st.write("""
        - **Circuito**: Circuit Gilles Villeneuve  
        - **Fecha**: 18 de Junio de 2023  
        - **Vueltas**: 70  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "Austrian Grand Prix":
        st.write("""
        - **Circuito**: Red Bull Ring  
        - **Fecha**: 2 de Julio de 2023  
        - **Vueltas**: 71  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "British Grand Prix":
        st.write("""
        - **Circuito**: Silverstone Circuit  
        - **Fecha**: 9 de Julio de 2023  
        - **Vueltas**: 52  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "Hungarian Grand Prix":
        st.write("""
        - **Circuito**: Hungaroring  
        - **Fecha**: 23 de Julio de 2023  
        - **Vueltas**: 70  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "Belgian Grand Prix":
        st.write("""
        - **Circuito**: Circuit de Spa-Francorchamps  
        - **Fecha**: 30 de Julio de 2023  
        - **Vueltas**: 44  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "Dutch Grand Prix":
        st.write("""
        - **Circuito**: Circuit Zandvoort  
        - **Fecha**: 27 de Agosto de 2023  
        - **Vueltas**: 72  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "Italian Grand Prix":
        st.write("""
        - **Circuito**: Monza Circuit  
        - **Fecha**: 3 de Septiembre de 2023  
        - **Vueltas**: 53  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "Singapore Grand Prix":
        st.write("""
        - **Circuito**: Marina Bay Street Circuit  
        - **Fecha**: 17 de Septiembre de 2023  
        - **Vueltas**: 61  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "Japanese Grand Prix":
        st.write("""
        - **Circuito**: Suzuka International Racing Course  
        - **Fecha**: 24 de Septiembre de 2023  
        - **Vueltas**: 53  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "Qatar Grand Prix":
        st.write("""
        - **Circuito**: Losail International Circuit  
        - **Fecha**: 8 de Octubre de 2023  
        - **Vueltas**: 57  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "United States Grand Prix":
        st.write("""
        - **Circuito**: Circuit of the Americas  
        - **Fecha**: 22 de Octubre de 2023  
        - **Vueltas**: 56  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "Mexico City Grand Prix":
        st.write("""
        - **Circuito**: Autódromo Hermanos Rodríguez  
        - **Fecha**: 29 de Octubre de 2023  
        - **Vueltas**: 71  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)
    elif gran_premio == "São Paulo Grand Prix":
        st.write("""
        - **Circuito**: Interlagos Circuit  
        - **Fecha**: 5 de Noviembre de 2023  
        - **Vueltas**: 71  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Sergio Perez", "Max Verstappen", "Fernando Alonso", "George Russell", "Lewis Hamilton"],
            "Equipo": ["Red Bull", "Red Bull", "Aston Martin", "Mercedes", "Mercedes"],
            "Tiempo": ["1:31:43.456", "+3.123", "+7.456", "+10.789", "+15.234"]
        }
        st.table(data)
    elif gran_premio == "Las Vegas Grand Prix":
        st.write("""
        - **Circuito**: Las Vegas Street Circuit  
        - **Fecha**: 18 de Noviembre de 2023  
        - **Vueltas**: 50  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Charles Leclerc", "Sergio Perez", "George Russell", "Lewis Hamilton", "Carlos Sainz"],
            "Equipo": ["Ferrari", "Red Bull", "Mercedes", "Mercedes", "Ferrari"],
            "Tiempo": ["1:30:43.456", "+4.123", "+6.456", "+9.789", "+14.234"]
        }
        st.table(data)
    elif gran_premio == "Abu Dhabi Grand Prix":
        st.write("""
        - **Circuito**: Yas Marina Circuit  
        - **Fecha**: 26 de Noviembre de 2023  
        - **Vueltas**: 55  
        """)
        data = {
            "Posición": [1, 2, 3, 4, 5],
            "Piloto": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Fernando Alonso", "Carlos Sainz"],
            "Equipo": ["Red Bull", "Ferrari", "Mercedes", "Aston Martin", "Ferrari"],
            "Tiempo": ["1:32:43.456", "+5.123", "+8.456", "+12.789", "+20.234"]
        }
        st.table(data)


    # Gráfica (placeholder)
    st.subheader("📊 Gráficas de Rendimiento")
    st.write("Aquí se incluirán las gráficas, como tiempos por vuelta o posiciones en pista.")
