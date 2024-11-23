import streamlit as st
import sys

# sys.path.append(r'.\page_prueba')
# APP\page_prueba\Grand_Prix.py
from page_prueba.Grand_Prix import app as gran_premio_app
from APP.page_prueba.Temporada import app as temporadas_app
from APP.page_prueba.Circuitos import app as circuitos_app
from APP.page_prueba.Contacto import app as contacto_app


# Configuración de la aplicación
st.set_page_config(
    page_title="Análisis F1", 
    page_icon="🏎️", 
    layout="wide",    
    initial_sidebar_state="collapsed"
)
# Aplicar el archivo CSS para la personalización del tema
with open(r".\APP\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Título y subtítulo
st.markdown("<h1 style='text-align: center;'>🏎️ Análisis de Datos de Fórmula 1</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Explora estadísticas avanzadas y visualizaciones detalladas del mundo de la F1.</h3>", unsafe_allow_html=True)

# Imagen de fondo o representativa
st.image(r".\APP\Customization\imagen_bienvenida.jpg", 
         caption="Bienvenido al análisis interactivo de la Fórmula 1",
         use_container_width =True, 
         width=1000)

# Breve introducción
st.write("""
Esta aplicación te permite:
- 📊 Analizar en detalle los resultados de cada Gran Premio.
- 🏁 Explorar comparativas entre pilotos y equipos.
- 📅 Consultar estadísticas históricas y actuales.
""")

# Configurar estado inicial de la aplicación
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Función para cambiar de página
def navigate_to(page):
    st.session_state["page"] = page

# Diccionario de páginas
# pages = {
#     "GP Analysis": gran_premio_app,
#     "Season Analysis": temporadas_app,
#     "Contact": contacto_app

# }

# Botones para navegación rápida
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📊 Análisis de GP"):
        navigate_to("📊 Análisis de GP")
with col2:
    if st.button("📅 Temporadas"):
        navigate_to("📅 Temporadas")
with col3:
    if st.button("🏟️ Circuitos"):
        navigate_to("🏟️ Circuitos")
with col4:
    if st.button("📩 Contacto"):
        navigate_to("📩 Contacto")


# Renderizado dinámico según la página seleccionada
if st.session_state["page"] == "📊 Análisis de GP":
    gran_premio_app()
elif st.session_state["page"] == "📅 Temporadas":
    temporadas_app()
elif st.session_state["page"] == "🏟️ Circuitos":
    circuitos_app()
elif st.session_state["page"] == "📩 Contacto":
    contacto_app()

# Pie de página (opcional)
st.write("---")
st.caption("Aplicación desarrollada por [Juan Torralbo](https://github.com/Juanto19) | Inspirado en la pasión por la Fórmula 1.")
