import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Configuración de la página y diseño
st.set_page_config(
    page_title="Penetrómetro de golpes - INTA EEA Corrientes", 
    page_icon="🌾", 
    layout="wide"
)

# --- Estilos CSS personalizados ---
st.markdown("""
    <style>
    .main {
        background-color: #f9fbfc;
    }
    .stSidebar {
        background-color: #f1f5f9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Penetrómetro de golpes")
st.markdown("Herramienta de procesamiento digital individual para el análisis de resistencia mecánica del suelo.")

# --- Panel Lateral de Configuración y Créditos ---
st.sidebar.header("⚙️ Parámetros del Ensayo")
masa_kg = st.sidebar.number_input("Masa del pilón (kg)", value=2.5, min_value=0.1, step=0.1)
altura_caida_m = st.sidebar.number_input("Altura de caída (m)", value=0.5, min_value=0.01, step=0.05)
diametro_cono_cm = st.sidebar.number_input("Diámetro de la base del cono (cm)", value=2.0, min_value=0.1, step=0.1)
escala_cm_px = st.sidebar.number_input("Calibración (cm por píxel)", value=0.05, min_value=0.001, step=0.005, format="%.4f")

radio_m = (diametro_cono_cm / 100.0) / 2.0
area_cono_m2 = np.pi * (radio_m ** 2)

# Créditos discretos en el pie del panel lateral
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 0.8em; color: gray; text-align: center;'>"
    "Desarrollado por:<br>"
    "<b>Ing. Agr. (Mgter.) Federico A. Paredes</b><br>"
    "<a href='mailto:paredesfedericoantonio@gmail.com'>paredesfedericoantonio@gmail.com</a>"
    "</div>", 
    unsafe_allow_html=True
)

# --- Creación de Pestañas Principales ---
tab_carga, tab_datos, tab_graficos, tab_manual = st.tabs([
    "📁 Carga de Faja", 
    "📊 Planilla y Datos", 
    "📈 Gráfico de Resistencia", 
    "📚 Biblioteca y Manual"
])

# Variables de sesión para persistencia individual
if 'df_resultado' not in st.session_state:
    st.session_state['df_resultado'] = None
if 'nombre_muestra_actual' not in st.session_state:
    st.session_state['nombre_muestra_actual'] = "Muestra"


# ==========================================
# 1. PESTAÑA: CARGA DE FAJA (📁)
# ==========================================
with tab_carga:
    st.header("📂 Carga de Faja Individual")
    st.markdown("Seleccione o arrastre la imagen de la faja digitalizada. **Sugerencia:** Nómbre previamente su archivo de imagen con el nombre de la muestra (ej: `Lote_SanCayetano.jpg`).")
    
    uploaded_file = st.file_uploader(
        "Seleccionar imagen de faja (Formatos: JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            st.image(img, caption=f"Imagen cargada: {uploaded_file.name}", use_column_width=True)
            
            # Tomar el nombre directamente del archivo subido (sin extensión)
            nombre_muestra = uploaded_file.name.rsplit('.', 1)[0]
            st.session_state['nombre_muestra_actual'] = nombre_muestra
            
            if st.button("🚀 Procesar Faja", type="primary"):
                # Procesamiento físico de marcas
                blur = cv2.GaussianBlur(img, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                perfil_proyeccion = np.sum(thresh, axis=1)
                
                from scipy.signal import find_peaks
                picos, _ = find_peaks(perfil_proyeccion, distance=10, prominence=500)
                
                profundidades_cm = picos * escala_cm_px
                profundidades_cm = np.sort(profundidades_cm)

                if len(profundidades_cm) >= 2:
                    prof_metros = profundidades_cm / 100.0
                    delta_e = np.diff(prof_metros)
                    delta_e = np.where(delta_e == 0, 0.0001, delta_e)

                    g = 9.81
                    energia_golpe = masa_kg * g * altura_caida_m
                    resistencia_mpa = (energia_golpe / (area_cono_m2 * delta_e)) / 1e6

                    df_golpes = pd.DataFrame({
                        'Muestra': nombre_muestra,
                        'Golpe': range(1, len(delta_e) + 1),
                        'Profundidad_Final_cm': profundidades_cm[1:],
                        'Penetracion_cm': delta_e * 100,
                        'Resistencia_MPa': resistencia_mpa
                    })

                    bins = np.arange(0, df_golpes['Profundidad_Final_cm'].max() + 10, 10)
                    labels = [f"{bins[i]}-{bins[i+1]} cm" for i in range(len(bins)-1)]
                    
                    df_golpes['Intervalo'] = pd.cut(df_golpes['Profundidad_Final_cm'], bins=bins, labels=labels, include_lowest=True)
                    
                    df_intervalos = df_golpes.groupby('Intervalo', observed=False)['Resistencia_MPa'].agg(
                        ['mean', 'count', 'max']
                    ).reset_index()
                    df_intervalos.rename(columns={
                        'mean': 'Resistencia_Media_MPa', 
                        'count': 'Total_Golpes', 
                        'max': 'Resistencia_Max_MPa'
                    }, inplace=True)
                    df_intervalos.insert(0, 'Muestra', nombre_muestra)

                    st.session_state['df_resultado'] = df_intervalos
                    st.success(f"🎉 ¡Faja '{nombre_muestra}' procesada con éxito! Revise las pestañas de Planilla o Gráfico.")
                else:
                    st.error("No se detectaron suficientes marcas claras en la faja. Verifique la nitidez o ajuste la calibración.")


# ==========================================
# 2. PESTAÑA: PLANILLA Y DATOS (📊)
# ==========================================
with tab_datos:
    st.header("📊 Planilla de Resultados")
    st.markdown("Valores calculados por intervalo de profundidad para la muestra actual, listos para exportar.")
    
    if st.session_state['df_resultado'] is not None:
        st.subheader(f"Muestra: {st.session_state['nombre_muestra_actual']}")
        st.dataframe(st.session_state['df_resultado'], use_container_width=True)
        
        csv_bytes = st.session_state['df_resultado'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar planilla CSV (para InfoStat)",
            data=csv_bytes,
            file_name=f"{st.session_state['nombre_muestra_actual']}_resumen.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Cargue y procese una faja en la pestaña **Carga de Faja** para visualizar los datos.")


# ==========================================
# 3. PESTAÑA: GRÁFICO DE RESISTENCIA (📈)
# ==========================================
with tab_graficos:
    st.header("📈 Gráfico de Resistencia Mecánica")
    st.markdown("Perfil de resistencia a la penetración expresado en Megapascals (MPa).")
    
    if st.session_state['df_resultado'] is not None:
        df_g = st.session_state['df_resultado']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_g['Intervalo'], df_g['Resistencia_Media_MPa'], color='sienna', alpha=0.8)
        ax.set_xlabel('Resistencia Media a la Penetración (MPa)')
        ax.set_ylabel('Intervalo de Profundidad (cm)')
        ax.set_title(f'Perfil de Resistencia Mecánica - {st.session_state["nombre_muestra_actual"]}')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
    else:
        st.info("ℹ️ Procese una faja en la pestaña **Carga de Faja** para habilitar el gráfico.")


# ==========================================
# 4. PESTAÑA: BIBLIOGRAFÍA Y MANUAL (📚)
# ==========================================
with tab_manual:
    st.header("📚 Biblioteca Técnica y Manual de Uso")
    st.markdown("Guía operativa e información de referencia provista para el procesamiento individual de fajas.")
    
    with st.expander("📖 Manual de Uso Paso a Paso", expanded=True):
        st.markdown("""
        1. **Preparación:** Registre la faja de papel junto a una escala o regla milimetrada.
        2. **Digitalización:** Tome una fotografía nítida o escanee la faja individualmente y asigne un nombre claro al archivo de imagen (ej: `Muestra_01.jpg`).
        3. **Configuración:** Ingrese los parámetros físicos de su equipo en el panel lateral izquierdo.
        4. **Carga:** Ingrese a la pestaña **📁 Carga de Faja**, cargue la imagen y presione **Procesar Faja**.
        5. **Análisis y Exportación:** Inspeccione la planilla y el gráfico generado, y descargue el archivo CSV para su análisis estadístico en InfoStat.
        """)
        
    with st.expander("🔬 Fundamento Físico y Bibliografía"):
        st.markdown("""
        * **Fórmula de cálculo dinámico:**
          $$q_d = \\frac{M \\cdot g \\cdot H}{A \\cdot e}$$
        * **Referencias:**
          * ASAE S313.3: Soil Cone Penetrometer.
          * Ensayos de resistencia mecánica e investigación de suelos en INTA EEA Corrientes.
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #555;'>"
        "Desarrollado institucionalmente para el INTA EEA Corrientes.<br>"
        "Contacto: <b>Ing. Agr. (Mgter.) Federico A. Paredes</b> (paredesfedericoantonio@gmail.com)"
        "</div>", 
        unsafe_allow_html=True
    )
