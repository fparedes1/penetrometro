import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import zipfile
import io

# Configuración de la página y diseño
st.set_page_config(
    page_title="Procesador de Fajas - INTA EEA Corrientes", 
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
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Automatización de Fajas de Penetrómetro de Golpes")
st.markdown("Herramienta de procesamiento digital para resistencia mecánica a la penetración de suelos.")

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

# --- Creación de Pestañas Principales con sus respectivos íconos ---
tab_carga, tab_datos, tab_graficos, tab_manual = st.tabs([
    "📁 Carga de Fajas", 
    "📊 Planillas y Datos", 
    "📈 Gráficos de Resistencia", 
    "📚 Biblioteca y Manual"
])

# Variables de sesión para persistencia entre pestañas
if 'datos_por_muestra' not in st.session_state:
    st.session_state['datos_por_muestra'] = {}
if 'zip_buffer' not in st.session_state:
    st.session_state['zip_buffer'] = None


# ==========================================
# 1. PESTAÑA: CARGA DE FAJAS (📁)
# ==========================================
with tab_carga:
    st.header("📂 Carga Masiva de Imágenes de Fajas")
    st.markdown("Arrastre o seleccione las fajas escaneadas en formato imagen. Asegúrese de que el nombre de la muestra esté escrito en la parte superior.")
    
    uploaded_files = st.file_uploader(
        "Seleccione las imágenes (Formatos: JPG, PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Se cargaron {len(uploaded_files)} archivos correctamente.")
        
        if st.button("🚀 Procesar Lote de Fajas", type="primary"):
            zip_buffer = io.BytesIO()
            resultados_globales = []
            st.session_state['datos_por_muestra'] = {}

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for uploaded_file in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue

                    # OCR para nombre
                    h_img, w_img = img.shape
                    zona_superior = img[0:int(h_img * 0.15), 0:w_img]
                    _, thresh_ocr = cv2.threshold(zona_superior, 150, 255, cv2.THRESH_BINARY)
                    nombre_detectado = pytesseract.image_to_string(thresh_ocr, config='--psm 7').strip()
                    
                    nombre_muestra = "".join([c for c in nombre_detectado if c.isalnum() or c in ('_', '-')])
                    if not nombre_muestra:
                        nombre_muestra = uploaded_file.name.split('.')[0]

                    # Procesamiento físico
                    blur = cv2.GaussianBlur(img, (5, 5), 0)
                    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                    perfil_proyeccion = np.sum(thresh, axis=1)
                    
                    from scipy.signal import find_peaks
                    picos, _ = find_peaks(perfil_proyeccion, distance=10, prominence=500)
                    
                    profundidades_cm = picos * escala_cm_px
                    profundidades_cm = np.sort(profundidades_cm)

                    if len(profundidades_cm) < 2:
                        continue

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

                    # Guardar en ZIP
                    csv_data = df_intervalos.to_csv(index=False).encode('utf-8')
                    zip_file.writestr(f"{nombre_muestra}_resumen.csv", csv_data)

                    resultados_globales.append(df_intervalos)
                    st.session_state['datos_por_muestra'][nombre_muestra] = df_intervalos

                if resultados_globales:
                    df_total = pd.concat(resultados_globales, ignore_index=True)
                    zip_file.writestr("consolidado_todas_las_muestras.csv", df_total.to_csv(index=False).encode('utf-8'))

            zip_buffer.seek(0)
            st.session_state['zip_buffer'] = zip_buffer
            st.success("🎉 ¡Procesamiento completado con éxito! Diríjase a las pestañas de Datos o Gráficos para inspeccionar los resultados.")


# ==========================================
# 2. PESTAÑA: PLANILLAS Y DATOS (📊)
# ==========================================
with tab_datos:
    st.header("📊 Salida de Datos para Análisis Estadístico")
    st.markdown("Consulte los valores calculados por intervalo de profundidad, listos para ser exportados e integrados en software estadístico (como InfoStat).")
    
    if st.session_state['datos_por_muestra']:
        muestras = list(st.session_state['datos_por_muestra'].keys())
        muestra_elegida = st.selectbox("Seleccione la muestra para ver la planilla:", muestras, key="sel_datos")
        
        df_actual = st.session_state['datos_por_muestra'][muestra_elegida]
        st.dataframe(df_actual, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📥 Descarga Consolidada")
        if st.session_state['zip_buffer']:
            st.download_button(
                label="📥 Descargar paquete comprimido (.zip con todos los CSV y Gráficos)",
                data=st.session_state['zip_buffer'],
                file_name="resultados_penetrometro.zip",
                mime="application/zip"
            )
    else:
        st.info("ℹ️ Aún no hay datos procesados. Por favor, cargue y procese las fajas en la pestaña **Carga de Fajas**.")


# ==========================================
# 3. PESTAÑA: GRÁFICOS DE RESISTENCIA (📈)
# ==========================================
with tab_graficos:
    st.header("📈 Visualización Gráfica de Resistencia Mecánica")
    st.markdown("Curvas y perfiles de resistencia a la penetración expresados en Megapascals (MPa).")
    
    if st.session_state['datos_por_muestra']:
        muestras_g = list(st.session_state['datos_por_muestra'].keys())
        muestra_g_elegida = st.selectbox("Seleccione la muestra a graficar:", muestras_g, key="sel_graf")
        
        df_g = st.session_state['datos_por_muestra'][muestra_g_elegida]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_g['Intervalo'], df_g['Resistencia_Media_MPa'], color='sienna', alpha=0.8)
        ax.set_xlabel('Resistencia Media a la Penetración (MPa)')
        ax.set_ylabel('Intervalo de Profundidad (cm)')
        ax.set_title(f'Perfil de Resistencia Mecánica - {muestra_g_elegida}')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
    else:
        st.info("ℹ️ Cargue las imágenes en la pestaña **Carga de Fajas** para habilitar los gráficos.")


# ==========================================
# 4. PESTAÑA: BIBLIOGRAFÍA Y MANUAL (📚)
# ==========================================
with tab_manual:
    st.header("📚 Biblioteca Técnica y Manual de Uso")
    st.markdown("Guía operativa e información de referencia provista por el desarrollador para el correcto uso del sistema.")
    
    with st.expander("📖 Manual de Uso Paso a Paso", expanded=True):
        st.markdown("""
        1. **Preparación a campo:** Escriba claramente el nombre de la muestra en el margen superior de la faja de papel antes de iniciar la lectura.
        2. **Digitalización:** Escanee o fotografíe las fajas de papel en formato JPG o PNG asegurando buena iluminación y nitidez.
        3. **Configuración:** Ingrese los parámetros físicos de su equipo (Masa, Altura de caída, Diámetro del cono y Escala de calibración) en el panel lateral izquierdo.
        4. **Carga masiva:** Ingrese a la pestaña **📁 Carga de Fajas**, arrastre todas las imágenes y presione **Procesar Lote de Fajas**.
        5. **Análisis:** Revise los resultados en las pestañas de **📊 Planillas y Datos** y **📈 Gráficos de Resistencia**.
        6. **Exportación:** Descargue el paquete comprimido `.zip` con todos los archivos CSV listos para análisis estadístico en InfoStat.
        """)
        
    with st.expander("🔬 Fundamento Físico y Bibliografía Recomendada"):
        st.markdown("""
        * **Fórmula de cálculo dinámico:** La resistencia mecánica ($q_d$) se calcula a partir de la energía cinética del impacto y la sección transversal del cono:
          $$q_d = \\frac{M \\cdot g \\cdot H}{A \\cdot e}$$
        * **Referencias bibliográficas sugeridas:**
          * ASAE S313.3: Soil Cone Penetrometer. American Society of Agricultural Engineers.
          * Restricción física al crecimiento radical y su relación con la resistencia mecánica en suelos agrícolas (Investigaciones INTA).
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #555;'>"
        "Desarrollado institucionalmente para el INTA EEA Corrientes.<br>"
        "Contacto: <b>Ing. Agr. (Mgter.) Federico A. Paredes</b> (paredesfedericoantonio@gmail.com)"
        "</div>", 
        unsafe_allow_html=True
    )
B. Archivo requirements.txt (Librerías de Python)
Crea un segundo archivo de texto llamado exactamente requirements.txt, pega esto dentro y guárdalo:

Plaintext
streamlit
opencv-python-headless
numpy
pandas
matplotlib
scipy
pytesseract