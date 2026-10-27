import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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
st.markdown("Automatización de ensayos, análisis estadístico por intervalos y registro fotográfico consolidado.")

# --- Panel Lateral de Configuración (Parámetros del Instrumento) ---
st.sidebar.header("⚙️ Parámetros del Instrumento")
masa_kg = st.sidebar.number_input("Peso / Masa de la masa (F en kg)", value=2.088, min_value=0.1, step=0.001, format="%.3f")
carrera_m = st.sidebar.number_input("Distancia o carrera de recorrido (E en m)", value=0.565, min_value=0.01, step=0.001, format="%.3f")
base_cono_cm = st.sidebar.number_input("Base del cono (cm)", value=2.0, min_value=0.1, step=0.1)
altura_cono_cm = st.sidebar.number_input("Altura del cono (cm)", value=3.0, min_value=0.1, step=0.1)

constante_aparato = masa_kg * carrera_m
st.sidebar.info(f"📌 Constante del aparato (F × E): **{constante_aparato:.4f} kg·m**")

radio_m = (base_cono_cm / 100.0) / 2.0
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
tab_carga, tab_datos, tab_registro, tab_graficos, tab_manual = st.tabs([
    "📸 Carga de Fotografías", 
    "📊 Planilla Excel Consolidada", 
    "🖼️ Registro Visual", 
    "📈 Gráfico Comparativo", 
    "📚 Manual Operativo"
])

# Variables de sesión para persistencia de datos
if 'df_consolidado' not in st.session_state:
    st.session_state['df_consolidado'] = None
if 'imagen_registro_bytes' not in st.session_state:
    st.session_state['imagen_registro_bytes'] = None


# ==========================================
# 1. PESTAÑA: CARGA DE FOTOGRAFÍAS (📸)
# ==========================================
with tab_carga:
    st.header("📸 Captura y Procesamiento Fotográfico")
    st.markdown("""
    Cargue las **fotografías de las fajas** (puede seleccionar varias a la vez). 
    * *Sugerencia operativa:* Nómbre previamente cada archivo de foto con la identificación de la muestra (ej: `Prueba_1.jpg`). El sistema extraerá ese nombre automáticamente.
    """)
    
    uploaded_files = st.file_uploader(
        "Seleccionar imágenes de fajas (Formatos: JPG, PNG)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Se han cargado {len(uploaded_files)} fotografía(s) listas para procesamiento automático.")
        
        if st.button("🚀 Procesar Fotografías y Generar Planilla/Registro", type="primary"):
            lista_resultados = []
            imagenes_para_stitch = []
            barra_progreso = st.progress(0)
            
            ancho_estandar = 1200
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_bgr is not None:
                    nombre_muestra = uploaded_file.name.rsplit('.', 1)[0]
                    h_img, w_img, _ = img_bgr.shape
                    
                    # Preparar imagen para el registro visual combinando una etiqueta con el nombre
                    ratio = ancho_estandar / float(w_img)
                    alto_estandar = int(h_img * ratio)
                    img_redimensionada = cv2.resize(img_bgr, (ancho_estandar, alto_estandar))
                    
                    altura_etiqueta = 60
                    franja_etiqueta = np.ones((altura_etiqueta, ancho_estandar, 3), dtype=np.uint8) * 255
                    cv2.putText(franja_etiqueta, f"Muestra: {nombre_muestra}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
                    
                    img_con_etiqueta = cv2.vconcat([franja_etiqueta, img_redimensionada])
                    separador = np.ones((15, ancho_estandar, 3), dtype=np.uint8) * 220
                    bloque_muestra = cv2.vconcat([img_con_etiqueta, separador])
                    imagenes_para_stitch.append(bloque_muestra)
                    
                    # 1. Calibración automática mediante la regla verde (40 cm)
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                    mask_green = cv2.inRange(hsv, lower_green, upper_green)
                    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    escala_cm_px = 0.05
                    y_inicio_faja = 0
                    y_fin_faja = int(h_img * 0.35)
                    
                    if contours:
                        c_regla = max(contours, key=cv2.contourArea)
                        x_r, y_r, w_r, h_r = cv2.boundingRect(c_regla)
                        if w_r > w_img * 0.3:
                            escala_cm_px = 40.0 / float(w_r)
                            # Aislar exclusivamente la faja superior y evitar cualquier contacto con la regla verde
                            y_fin_faja = max(10, y_r - 5)
                            y_inicio_faja = max(0, y_r - int(h_r * 1.8))

                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_faja = img_gray[y_inicio_faja:y_fin_faja, 0:w_img]
                    if img_faja.shape[0] == 0:
                        img_faja = img_gray[0:int(h_img*0.35), 0:w_img]

                    # 2. Procesamiento limpio de picos para evitar falsos golpes por ruido
                    blur = cv2.GaussianBlur(img_faja, (5, 5), 0)
                    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
                    perfil_proyeccion = np.sum(thresh, axis=1)
                    
                    # Distancia mínima ampliada para evitar que las marcas cercanas se superpongan
                    picos, _ = find_peaks(perfil_proyeccion, distance=14, prominence=800)
                    profundidades_cm = picos * escala_cm_px
                    profundidades_cm = np.sort(profundidades_cm)

                    if len(profundidades_cm) >= 2:
                        prof_metros = profundidades_cm / 100.0
                        delta_e = np.diff(prof_metros)
                        delta_e = np.where(delta_e == 0, 0.0001, delta_e)

                        g = 9.81
                        energia_golpe = masa_kg * g * carrera_m
                        resistencia_mpa = (energia_golpe / (area_cono_m2 * delta_e)) / 1e6

                        df_golpes = pd.DataFrame({
                            'Muestra': nombre_muestra,
                            'Golpe': range(1, len(delta_e) + 1),
                            'Profundidad_Final_cm': profundidades_cm[1:],
                            'Penetracion_cm': delta_e * 100,
                            'Resistencia_MPa': resistencia_mpa
                        })

                        # Corte estricto hasta los 30 cm reales
                        df_golpes = df_golpes[df_golpes['Profundidad_Final_cm'] <= 30.0]

                        # Agrupación exacta en los tres intervalos requeridos
                        bins = [0, 10, 20, 30]
                        labels = ["0-10 cm", "10-20 cm", "20-30 cm"]
                        df_golpes['Intervalo'] = pd.cut(df_golpes['Profundidad_Final_cm'], bins=bins, labels=labels, include_lowest=True)
                        
                        df_intervalos = df_golpes.groupby(['Muestra', 'Intervalo'], observed=False)['Resistencia_MPa'].agg(
                            ['mean', 'count', 'max']
                        ).reset_index()
                        
                        df_intervalos.rename(columns={
                            'mean': 'Resistencia_Media_MPa', 
                            'count': 'Total_Golpes', 
                            'max': 'Resistencia_Max_MPa'
                        }, inplace=True)
                        
                        lista_resultados.append(df_intervalos)
                
                barra_progreso.progress((i + 1) / len(uploaded_files))
            
            if lista_resultados and imagenes_para_stitch:
                st.session_state['df_consolidado'] = pd.concat(lista_resultados, ignore_index=True)
                
                imagen_final_combinada = cv2.vconcat(imagenes_para_stitch)
                _, buffer = cv2.imencode('.jpg', imagen_final_combinada)
                st.session_state['imagen_registro_bytes'] = buffer.tobytes()
                
                st.success("🎉 ¡Procesamiento completado con éxito! Verifique la planilla y el gráfico.")
            else:
                st.error("No se pudieron procesar las fotografías. Asegúrese de que las fajas estén enfocadas y nítidas.")


# ==========================================
# 2. PESTAÑA: PLANILLA EXCEL CONSOLIDADA (📊)
# ==========================================
with tab_datos:
    st.header("📊 Planilla Excel para Análisis Estadístico")
    st.markdown("Resumen calculado automáticamente por muestra y los intervalos requeridos (**0-10 cm, 10-20 cm y 20-30 cm**).")
    
    if st.session_state['df_consolidado'] is not None:
        st.dataframe(st.session_state['df_consolidado'], use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state['df_consolidado'].to_excel(writer, index=False, sheet_name='Resistencia_Suelos')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 Descargar planilla en formato Excel (.xlsx) para InfoStat",
            data=excel_data,
            file_name="resistencia_suelos_intervalos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("ℹ️ Cargue y procese las fotografías en la pestaña **Carga de Fotografías** para generar la planilla Excel.")


# ==========================================
# 3. PESTAÑA: REGISTRO VISUAL (🖼️)
# ==========================================
with tab_registro:
    st.header("🖼️ Registro Visual Consolidado de Fajas")
    st.markdown("Imagen única que combina secuencialmente todas las fotografías analizadas con sus respectivas etiquetas.")
    
    if st.session_state['imagen_registro_bytes'] is not None:
        st.image(st.session_state['imagen_registro_bytes'], caption="Registro fotográfico unificado de fajas", use_column_width=True)
        
        st.download_button(
            label="📥 Descargar imagen de registro unificada (JPG)",
            data=st.session_state['imagen_registro_bytes'],
            file_name="registro_fotografico_fajas.jpg",
            mime="image/jpeg"
        )
    else:
        st.info("ℹ️ Procese las fotografías en la pestaña **Carga de Fotografías** para habilitar el registro visual.")


# ==========================================
# 4. PESTAÑA: GRÁFICO COMPARATIVO (📈)
# ==========================================
with tab_graficos:
    st.header("📈 Gráfico Comparativo de Resistencia")
    st.markdown("Perfil horizontal de resistencia media a la penetración (ordenado de superficie a profundidad).")
    
    if st.session_state['df_consolidado'] is not None:
        df_c = st.session_state['df_consolidado']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        muestras = df_c['Muestra'].unique()
        intervalos = ["0-10 cm", "10-20 cm", "20-30 cm"] # Orden estricto de superficie a profundidad
        
        y = np.arange(len(intervalos))
        height = 0.8 / max(len(muestras), 1)
        
        for i, muestra in enumerate(muestras):
            df_m = df_c[df_c['Muestra'] == muestra]
            x_vals = [df_m[df_m['Intervalo'] == interv]['Resistencia_Media_MPa'].values[0] 
                      if not df_m[df_m['Intervalo'] == interv].empty else 0 for interv in intervalos]
            ax.barh(y + i * height, x_vals, height, label=muestra, alpha=0.85)
            
        ax.set_ylabel('Intervalo de Profundidad')
        ax.set_xlabel('Resistencia Media a la Penetración (MPa)')
        ax.set_title('Perfil de Resistencia Mecánica por Intervalos y Muestras')
        ax.set_yticks(y + height * (len(muestras) - 1) / 2)
        ax.set_yticklabels(intervalos)
        ax.invert_yaxis() # Invierte el eje Y para que 0-10 cm quede arriba y 20-30 cm abajo
        ax.legend(title="Muestras")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
    else:
        st.info("ℹ️ Procese las fotografías para habilitar el gráfico comparativo.")


# ==========================================
# 5. PESTAÑA: MANUAL OPERATIVO (📚)
# ==========================================
with tab_manual:
    st.header("📚 Manual Operativo Simplificado")
    st.markdown("Instrucciones rápidas para el personal encargado de la toma de registros y procesamiento.")
    
    with st.expander("📖 Guía paso a paso para el operario", expanded=True):
        st.markdown("""
        1. **Configuración:** Verifique en el panel lateral los parámetros físicos del instrumento (Masa F, Carrera E, Base y Altura del cono).
        2. **Identificación:** Renombre cada archivo fotográfico con el nombre de la muestra (ej: `Prueba_1.jpg`).
        3. **Captura:** Tome la foto con la faja apoyada junto a la regla verde de 40 cm.
        4. **Carga y Procesamiento:** Ingrese a **📸 Carga de Fotografías**, suba todas las imágenes y presione el botón de procesamiento.
        5. **Salida y Archivo:** 
           * En **📊 Planilla Excel Consolidada** descargue el archivo `.xlsx` listo para InfoStat.
           * En **📈 Gráfico Comparativo** observe el perfil horizontal ordenado de superficie a profundidad.
        """)
        
    with st.expander("🔬 Fundamento Físico y Referencias"):
        st.markdown("""
        * **Cálculo dinámico aplicado:**
          $$q_d = \\frac{M \\cdot g \\cdot H}{A \\cdot e}$$
        * **Referencias metodológicas:**
          * ASAE S313.3: Soil Cone Penetrometer.
          * Área de Recursos Naturales y Extensión Rural - INTA EEA Corrientes.
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #555;'>"
        "Desarrollado institucionalmente para el INTA EEA Corrientes.<br>"
        "Contacto: <b>Ing. Agr. (Mgter.) Federico A. Paredes</b> (paredesfedericoantonio@gmail.com)"
        "</div>", 
        unsafe_allow_html=True
    )
