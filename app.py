import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io

# Configuración de la página
st.set_page_config(
    page_title="Resistencia mecánica a la penetración de raíces - INTA EEA Corrientes", 
    page_icon="🌾", 
    layout="wide"
)

# Estilos CSS
st.markdown("""
    <style>
    .main { background-color: #f9fbfc; }
    .stSidebar { background-color: #f1f5f9; }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Resistencia mecánica a la penetración de raíces")
st.markdown("Procesamiento de mediciones a campo con penetrómetro de golpes.")

# --- Panel Lateral de Parámetros del Instrumento y Sensibilidad ---
st.sidebar.header("⚙️ Parámetros del Instrumento")
masa_kg = st.sidebar.number_input("Peso / Masa (F en kg)", value=2.088, min_value=0.1, step=0.001, format="%.3f")
carrera_m = st.sidebar.number_input("Carrera / Recorrido (E en m)", value=0.565, min_value=0.01, step=0.001, format="%.3f")
base_cono_cm = st.sidebar.number_input("Base del cono (cm)", value=2.0, min_value=0.1, step=0.1)
altura_cono_cm = st.sidebar.number_input("Altura del cono (cm)", value=3.0, min_value=0.1, step=0.1)

constante_aparato = masa_kg * carrera_m
st.sidebar.info(f"📌 Constante del aparato (F × E): **{constante_aparato:.4f} kg·m**")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Calibración de Detección")
sensibilidad_picos = st.sidebar.slider(
    "Sensibilidad de detección (Prominencia)", 
    min_value=0.02, max_value=0.40, value=0.10, step=0.02,
    help="Baje este valor si faltan golpes por detectar; súbalo si detecta marcas falsas o arrugas."
)
distancia_minima_px = st.sidebar.slider(
    "Distancia mínima entre golpes (px)", 
    min_value=15, max_value=60, value=25, step=5,
    help="Controla la separación mínima horizontal esperada entre cada golpe."
)

radio_m = (base_cono_cm / 100.0) / 2.0
area_cono_m2 = np.pi * (radio_m ** 2)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 0.8em; color: gray; text-align: center;'>"
    "Desarrollado por:<br><b>Ing. Agr. (Mgter.) Federico A. Paredes</b><br>"
    "<a href='mailto:paredesfedericoantonio@gmail.com'>paredesfedericoantonio@gmail.com</a></div>", 
    unsafe_allow_html=True
)

# --- Pestañas Principales (con los nuevos nombres solicitados) ---
tab_carga, tab_datos, tab_registro, tab_graficos, tab_manual = st.tabs([
    "📸 Carga de imágenes", 
    "📊 Planillas resumen", 
    "🖼️ Registro visual", 
    "📈 Gráfico Comparativo", 
    "📚 Manual Operativo"
])

if 'df_consolidado' not in st.session_state:
    st.session_state['df_consolidado'] = None
if 'imagen_registro_bytes' not in st.session_state:
    st.session_state['imagen_registro_bytes'] = None


# ==========================================
# 1. PESTAÑA: CARGA DE IMÁGENES
# ==========================================
with tab_carga:
    st.header("📸 Captura y Procesamiento Fotográfico")
    st.markdown("Cargue las fotografías de las fajas. Si nota que algún golpe no se detecta correctamente, ajuste la **Sensibilidad de detección** en la barra lateral izquierda.")
    
    uploaded_files = st.file_uploader(
        "Seleccionar imágenes de fajas (JPG, PNG)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Se cargaron {len(uploaded_files)} imagen(es) para procesar.")
        
        if st.button("🚀 Procesar Fajas Automáticamente", type="primary"):
            lista_resultados = []
            imagenes_para_stitch = []
            barra_progreso = st.progress(0)
            
            ancho_estandar = 1400
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_bgr is not None:
                    nombre_muestra = uploaded_file.name.rsplit('.', 1)[0]
                    h_img, w_img, _ = img_bgr.shape
                    
                    ratio = ancho_estandar / float(w_img)
                    alto_estandar = int(h_img * ratio)
                    img_res = cv2.resize(img_bgr, (ancho_estandar, alto_estandar))
                    
                    gray_full = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                    _, thresh_paper = cv2.threshold(gray_full, 190, 255, cv2.THRESH_BINARY)
                    row_sums = np.sum(thresh_paper, axis=1)
                    valid_rows = np.where(row_sums > (ancho_estandar * 0.4))[0]
                    
                    if len(valid_rows) > 10:
                        y_min, y_max = valid_rows[0], valid_rows[-1]
                        y_min = max(0, y_min - 15)
                        y_max = min(alto_estandar, y_max + 15)
                    else:
                        y_min, y_max = int(alto_estandar * 0.2), int(alto_estandar * 0.8)
                    
                    faja_recortada = gray_full[y_min:y_max, :]
                    
                    blur = cv2.GaussianBlur(faja_recortada, (3, 3), 0)
                    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
                    
                    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int((y_max - y_min) * 0.4))))
                    lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
                    
                    perfil_x = np.sum(lineas_v, axis=0)
                    
                    picos_x, _ = find_peaks(
                        perfil_x, 
                        distance=distancia_minima_px, 
                        prominence=np.max(perfil_x) * sensibilidad_picos
                    )
                    
                    img_vis = img_res.copy()
                    for px in picos_x:
                        cv2.line(img_vis, (px, 0), (px, alto_estandar), (0, 0, 255), 2)
                    
                    altura_etiqueta = 50
                    franja_etiqueta = np.ones((altura_etiqueta, ancho_estandar, 3), dtype=np.uint8) * 255
                    cv2.putText(franja_etiqueta, f"Muestra: {nombre_muestra} | Golpes detectados: {len(picos_x)}", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
                    
                    bloque_muestra = cv2.vconcat([franja_etiqueta, img_vis, np.ones((15, ancho_estandar, 3), dtype=np.uint8)*220])
                    imagenes_para_stitch.append(bloque_muestra)

                    if len(picos_x) >= 2:
                        x_inicio = picos_x[0]
                        x_fin = picos_x[-1]
                        distancia_px = x_fin - x_inicio
                        
                        escala_cm_px = 30.0 / float(distancia_px) if distancia_px > 0 else 0.05
                        posiciones_cm = (picos_x - x_inicio) * escala_cm_px
                        
                        prof_metros = posiciones_cm / 100.0
                        delta_e = np.diff(prof_metros)
                        delta_e = np.where(delta_e <= 0, 0.001, delta_e)

                        g = 9.81
                        energia_golpe = masa_kg * g * carrera_m
                        resistencia_mpa = (energia_golpe / (area_cono_m2 * delta_e)) / 1e6

                        df_golpes = pd.DataFrame({
                            'Muestra': nombre_muestra,
                            'Golpe': range(1, len(delta_e) + 1),
                            'Profundidad_Final_cm': posiciones_cm[1:],
                            'Penetracion_cm': delta_e * 100,
                            'Resistencia_MPa': resistencia_mpa
                        })

                        df_golpes = df_golpes[df_golpes['Profundidad_Final_cm'] <= 30.0]

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
                
                st.success("🎉 ¡Procesamiento completado! Revise la pestaña Registro visual para confirmar que las líneas rojas coinciden con cada golpe.")
            else:
                st.error("No se pudieron detectar marcas. Ajuste la sensibilidad en el panel lateral.")


# ==========================================
# 2. PESTAÑA: PLANILLAS RESUMEN
# ==========================================
with tab_datos:
    st.header("📊 Planillas resumen para análisis estadístico")
    st.markdown("Resumen calculado por muestra y los intervalos de profundidad (**0-10 cm, 10-20 cm y 20-30 cm**).")
    
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
        st.info("ℹ️ Cargue y procese fotografías para visualizar la planilla.")


# ==========================================
# 3. PESTAÑA: REGISTRO VISUAL
# ==========================================
with tab_registro:
    st.header("🖼️ Registro visual de la detección de golpes")
    st.markdown("Muestra las líneas rojas verticales sobre cada marca detectada para auditar visualmente el conteo.")
    
    if st.session_state['imagen_registro_bytes'] is not None:
        st.image(st.session_state['imagen_registro_bytes'], caption="Verificación fotográfica de golpes detectados (Líneas rojas)", use_column_width=True)
        
        st.download_button(
            label="📥 Descargar registro visual unificado (JPG)",
            data=st.session_state['imagen_registro_bytes'],
            file_name="registro_fotografico_fajas.jpg",
            mime="image/jpeg"
        )
    else:
        st.info("ℹ️ Procese fotografías para habilitar el registro visual.")


# ==========================================
# 4. PESTAÑA: GRÁFICO COMPARATIVO
# ==========================================
with tab_graficos:
    st.header("📈 Gráfico Comparativo de Resistencia")
    st.markdown("Perfil horizontal de resistencia media a la penetración (de superficie a profundidad).")
    
    if st.session_state['df_consolidado'] is not None:
        df_c = st.session_state['df_consolidado']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        muestras = df_c['Muestra'].unique()
        intervalos = ["0-10 cm", "10-20 cm", "20-30 cm"]
        
        y = np.arange(len(intervalos))
        height = 0.8 / max(len(muestras), 1)
        
        for i, muestra in enumerate(muestras):
            df_m = df_c[df_c['Muestra'] == muestra]
            x_vals = [df_m[df_m['Intervalo'] == interv]['Resistencia_Media_MPa'].values[0] 
                      if not df_m[df_m['Intervalo'] == interv].empty and not pd.isna(df_m[df_m['Intervalo'] == interv]['Resistencia_Media_MPa'].values[0]) else 0 for interv in intervalos]
            ax.barh(y + i * height, x_vals, height, label=muestra, alpha=0.85)
            
        ax.set_ylabel('Intervalo de Profundidad')
        ax.set_xlabel('Resistencia Media a la Penetración (MPa)')
        ax.set_title('Perfil de Resistencia Mecánica por Intervalos y Muestras')
        ax.set_yticks(y + height * (len(muestras) - 1) / 2)
        ax.set_yticklabels(intervalos)
        ax.invert_yaxis()
        ax.legend(title="Muestras")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
    else:
        st.info("ℹ️ Procese fotografías para habilitar el gráfico comparativo.")


# ==========================================
# 5. PESTAÑA: MANUAL OPERATIVO
# ==========================================
with tab_manual:
    st.header("📚 Manual Operativo Simplificado")
    
    with st.expander("📖 Guía de uso y calibración de sensibilidad", expanded=True):
        st.markdown("""
        1. **Configuración:** Ajuste los parámetros físicos del instrumento en la barra lateral.
        2. **Control de Sensibilidad:** Si al procesar nota que faltan marcas o se detectan de más, mueva el deslizador de **Sensibilidad de detección** en la barra lateral.
        3. **Auditoría:** En **🖼️ Registro visual**, compruebe que las líneas rojas coincidan exactamente con cada marca de la faja.
        4. **Exportación:** Descargue el Excel `.xlsx` listo para InfoStat.
        """)
        
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #555;'>"
        "Desarrollado por.<br>"
        "Contacto: <b>Ing. Agr. (Mgter.) Federico A. Paredes</b> (paredesfedericoantonio@gmail.com)</div>", 
        unsafe_allow_html=True
    )
