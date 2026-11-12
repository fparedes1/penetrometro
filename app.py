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

# --- Panel Lateral de Parámetros del Instrumento y Configuración ---
st.sidebar.header("⚙️ Parámetros del Instrumento")
masa_kg = st.sidebar.number_input("Peso / Masa (F en kg)", value=2.088, min_value=0.1, step=0.001, format="%.3f")
carrera_m = st.sidebar.number_input("Carrera / Recorrido (E en m)", value=0.565, min_value=0.01, step=0.001, format="%.3f")
base_cono_cm = st.sidebar.number_input("Base del cono (cm)", value=2.0, min_value=0.1, step=0.1)
altura_cono_cm = st.sidebar.number_input("Altura del cono (cm)", value=3.0, min_value=0.1, step=0.1)

constante_aparato = masa_kg * carrera_m
st.sidebar.info(f"📌 Constante del aparato (F × E): **{constante_aparato:.4f} kg·m**")

st.sidebar.markdown("---")
st.sidebar.header("📏 Intervalos de Profundidad")
intervalos_input = st.sidebar.text_input(
    "Definir intervalos (cm)", 
    value="0-10, 10-20, 20-30",
    help="Ingrese los intervalos separados por comas. Define automáticamente si el ensayo es a 30 cm, 50 cm u otra profundidad."
)

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

# --- Pestañas Principales ---
tab_carga, tab_datos, tab_registro, tab_perfil, tab_graficos, tab_manual = st.tabs([
    "📸 Carga de imágenes", 
    "📊 Planillas resumen", 
    "🖼️ Registro visual", 
    "📉 Perfil de RMP",
    "📈 Gráfico Comparativo", 
    "📚 Instructivo de uso"
])

if 'df_consolidado' not in st.session_state:
    st.session_state['df_consolidado'] = None
if 'df_golpes_todos' not in st.session_state:
    st.session_state['df_golpes_todos'] = None
if 'imagen_registro_bytes' not in st.session_state:
    st.session_state['imagen_registro_bytes'] = None


# ==========================================
# 1. PESTAÑA: CARGA DE IMÁGENES
# ==========================================
with tab_carga:
    st.header("📸 Captura y Procesamiento Fotográfico")
    st.markdown("Cargue las fotografías de las fajas de campo. El sistema detecta automáticamente la profundidad máxima según los intervalos configurados.")
    
    uploaded_files = st.file_uploader(
        "Seleccionar imágenes de fajas (JPG, PNG)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Se cargaron {len(uploaded_files)} imagen(es) para procesar.")
        
        if st.button("🚀 Procesar Imágenes", type="primary"):
            # Parsear los intervalos y determinar dinámicamente la profundidad máxima
            limites_intervalos = []
            labels_intervalos = []
            max_profundidad_ensayo = 30.0
            
            try:
                for parte in intervalos_input.split(','):
                    partes_nums = parte.strip().split('-')
                    if len(partes_nums) == 2:
                        z_inf = float(partes_nums[0].strip())
                        z_sup = float(partes_nums[1].strip())
                        nombre_int = f"{int(z_inf)}-{int(z_sup) if not z_sup.is_integer() else int(z_sup)} cm" if z_inf.is_integer() else f"{z_inf}-{z_sup} cm"
                        limites_intervalos.append((z_inf, z_sup, nombre_int))
                        labels_intervalos.append(nombre_int)
                if limites_intervalos:
                    max_profundidad_ensayo = max(sup for inf, sup, name in limites_intervalos)
            except Exception:
                limites_intervalos = [(0.0, 10.0, "0-10 cm"), (10.0, 20.0, "10-20 cm"), (20.0, 30.0, "20-30 cm")]
                labels_intervalos = ["0-10 cm", "10-20 cm", "20-30 cm"]
                max_profundidad_ensayo = 30.0
            
            if not limites_intervalos:
                limites_intervalos = [(0.0, 10.0, "0-10 cm"), (10.0, 20.0, "10-20 cm"), (20.0, 30.0, "20-30 cm")]
                labels_intervalos = ["0-10 cm", "10-20 cm", "20-30 cm"]
                max_profundidad_ensayo = 30.0

            lista_resultados = []
            lista_golpes_detallados = []
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
                        
                        # Escala adaptada exactamente a la profundidad máxima configurada
                        escala_cm_px = max_profundidad_ensayo / float(distancia_px) if distancia_px > 0 else (max_profundidad_ensayo / 1000.0)
                        posiciones_cm = (picos_x - x_inicio) * escala_cm_px
                        
                        prof_iniciales = posiciones_cm[:-1]
                        prof_finales = posiciones_cm[1:]
                        
                        g = 9.81
                        energia_golpe = masa_kg * g * carrera_m
                        
                        delta_e_detalles = np.diff(posiciones_cm) / 100.0
                        delta_e_detalles = np.where(delta_e_detalles <= 0, 0.001, delta_e_detalles)
                        res_detalles = (energia_golpe / (area_cono_m2 * delta_e_detalles)) / 1e6
                        
                        res_primer_golpe = res_detalles[0] if len(res_detalles) > 0 else 0.0
                        df_golpes_indiv = pd.DataFrame({
                            'Muestra': [nombre_muestra],
                            'Profundidad_Final_cm': [0.0],
                            'Resistencia_MPa': [res_primer_golpe]
                        })
                        
                        df_golpes_resto = pd.DataFrame({
                            'Muestra': nombre_muestra,
                            'Profundidad_Final_cm': posiciones_cm[1:],
                            'Resistencia_MPa': res_detalles
                        })
                        
                        df_golpes_indiv = pd.concat([df_golpes_indiv, df_golpes_resto], ignore_index=True)
                        lista_golpes_detallados.append(df_golpes_indiv)
                        
                        registros_intervalo = []
                        for z_inf, z_sup, nombre_int in limites_intervalos:
                            suma_resist_parcial = 0.0
                            suma_fraccion_golpes = 0.0
                            
                            for z_i, z_f in zip(prof_iniciales, prof_finales):
                                overlap_ini = max(z_i, z_inf)
                                overlap_fin = min(z_f, z_sup)
                                
                                if overlap_fin > overlap_ini:
                                    delta_e_overlap = (overlap_fin - overlap_ini) / 100.0
                                    delta_e_total = max(0.0001, (z_f - z_i) / 100.0)
                                    
                                    if delta_e_overlap > 0:
                                        resistencia_golpe_mpa = (energia_golpe / (area_cono_m2 * delta_e_total)) / 1e6
                                        fraccion = (overlap_fin - overlap_ini) / (z_f - z_i)
                                        
                                        suma_resist_parcial += resistencia_golpe_mpa * fraccion
                                        suma_fraccion_golpes += fraccion
                            
                            res_media = suma_resist_parcial / suma_fraccion_golpes if suma_fraccion_golpes > 0 else 0.0
                                
                            registros_intervalo.append({
                                'Muestra': nombre_muestra,
                                'Intervalo': nombre_int,
                                'Resistencia_Media_MPa': res_media,
                                'Total_Golpes': round(suma_fraccion_golpes, 2),
                                'Resistencia_Max_MPa': res_media
                            })
                        
                        df_intervalos = pd.DataFrame(registros_intervalo)
                        lista_resultados.append(df_intervalos)
                
                barra_progreso.progress((i + 1) / len(uploaded_files))
            
            if lista_resultados and imagenes_para_stitch:
                st.session_state['df_consolidado'] = pd.concat(lista_resultados, ignore_index=True)
                st.session_state['df_golpes_todos'] = pd.concat(lista_golpes_detallados, ignore_index=True)
                st.session_state['max_prof_actual'] = max_profundidad_ensayo
                st.session_state['labels_actuales'] = labels_intervalos
                
                imagen_final_combinada = cv2.vconcat(imagenes_para_stitch)
                _, buffer = cv2.imencode('.jpg', imagen_final_combinada)
                st.session_state['imagen_registro_bytes'] = buffer.tobytes()
                
                st.success(f"🎉 ¡Procesamiento completado hasta {int(max_profundidad_ensayo)} cm! Revise las planillas y gráficos.")
            else:
                st.error("No se pudieron detectar marcas. Ajuste la sensibilidad en el panel lateral.")


# ==========================================
# 2. PESTAÑA: PLANILLAS RESUMEN
# ==========================================
with tab_datos:
    st.header("📊 Planillas resumen")
    st.markdown("Resumen calculado por muestra y los intervalos de profundidad configurados con prorrateo proporcional.")
    
    if st.session_state['df_consolidado'] is not None:
        st.dataframe(st.session_state['df_consolidado'], use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state['df_consolidado'].to_excel(writer, index=False, sheet_name='Resistencia_Suelos')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 Descargar planilla en formato Excel (.xlsx)",
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
# 4. PESTAÑA: PERFIL DE RMP
# ==========================================
with tab_perfil:
    st.header("📉 Perfil de Resistencia Mecánica a la Penetración (RMP)")
    st.markdown("Gráfico de la resistencia mecánica a la penetración de las raíces en función de la profundidad.")
    
    if st.session_state['df_golpes_todos'] is not None:
        fig_perfil, ax_perfil = plt.subplots(figsize=(10, 8))
        df_golpes = st.session_state['df_golpes_todos']
        max_p = st.session_state.get('max_prof_actual', 30.0)
        
        for muestra in df_golpes['Muestra'].unique():
            df_m = df_golpes[df_golpes['Muestra'] == muestra].sort_values('Profundidad_Final_cm')
            ax_perfil.plot(df_m['Resistencia_MPa'], df_m['Profundidad_Final_cm'], marker='o', linestyle='-', label=muestra, linewidth=2)
            
        ax_perfil.set_xlabel('Resistencia Mecánica (MPa)', fontsize=12)
        ax_perfil.set_ylabel('Profundidades (cm)', fontsize=12)
        ax_perfil.set_title(f'Perfil Continuo de RMP por Muestra (hasta {int(max_p)} cm)', fontsize=14, fontweight='bold')
        ax_perfil.set_xlim(left=0)
        ax_perfil.set_ylim(bottom=max_p + 2, top=0)  # Límite dinámico ajustado a la profundidad máxima configurada
        ax_perfil.legend(title="Muestras")
        ax_perfil.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig_perfil)
    else:
        st.info("ℹ️ Procese las fotografías en la pestaña **Carga de imágenes** para generar el perfil de RMP.")


# ==========================================
# 5. PESTAÑA: GRÁFICO COMPARATIVO
# ==========================================
with tab_graficos:
    st.header("📈 Gráfico Comparativo de Resistencia")
    st.markdown("RMP por intervalo de profundidad.")
    
    if st.session_state['df_consolidado'] is not None:
        df_c = st.session_state['df_consolidado']
        intervalos = st.session_state.get('labels_actuales', ["0-10 cm", "10-20 cm", "20-30 cm"])
        
        fig, ax = plt.subplots(figsize=(10, 7))
        muestras = df_c['Muestra'].unique()
        
        y = np.arange(len(intervalos))
        height = 0.8 / max(len(muestras), 1)
        
        for i, muestra in enumerate(muestras):
            df_m = df_c[df_c['Muestra'] == muestra]
            x_vals = []
            for interv in intervalos:
                val_row = df_m[df_m['Intervalo'] == interv]
                if not val_row.empty and not pd.isna(val_row['Resistencia_Media_MPa'].values[0]):
                    x_vals.append(val_row['Resistencia_Media_MPa'].values[0])
                else:
                    x_vals.append(0)
            ax.barh(y + i * height, x_vals, height, label=muestra, alpha=0.85)
            
        ax.set_ylabel('Intervalo de Profundidad')
        ax.set_xlabel('Resistencia Mecánica (MPa)')
        ax.set_title('Resistencia Mecánica por Intervalo de profundidad')
        ax.set_yticks(y + height * (len(muestras) - 1) / 2)
        ax.set_yticklabels(intervalos)
        ax.invert_yaxis()
        ax.legend(title="Muestras")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
    else:
        st.info("ℹ️ Procese fotografías para habilitar el gráfico comparativo.")


# ==========================================
# 6. PESTAÑA: INSTRUCTIVO DE USO
# ==========================================
with tab_manual:
    st.header("📚 Instructivo de uso")
    
    with st.expander("📖 Guía de uso de la aplicación", expanded=True):
        st.markdown("""
        * **Carga de imágenes:** Permite seleccionar y cargar múltiples fotografías de fajas de papel en formatos JPG o PNG.
        * **Planillas resumen:** Muestra el cálculo consolidado por intervalos personalizados (configurables en la barra lateral para 30 cm, 50 cm u otros) aplicando prorrateo proporcional, y habilita la descarga directa en Excel (`.xlsx`).
        * **Registro visual:** Presenta una imagen unificada con líneas rojas verticales superpuestas sobre cada golpe detectado, permitiendo auditar visualmente el conteo.
        * **Perfil de RMP:** Muestra el gráfico de la resistencia mecánica a la penetración de las raíces en función de la profundidad según el rango configurado, proyectado suavemente desde la superficie.
        * **Gráfico Comparativo:** Visualiza la RMP por intervalo de profundidad en un gráfico de barras horizontales ordenado de superficie a profundidad respetando exactamente los intervalos definidos.
        """)
        
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #555;'>"
        "Desarrollado institucionalmente para el INTA EEA Corrientes.<br>"
        "Contacto: <b>Ing. Agr. (Mgter.) Federico A. Paredes</b> (paredesfedericoantonio@gmail.com)</div>", 
        unsafe_allow_html=True
    )
