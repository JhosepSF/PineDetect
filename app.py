"""PineDetect - Aplicación Web Principal (Streamlit).

Detección y clasificación inteligente del estado de madurez de piñas
utilizando Ultralytics YOLOv8.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import streamlit as st

from src.config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_DESCRIPTION,
    DISCLAIMER_TEXT,
    PRIVACY_NOTICE_TEXT,
    MODEL_PATH,
    MODEL_FILENAME,
    CLASS_DISPLAY_NAMES,
    CLASS_COLORS_HEX,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    CONFIDENCE_STEP,
    IOU_MIN,
    IOU_MAX,
    IOU_STEP,
    ASSETS_DIR,
)
from src.model_loader import load_model, check_model_exists, get_device_info
from src.validators import validate_image_file, ImageValidationResult
from src.inference import run_pineapple_detection, Detection, InferenceSummary
from src.visualization import (
    draw_detections_on_image,
    create_class_distribution_chart,
)
from src.exporters import (
    export_image_to_png_bytes,
    export_detections_to_csv_bytes,
    export_summary_to_json_bytes,
)

# Configuración del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pinedetect")

# Configuración inicial de la página
st.set_page_config(
    page_title="PineDetect — Detección de Madurez de Piñas",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css() -> None:
    """Carga y aplica los estilos CSS personalizados desde assets/styles.css."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.is_file():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as e:
            logger.warning("No se pudo cargar styles.css: %s", str(e))


def render_header() -> None:
    """Renderiza el banner superior de bienvenida con diseño institucional."""
    st.markdown(
        f"""
        <div class="pine-hero-container">
            <div class="pine-hero-badge">🍍 Visión por Computador con YOLOv8</div>
            <h1 class="pine-hero-title">{APP_NAME}</h1>
            <div class="pine-hero-subtitle">{APP_SUBTITLE}</div>
            <p class="pine-hero-desc">{APP_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar(model_available: bool, device_info: str) -> dict:
    """Renderiza la barra lateral con controles, estado del modelo y leyendas."""
    with st.sidebar:
        # Marca e icono en la barra lateral
        logo_path = ASSETS_DIR / "logo_placeholder.png"
        if logo_path.is_file():
            st.image(str(logo_path), width=72)

        st.markdown(
            f"""
            <div class="sidebar-header-box">
                <div class="sidebar-brand-title">🍍 {APP_NAME}</div>
                <div class="sidebar-brand-badge">Sistema de Clasificación Agrícola</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Estado del Modelo
        st.subheader("Estado del Sistema")
        if model_available:
            st.markdown(
                """
                <div class="model-status-badge ok">
                    <span>🟢</span> <b>Modelo cargado correctamente</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="model-status-badge missing">
                    <span>🟠</span> <b>Modelo no disponible</b><br>
                    <small style="font-weight:normal;">Coloca '{MODEL_FILENAME}' en la carpeta 'models/'.</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Parámetros de Inferencia
        st.subheader("Parámetros de Detección")
        conf_thresh = st.slider(
            "Confianza mínima",
            min_value=CONFIDENCE_MIN,
            max_value=CONFIDENCE_MAX,
            value=DEFAULT_CONFIDENCE,
            step=CONFIDENCE_STEP,
            help="Umbral mínimo de certeza del modelo para considerar una detección válida."
        )

        iou_thresh = st.slider(
            "Umbral de IoU (NMS)",
            min_value=IOU_MIN,
            max_value=IOU_MAX,
            value=DEFAULT_IOU,
            step=IOU_STEP,
            help="Superposición máxima permitida entre cajas de una misma piña (Non-Maximum Suppression)."
        )

        # Leyenda de Clases
        st.subheader("Leyenda de Madurez")
        st.markdown(
            f"""
            <div class="pine-class-pill inmadura">
                <span><span class="pine-pill-indicator inmadura"></span> Inmadura</span>
                <small>Verde</small>
            </div>
            <div class="pine-class-pill madura">
                <span><span class="pine-pill-indicator madura"></span> Madura</span>
                <small>Amarillo / Naranja</small>
            </div>
            <div class="pine-class-pill sobremadura">
                <span><span class="pine-pill-indicator sobremadura"></span> Sobremadura</span>
                <small>Rojo</small>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Información Técnica
        st.markdown(
            f"""
            <div class="sidebar-info-card">
                <b>Especificaciones Técnicas</b><br>
                • <b>Arquitectura:</b> YOLOv8m<br>
                • <b>Tamaño de entrada:</b> {DEFAULT_IMGSZ} px<br>
                • <b>Salida:</b> Cajas y clasificación<br>
                • <b>Dispositivo:</b> {device_info}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Aviso / Disclaimer
        st.markdown(
            f"""
            <div class="sidebar-disclaimer-box">
                ℹ️ <b>Aviso:</b> {DISCLAIMER_TEXT}
            </div>
            """,
            unsafe_allow_html=True
        )

    return {
        "conf": conf_thresh,
        "iou": iou_thresh,
        "imgsz": DEFAULT_IMGSZ
    }


def render_initial_guide() -> None:
    """Muestra la guía inicial de uso cuando no se ha cargado una imagen."""
    st.markdown("### 📋 Recomendaciones para el Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="pine-card">
                <div class="pine-card-title">💡 1. Buena Iluminación</div>
                <p style="color: #475569; font-size: 0.9rem; margin: 0;">
                    Usa fotos con luz natural o ambiente uniforme para resaltar los tonos de la cáscara.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="pine-card">
                <div class="pine-card-title">📸 2. Visibilidad y Enfoque</div>
                <p style="color: #475569; font-size: 0.9rem; margin: 0;">
                    Procura que la piña sea claramente visible y evita desenfoques o movimientos bruscos.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="pine-card">
                <div class="pine-card-title">🍍 3. Múltiples Frutos</div>
                <p style="color: #475569; font-size: 0.9rem; margin: 0;">
                    El modelo puede identificar y clasificar múltiples piñas en una sola toma fotográfica.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main() -> None:
    """Función principal de la aplicación PineDetect."""
    load_custom_css()
    render_header()

    # Comprobar presencia del modelo
    model_exists = check_model_exists()
    device_info = get_device_info()
    
    # Renderizar barra lateral y obtener parámetros
    params = render_sidebar(model_available=model_exists, device_info=device_info)

    # Intentar cargar el modelo si existe
    model = None
    if model_exists:
        model, load_err = load_model(use_cache=True)
        if load_err:
            st.error(f"⚠️ Error al inicializar el modelo: {load_err}")

    # Zona de Carga de Imagen
    st.markdown("### 📤 Cargar Imagen para Detección")
    uploaded_file = st.file_uploader(
        "Arrastra y suelta una imagen o haz clic para seleccionarla (JPG, JPEG, PNG, WEBP — Máx. 10 MB)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos admitidos: JPG, JPEG, PNG, WEBP. Tamaño máximo: 10 MB."
    )

    if uploaded_file is None:
        render_initial_guide()
        # Resetear estado si se descarta el archivo
        if "detection_state" in st.session_state:
            del st.session_state["detection_state"]
    else:
        # 1. Validar la imagen cargada
        file_bytes = uploaded_file.getvalue()
        val_result: ImageValidationResult = validate_image_file(file_bytes, uploaded_file.name)

        if not val_result.is_valid:
            st.error(f"❌ **Error en la imagen:** {val_result.error_message}")
            st.info("💡 Por favor, verifica el formato, tamaño y estado del archivo e intenta nuevamente.")
            return

        valid_image = val_result.image
        
        # Mostrar información preliminar de la imagen
        st.caption(
            f"ℹ️ Archivo: **{uploaded_file.name}** | Formato: **{val_result.format}** | "
            f"Resolución: **{val_result.width} × {val_result.height} px** | "
            f"Tamaño: **{val_result.size_mb:.2f} MB**"
        )

        # Botón para ejecutar el análisis
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button(
                "🍍 Analizar imagen",
                type="primary",
                use_container_width=True,
                disabled=not model_exists or model is None
            )

        if not model_exists:
            st.warning(
                f"⚠️ Para ejecutar la detección, asegúrate de colocar el archivo "
                f"**{MODEL_FILENAME}** en la carpeta `models/` del proyecto."
            )

        # Ejecutar inferencia cuando se pulsa el botón
        if analyze_clicked and model is not None:
            with st.spinner("Analizando la imagen con PineDetect..."):
                try:
                    detections, summary = run_pineapple_detection(
                        model=model,
                        image_input=valid_image,
                        imgsz=params["imgsz"],
                        conf=params["conf"],
                        iou=params["iou"]
                    )
                    annotated_image = draw_detections_on_image(valid_image, detections)
                    
                    # Guardar en session_state para persistencia
                    st.session_state["detection_state"] = {
                        "filename": uploaded_file.name,
                        "original_image": valid_image,
                        "annotated_image": annotated_image,
                        "detections": detections,
                        "summary": summary,
                        "params": params
                    }
                except Exception as exc:
                    logger.error("Error durante la inferencia: %s", str(exc), exc_info=True)
                    st.error(f"Ocurrió un error inesperado durante el procesamiento: {str(exc)}")

        # Renderizar resultados si existen en session_state
        state = st.session_state.get("detection_state")
        if state and state.get("filename") == uploaded_file.name:
            detections = state["detections"]
            summary: InferenceSummary = state["summary"]
            annotated_img = state["annotated_image"]

            st.markdown("---")
            st.markdown("## 📊 Resultados del Análisis")

            # Comparativa Visual: 2 Columnas
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="image-box-title">📷 Imagen Original</div>', unsafe_allow_html=True)
                st.image(state["original_image"], use_container_width=True)

            with col_right:
                st.markdown('<div class="image-box-title">🎯 Imagen Analizada (Detección y Madurez)</div>', unsafe_allow_html=True)
                st.image(annotated_img, use_container_width=True)

            # Caso: Sin Detecciones
            if summary.total_detections == 0:
                st.warning("⚠️ **No se detectaron piñas con el nivel de confianza seleccionado.**")
                st.info(
                    "💡 **Recomendación:** Prueba con una imagen más clara o reduce ligeramente el "
                    "umbral de confianza mínima en la barra lateral para volver a analizar."
                )
            else:
                # Tarjetas de Resumen
                st.markdown("### 📈 Métricas Generales")
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(
                        f"""
                        <div class="pine-metric-card">
                            <div class="pine-metric-value">{summary.total_detections}</div>
                            <div class="pine-metric-label">Piñas Detectadas</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m2:
                    avg_conf_str = (
                        f"{summary.average_confidence * 100:.1f} %"
                        if summary.average_confidence is not None
                        else "N/A"
                    )
                    st.markdown(
                        f"""
                        <div class="pine-metric-card">
                            <div class="pine-metric-value">{avg_conf_str}</div>
                            <div class="pine-metric-label">Confianza Promedio</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m3:
                    st.markdown(
                        f"""
                        <div class="pine-metric-card">
                            <div class="pine-metric-value">{summary.inference_time_ms:.1f} <small style="font-size:1rem;">ms</small></div>
                            <div class="pine-metric-label">Tiempo de Inferencia</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m4:
                    st.markdown(
                        f"""
                        <div class="pine-metric-card accent">
                            <div class="pine-metric-value" style="font-size: 1.45rem;">{summary.predominant_class}</div>
                            <div class="pine-metric-label">Clase Predominante</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Desglose por Clase y Gráfico Interactivo
                st.markdown("### 🍍 Conteo por Estado de Madurez")
                col_counts, col_chart = st.columns([1, 2])

                with col_counts:
                    inm_cnt = summary.counts_by_class.get("inmadura", 0)
                    inm_pct = summary.percentages_by_class.get("inmadura", 0.0)
                    mad_cnt = summary.counts_by_class.get("madura", 0)
                    mad_pct = summary.percentages_by_class.get("madura", 0.0)
                    sob_cnt = summary.counts_by_class.get("sobremadura", 0)
                    sob_pct = summary.percentages_by_class.get("sobremadura", 0.0)

                    st.markdown(
                        f"""
                        <div class="pine-class-pill inmadura">
                            <span><span class="pine-pill-indicator inmadura"></span> <b>Inmadura</b></span>
                            <span>{inm_cnt} ({inm_pct:.1f}%)</span>
                        </div>
                        <div class="pine-class-pill madura">
                            <span><span class="pine-pill-indicator madura"></span> <b>Madura</b></span>
                            <span>{mad_cnt} ({mad_pct:.1f}%)</span>
                        </div>
                        <div class="pine-class-pill sobremadura">
                            <span><span class="pine-pill-indicator sobremadura"></span> <b>Sobremadura</b></span>
                            <span>{sob_cnt} ({sob_pct:.1f}%)</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_chart:
                    chart_fig = create_class_distribution_chart(summary.counts_by_class)
                    if chart_fig is not None:
                        st.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})

                # Tabla Detallada de Detecciones
                st.markdown("### 📋 Tabla de Detecciones")
                table_rows = [d.to_table_row() for d in detections]
                df_detections = pd.DataFrame(table_rows)
                st.dataframe(
                    df_detections,
                    use_container_width=True,
                    hide_index=True
                )

            # Sección de Descargas
            st.markdown("### 💾 Exportar y Descargar Resultados")
            col_d1, col_d2, col_d3 = st.columns(3)

            # 1. Descargar imagen analizada (PNG)
            with col_d1:
                png_bytes = export_image_to_png_bytes(annotated_img)
                st.download_button(
                    label="🖼️ Descargar imagen analizada",
                    data=png_bytes,
                    file_name=f"pinedetect_{Path(uploaded_file.name).stem}_analizada.png",
                    mime="image/png",
                    use_container_width=True
                )

            # 2. Descargar CSV con BOM
            with col_d2:
                csv_bytes = export_detections_to_csv_bytes(detections)
                st.download_button(
                    label="📊 Descargar resultados CSV",
                    data=csv_bytes,
                    file_name=f"pinedetect_{Path(uploaded_file.name).stem}_detecciones.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # 3. Descargar resumen JSON
            with col_d3:
                json_bytes = export_summary_to_json_bytes(
                    summary=summary,
                    detections=detections,
                    thresholds=params,
                    model_name=MODEL_FILENAME,
                    additional_metadata={"archivo_origen": uploaded_file.name}
                )
                st.download_button(
                    label="📋 Descargar resumen JSON",
                    data=json_bytes,
                    file_name=f"pinedetect_{Path(uploaded_file.name).stem}_resumen.json",
                    mime="application/json",
                    use_container_width=True
                )
        elif not state:
            # Vista previa de la imagen recién cargada antes de analizar
            st.markdown("#### Vista previa de la imagen:")
            st.image(valid_image, caption="Imagen lista para análisis", use_container_width=True)

    # Pie de página y Privacidad
    st.markdown(
        f"""
        <div class="pine-footer">
            <div class="pine-privacy-badge">🔒 Aviso de Privacidad</div><br>
            {PRIVACY_NOTICE_TEXT}<br>
            <small>© {APP_NAME} — Detección y Clasificación Inteligente de Piñas</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
