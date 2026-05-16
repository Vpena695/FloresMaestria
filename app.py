
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Clasificador de Especies de Iris",
    page_icon="🌺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Aplicar principios de UX y teoría del color ---
st.markdown('''
<style>
/* Estilos generales */
.stApp {
    background-color: #e0f7fa; /* Un azul claro muy suave para el fondo */
    color: #212121; /* Texto oscuro para buena legibilidad */
}

/* Estilos para encabezados */
h1, h2, h3, h4, h5, h6 {
    color: #00796b; /* Un verde azulado agradable para los encabezados */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Estilos para el texto principal */
.stMarkdown {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #424242;
}

/* Estilos para los botones (sidebar sliders, etc.) */
.st-bv, .st-bl, .st-bb, .st-bg, .st-bi { /* Algunos selectores de componentes Streamlit para sliders, etc. */
    color: #004d40; /* Verde oscuro para elementos interactivos */
}

/* Estilos para la barra lateral */
.css-1d391kg, .css-1dp5dkx {
    background-color: #b2dfdb; /* Un verde azulado más claro para la sidebar */
    color: #004d40;
}
.css-1d391kg .st-d7, .css-1dp5dkx .st-d7 { /* Encabezados en la sidebar */
    color: #004d40;
}

/* Estilos para mensajes de éxito */
.stSuccess {
    background-color: #e8f5e9; /* Fondo muy claro para éxito */
    color: #2e7d32; /* Texto verde oscuro */
    border-left: 5px solid #4CAF50; /* Borde verde */
}

/* Estilos para mensajes de error */
.stError {
    background-color: #ffebee; /* Fondo muy claro para error */
    color: #c62828; /* Texto rojo oscuro */
    border-left: 5px solid #ef5350; /* Borde rojo */
}

/* Estilos para el dataframe */
.dataframe {
    border-radius: 8px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}

</style>
''', unsafe_allow_html=True)
# --- Fin de estilos UX/Color ---

st.title("🌸 Clasificador de Especies de Iris")
st.markdown("---")
st.subheader("¡Descubre la Especie de tu Flor de Iris!")
st.write("Ajusta los parámetros del sépalo y el pétalo usando los deslizadores en la barra lateral para obtener una predicción instantánea de la especie de Iris (setosa, versicolor o virginica).")
st.markdown("---")

# Cargar el modelo y el LabelEncoder
try:
    @st.cache_resource
    def load_model_and_encoder():
        model = joblib.load('best_knn_model.joblib')
        encoder = joblib.load('label_encoder.joblib')
        return model, encoder

    best_knn_model, label_encoder = load_model_and_encoder()
    st.success("✅ Modelos y codificador cargados exitosamente. ¡Listo para predecir!")
except Exception as e:
    st.error(f"❌ Error crítico al cargar los modelos: {e}. Asegúrate de que los archivos 'best_knn_model.joblib' y 'label_encoder.joblib' estén en la ruta correcta.")
    st.stop()

st.sidebar.header('📏 Introduce las Medidas de la Flor')

def user_input_features():
    sepal_length = st.sidebar.slider('Longitud del Sépalo (cm)', 4.3, 7.9, 5.4, 0.1)
    sepal_width = st.sidebar.slider('Ancho del Sépalo (cm)', 2.0, 4.4, 3.4, 0.1)
    petal_length = st.sidebar.slider('Longitud del Pétalo (cm)', 1.0, 6.9, 1.3, 0.1)
    petal_width = st.sidebar.slider('Ancho del Pétalo (cm)', 0.1, 2.5, 0.2, 0.1)

    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('📊 Valores de Entrada Seleccionados:')
st.dataframe(df_input.style.format("{:.2f} cm")) # Display with 2 decimal places and 'cm' suffix

# Realizar la predicción
prediction_encoded = best_knn_model.predict(df_input)
predicted_species = label_encoder.inverse_transform(prediction_encoded)

st.markdown("---")
st.subheader('🎯 Resultado de la Predicción:')
st.markdown(f"La especie de Iris predicha es: <p style='color:#00796b;font-size:24px;font-weight:bold;'>{predicted_species[0].upper()}</p>", unsafe_allow_html=True)
st.markdown("---")

st.info("💡 Esta aplicación es un ejemplo de cómo desplegar modelos de Machine Learning con Streamlit y hacerlos accesibles públicamente con ngrok.")
