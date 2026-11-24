
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Clasificador de residuos",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Clasificador de residuos")

st.write("Proyecto de Computación en la Nube - UTH 2026")

st.write(
    "Tome una fotografía o cargue una imagen "
    "para identificar el tipo de residuo."
)

st.caption("Clases: Cartón, Vidrio, Papel y Plástico")

IMG_SIZE = (224, 224)

MODEL_DIR = Path(__file__).resolve().parent
CLASS_PATH = MODEL_DIR / "classes.json"
MODEL_PATH = MODEL_DIR / "modelo_residuos.keras"

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "paper": "Papel",
    "plastic": "Plástico"
}

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

@st.cache_data
def cargar_clases():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def preparar_imagen(img):
    img = img.convert("RGB").resize(
        IMG_SIZE, Image.Resampling.BILINEAR
    )
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

modelo = cargar_modelo()
clases = cargar_clases()

# Seleccionar el método de captura
opcion = st.radio(
    "Seleccione una opción:",
    ["Tomar fotografía", "Cargar imagen"],
    horizontal=True
)

if opcion == "Tomar fotografía":
    archivo = st.camera_input(
        "Tome una fotografía del residuo"
    )
else:
    archivo = st.file_uploader(
        "Seleccione una imagen",
        type=["jpg", "jpeg", "png"]
    )

if archivo is not None:

    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen analizada",
        use_container_width=True
    )

    # Realizar predicción
    preds = modelo.predict(
        preparar_imagen(imagen),
        verbose=0
    )[0]

    top3 = np.argsort(preds)[-3:][::-1]

    mejor = top3[0]

    # Predicción principal
    st.subheader("Predicción principal")

    nombre = LABELS_ES.get(
        clases[mejor], clases[mejor]
    )

    confianza = float(preds[mejor]) * 100

    st.success(f"{nombre}: {confianza:.2f}%")

    # Tres mejores predicciones
    st.subheader("Top 3")

    for i in top3:
        nombre = LABELS_ES.get(
            clases[i], clases[i]
        )
        porcentaje = float(preds[i]) * 100

        st.markdown(
            f"- {nombre}: {porcentaje:.2f}%"
        )

    # Detalle de todas las clases
    st.subheader("Detalle por clase")

    for i, clase in enumerate(clases):

        nombre = LABELS_ES.get(clase, clase)
        probabilidad = float(preds[i])

        st.write(
            f"{nombre}: {probabilidad * 100:.2f}%"
        )

        st.progress(probabilidad)

else:
    st.info(
        "Tome una fotografía o cargue una imagen "
        "para iniciar la clasificación."
    )

st.divider()

st.caption("Universidad Tecnológica de Honduras")
st.caption("Asignatura: Computación en la Nube")
st.caption("Docente: Ing. Asalia Zavala")
st.caption("Estudiante: Linda Rosibel Medina Lanza")
