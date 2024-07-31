pegue as alteraçoes dele  do codigo dele e bote no meu codigo 

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Carregar o modelo
model_path = "modelo/my_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {str(e)}")
    st.stop()


# Função de previsão
def make_prediction(image):
    try:
        img = image.convert("RGB")
        img = img.resize((128, 128))  # Corrigido para (128, 128)
        img_array = np.array(img) / 255.0  # Normaliza a imagem
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        class_names = ["Gato", "Cachorro"]
        max_prob_index = np.argmax(prediction[0])
        predicted_class = class_names[max_prob_index]
        max_probability = 100 * prediction[0][max_prob_index]

        return predicted_class, max_probability
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {str(e)}")
        return None, None

# Streamlit UI
st.title("Cats_and_Dogs_IA: Classificador de Gatos e Cachorros 🐶🐱🐾")
st.subheader("Faça uma previsão: Cachorro ou Gato?")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem de Entrada", use_column_width=True)

    if st.button("Fazer Previsão"):
        prediction = make_prediction(uploaded_file)
        if prediction:
            st.write(f"Isto é um : {prediction}")
        else:
            st.error("Ocorreu um erro ao tentar fazer a previsão.")

