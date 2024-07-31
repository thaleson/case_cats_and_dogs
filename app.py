import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

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
        # Simula um processo de classificação com uma barra de progresso
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        # Faz a previsão
        predicted_class, max_probability = make_prediction(image)
        
        if predicted_class:
            if max_probability < 90:  # Se a maior probabilidade for menor que 90%
                st.warning("Parece que você não enviou uma foto clara de um gato ou cachorro. Por favor, tente outra imagem!")
            else:
                # Obtém as probabilidades para Gato e Cachorro
                prob_gato = 100 * (1 - prediction[0][1])
                prob_cachorro = 100 * prediction[0][1]

                # Mostra a classificação
                st.success(f"O modelo classificou a imagem como um {predicted_class}.")
                
                # Mostra as probabilidades
                st.success(f"Com {prob_gato:.1f}% para Gato e {prob_cachorro:.1f}% para Cachorro!")
        else:
            st.error("Ocorreu um erro ao tentar fazer a previsão.")
