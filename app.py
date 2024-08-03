import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from pag.pag1 import show_results

# Configuração da página principal
st.set_page_config(page_title="CatsandDogs", page_icon="🌎")

# Aplicar estilos de CSS à página
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Carregue o modelo treinado
try:
    model = tf.keras.models.load_model("modelo/my_model.h5")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {str(e)}")
    st.stop()

# Função para fazer previsões
def make_prediction(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((128, 128))  # Corrigido para (128, 128)
        img_array = np.array(img) / 255.0  # Normaliza a imagem
        img_array = np.expand_dims(img_array, axis=0)

        # Simulação de progresso (apenas para efeito visual)
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.1)
            progress_bar.progress(percent)
        
        prediction = model.predict(img_array)

        # Obter a classe prevista e a probabilidade associada
        class_names = ["Gato", "Cachorro"]
        max_prob_index = np.argmax(prediction[0])
        predicted_class = class_names[max_prob_index]
        max_probability = 100 * prediction[0][max_prob_index]

        # Obter as probabilidades para Gato e Cachorro
        prob_gato = 100 * prediction[0][0]
        prob_cachorro = 100 * prediction[0][1]

        # Verificar se a imagem pode ser considerada como Gato ou Cachorro
        min_prob_threshold = 10  # Define um limiar mínimo de probabilidade
        if prob_gato < min_prob_threshold and prob_cachorro < min_prob_threshold:
            return None, "Imagem não reconhecida como Gato ou Cachorro. Por favor, carregue uma imagem válida de um gato ou cachorro."

        # Retorna o resultado e as probabilidades
        return (predicted_class, prob_gato, prob_cachorro), None
        
    except Exception as e:
        return None, str(e)

# Configurações do Streamlit
st.title("Cats_and_Dogs_IA: Classificador de Gatos e Cachorros 🐶🐱🐾")
st.subheader("Faça uma previsão: Cachorro ou Gato?")
st.info("Este modelo só pode prever se uma imagem é de um gato ou cachorro. Por favor, carregue uma imagem correspondente.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exiba a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem de Entrada", use_column_width=True)

# Faça a previsão quando o botão for pressionado
if st.button("Fazer Previsão"):
    if uploaded_file is None:
        st.warning("Por favor, carregue uma imagem antes de fazer a previsão.")
    else:
        result, error = make_prediction(uploaded_file)
        if error:
            st.error(error)
            st.image("media/error_image.png", caption="Imagem inválida", use_column_width=True)  # Exibe uma imagem de erro
        else:
            predicted_class, prob_gato, prob_cachorro = result
            st.success(f"O modelo classificou a imagem como um {predicted_class}.")
            st.success(f"Com {prob_gato:.1f}% para Gato e {prob_cachorro:.1f}% para Cachorro!")

# Adicione uma foto sua
try:
    st.sidebar.image(
        Image.open("media/eu1.jpeg"),
        caption="Desenvolvedor: Thaleson Silva",
        use_column_width=True,
    )
except FileNotFoundError:
    st.sidebar.warning("A imagem do desenvolvedor não foi encontrada.")

linkedin_link = "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/thaleson-silva-9298a0296/)"
github_link = "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thaleson)"
instagram_link = "[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/_thaleson/)"

st.sidebar.markdown(linkedin_link, unsafe_allow_html=True)
st.sidebar.markdown(github_link, unsafe_allow_html=True)
st.sidebar.markdown(instagram_link, unsafe_allow_html=True)

# Adicione uma seção para a análise de resultados
st.header("Análise de Resultados")

# Botão para visualizar os resultados
if st.button("Ver Resultados"):
    show_results()
