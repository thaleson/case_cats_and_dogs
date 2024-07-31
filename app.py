import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from pag.pag1 import show_results

# Carregue o modelo treinado
# Substitua pelo caminho real do seu modelo
try:
    model = tf.keras.models.load_model("modelo/my_model.h5")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {str(e)}")
    st.stop()

# Função para fazer previsões
def make_prediction(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128))  # Corrigido para (128, 128)
        img_array = np.array(img) / 255.0  # Normaliza a imagem
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        # Interpretação da previsão
        if prediction[0, 0] > 0.5:
            result = "Cachorro"
        else:
            result = "Gato"
        return result
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {str(e)}")
        return None

# Configurações do Streamlit
st.title("Cats_and_Dogs_IA: Classificador de Gatos e Cachorros 🐶🐱🐾")
st.subheader("Faça uma previsão: Cachorro ou Gato?")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exiba a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem de Entrada", use_column_width=True)

# Faça a previsão quando o botão for pressionado
if st.button("Fazer Previsão"):
    # Verifique se uma imagem foi carregada
    if uploaded_file is None:
        st.warning("Por favor, carregue uma imagem antes de fazer a previsão.")
    else:
        # Faça a previsão e exiba o resultado
        prediction = make_prediction(uploaded_file)
        if prediction:
            st.write(f"Isto é um : {prediction}")
        else:
            st.error("Ocorreu um erro ao tentar fazer a previsão.")

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
    # Chame a função da nova página para exibir os resultados
    show_results()
