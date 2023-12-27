# Importe as bibliotecas necessárias


import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from keras.preprocessing import image

import streamlit as st
from pag.pag1 import show_results

# Carregue o modelo treinado
# Substitua pelo caminho real do seu modelo
model = tf.keras.models.load_model("modelo/my_m.h5")


# ...

# Função para fazer previsões


def make_prediction(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normaliza a imagem
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    # Interpretação da previsão
    if prediction[0, 0] > 0.5:
        result = "Cachorro"
    else:
        result = "Gato"

    return result


# ...


# Configurações do Streamlit
st.title("Cats_and_Dogs_IA: Classificador de Gatos e Cachorros 🐶🐱🐾")
st.subheader("Faça uma previsão: Cachorro ou Gato?")

uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    # Exiba a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem de Entrada", use_column_width=True)

    # Faça a previsão quando o botão for pressionado
# ...

# Faça a previsão quando o botão for pressionado
if st.button("Fazer Previsão"):
    # Verifique se uma imagem foi carregada
    if uploaded_file is None:
        st.warning("Por favor, carregue uma imagem antes de fazer a previsão.")
    else:
        # Verifique se a imagem é uma foto de gato ou cachorro
        valid_extensions = ["jpg", "jpeg", "png"]
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension not in valid_extensions:
            st.warning(
                "Por favor, carregue uma imagem válida no formato JPG, JPEG ou PNG."
            )
        else:
            # Faça a previsão e exiba o resultado
            prediction = make_prediction(uploaded_file)
            st.write(f"Isto é um : {prediction}")

# ...


# Adicione uma foto sua
st.sidebar.image(
    Image.open("C:/Users/thale/Challenge_1/media/eu1.jpeg"),
    caption="Desenvolvedor: Thaleson silva",
    use_column_width=True,
)


linkedin_link = "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/thaleson-silva-9298a0296/)"
github_link = "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thaleson)"
instagram_link = "[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/_thaleson/)"

st.sidebar.markdown(linkedin_link, unsafe_allow_html=True)
st.sidebar.markdown(github_link, unsafe_allow_html=True)
st.sidebar.markdown(instagram_link, unsafe_allow_html=True)

# Importe a função da nova página

# Adicione outras seções ou funcionalidades da sua página principal aqui

# Adicione uma seção para a análise de resultados
st.header("Análise de Resultados")

# Botão para visualizar os resultados
if st.button("Ver Resultados"):
    # Chame a função da nova página para exibir os resultados
    show_results()
