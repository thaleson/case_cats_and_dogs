import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#predict exemplo
def preprocess_image(image_path):
    # Carregue a imagem e faça o pré-processamento
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image


def plot_prediction(image, prediction):
    classes = ["Gato", "Cachorro"]
    result = "Cachorro" if prediction[0][0] > 0.5 else "Gato"

    # Exibir a imagem
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Isto é um: {result}")
    plt.show()


def main():
    # Carregue o modelo
    model = tf.keras.models.load_model("modelo/my_Model.h5")

    # Substitua 'path/to/your/image.jpg' pelo caminho da imagem que você deseja prever
    image_path = "models/ca3.jpg"

    # Pré-processamento da imagem
    processed_image, original_image = preprocess_image(image_path)

    # Faça a previsão
    prediction = model.predict(processed_image)

    # Exiba o resultado e a imagem
    plot_prediction(original_image, prediction)


if __name__ == "__main__":
    main()
