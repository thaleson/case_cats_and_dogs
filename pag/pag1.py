import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



def show_results():
    # Carregue os dados do CSV
    csv_path = "results/resultados.csv"
    # Carregue os dados para um DataFrame
    df = pd.read_csv(csv_path)

    # Título da página
    st.title(" Resultados")

    # Subtítulo
    st.subheader("Visualizar Dados")
    st.dataframe(df)

    # Subtítulo
    st.subheader("Análise Estatística Básica")
    st.write(df.describe())

    # Subtítulo
    st.subheader("Gráficos")

    # Gráfico de Linhas para Loss e Val Loss
    fig, ax = plt.subplots(figsize=(10, 6))
     
    ax.plot(df['loss'], label='Loss')
    ax.plot(df['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    st.pyplot(fig)

    # Gráfico de Barras para Accuracy e Val Accuracy
   
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['loss'], label='Loss')
    ax.plot(df['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)


# Execute a função quando este script for executado diretamente
if __name__ == "__main__":
    show_results()
