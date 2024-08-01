import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_results():
    # Carregue os dados do CSV
    csv_path = "results/resultados.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("Erro: O arquivo de resultados não foi encontrado.")
        return

    # Detecta se o fundo é claro ou escuro
    background_color = st.get_option("theme.backgroundColor")  # Obtém a cor de fundo atual
    if background_color:
        light_mode = True if background_color.lower() in ["#ffffff", "white", "rgb(255, 255, 255)"] else False
    else:
        light_mode = False

    text_color = '#000000' if light_mode else '#ecf0f1'
    subtitle_color = '#2c3e50' if light_mode else '#bdc3c7'

    # Título da página
    st.markdown(f"<h1 style='color: {text_color};'>Resultados de Treinamento e Validação</h1>", unsafe_allow_html=True)

  

    # Descrição introdutória
    st.markdown(f"""
    <div style="font-size:18px; color:{subtitle_color}; line-height:1.5;">
        Bem-vindo à página de resultados! Aqui, você pode visualizar os dados de treinamento e validação
        do modelo de classificação de imagens. Os gráficos abaixo mostram as métricas de desempenho ao longo
        das épocas de treinamento. Use estas informações para avaliar o comportamento e a eficácia do modelo.
    </div>
    """, unsafe_allow_html=True)

    # Visualização de dados brutos
    st.markdown(f"<h2 style='color: {text_color};'>Visualização de Dados Brutos</h2>", unsafe_allow_html=True)
    st.dataframe(df)

    # Análise estatística básica
    st.markdown(f"<h2 style='color: {text_color};'>Análise Estatística Básica</h2>", unsafe_allow_html=True)
    st.write("Aqui estão algumas estatísticas básicas dos dados de treinamento e validação:")
    st.write(df.describe())

    # Gráficos
    st.markdown(f"<h2 style='color: {text_color};'>Visualizações Gráficas</h2>", unsafe_allow_html=True)

    # Gráfico de Linhas para Loss e Val Loss
    st.markdown(f"<h3 style='color: {text_color};'>Loss e Val Loss por Época</h3>", unsafe_allow_html=True)
    st.write(f"""
    <div style="color: {subtitle_color};">
    Este gráfico mostra a evolução da perda (loss) e da perda de validação (val loss) ao longo das épocas.
    A perda é uma métrica que indica o quanto as previsões do modelo estão longe dos valores reais. 
    O objetivo é minimizar tanto a loss quanto a val loss.
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['loss'], label='Loss', color='skyblue', linewidth=2)
    ax.plot(df['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax.set_xlabel('Época', color=text_color)
    ax.set_ylabel('Perda (Loss)', color=text_color)
    ax.set_title('Loss e Val Loss ao Longo das Épocas', color=text_color)
    ax.legend()
    plt.setp(ax.get_xticklabels(), color=text_color)
    plt.setp(ax.get_yticklabels(), color=text_color)
    st.pyplot(fig)

    # Gráfico de Barras para Accuracy e Val Accuracy
    st.markdown(f"<h3 style='color: {text_color};'>Acurácia e Val Acurácia por Época</h3>", unsafe_allow_html=True)
    st.write(f"""
    <div style="color: {subtitle_color};">
    Este gráfico apresenta a acurácia de treinamento e validação do modelo ao longo das épocas. 
    A acurácia mede a proporção de previsões corretas feitas pelo modelo. O objetivo é maximizar
    tanto a acurácia de treinamento quanto a de validação.
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['accuracy'], label='Acurácia', color='limegreen', linewidth=2)
    ax.plot(df['val_accuracy'], label='Val Acurácia', color='tomato', linewidth=2)
    ax.set_xlabel('Época', color=text_color)
    ax.set_ylabel('Acurácia', color=text_color)
    ax.set_title('Acurácia e Val Acurácia ao Longo das Épocas', color=text_color)
    ax.legend()
    plt.setp(ax.get_xticklabels(), color=text_color)
    plt.setp(ax.get_yticklabels(), color=text_color)
    st.pyplot(fig)

    # Mensagem final
    st.markdown(f"""
    <div style="font-size:16px; color:{subtitle_color}; line-height:1.5;">
        Esses resultados ajudam a identificar se o modelo está aprendendo de forma eficiente
        ou se há sinais de overfitting ou underfitting. Continue ajustando e validando seu modelo
        para alcançar uma performance ótima.
    </div>
    """, unsafe_allow_html=True)


# Execute a função quando este script for executado diretamente
if __name__ == "__main__":
    show_results()
