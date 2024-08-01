import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Defina a função para mostrar os resultados
def show_results():
    # Carregue os dados do CSV
    csv_path = "results/resultados.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("Erro: O arquivo de resultados não foi encontrado.")
        return

    # Título da página
    st.title("Resultados de Treinamento e Validação")

    # Descrição introdutória
    st.markdown("""
    <div style="font-size:18px; color:#2c3e50; line-height:1.5;">
        Bem-vindo à página de resultados! Aqui, você pode visualizar os dados de treinamento e validação
        do modelo de classificação de imagens. Os gráficos abaixo mostram as métricas de desempenho ao longo
        das épocas de treinamento. Use estas informações para avaliar o comportamento e a eficácia do modelo.
    </div>
    """, unsafe_allow_html=True)

    # Subtítulo para visualizar os dados
    st.subheader("Visualização de Dados Brutos")
    st.dataframe(df)

    # Subtítulo para análise estatística básica
    st.subheader("Análise Estatística Básica")
    st.write("Aqui estão algumas estatísticas básicas dos dados de treinamento e validação:")
    st.write(df.describe())

    # Subtítulo para gráficos
    st.subheader("Visualizações Gráficas")

    # Gráfico de Linhas para Loss e Val Loss
    st.markdown("### Loss e Val Loss por Época")
    st.write("""
    Este gráfico mostra a evolução da perda (loss) e da perda de validação (val loss) ao longo das épocas.
    A perda é uma métrica que indica o quanto as previsões do modelo estão longe dos valores reais. 
    O objetivo é minimizar tanto a loss quanto a val loss.
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['loss'], label='Loss', color='skyblue', linewidth=2)
    ax.plot(df['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax.set_xlabel('Época')
    ax.set_ylabel('Perda (Loss)')
    ax.set_title('Loss e Val Loss ao Longo das Épocas')
    ax.legend()
    st.pyplot(fig)

    # Gráfico de Barras para Accuracy e Val Accuracy
    st.markdown("### Acurácia e Val Acurácia por Época")
    st.write("""
    Este gráfico apresenta a acurácia de treinamento e validação do modelo ao longo das épocas. 
    A acurácia mede a proporção de previsões corretas feitas pelo modelo. O objetivo é maximizar
    tanto a acurácia de treinamento quanto a de validação.
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['accuracy'], label='Acurácia', color='limegreen', linewidth=2)
    ax.plot(df['val_accuracy'], label='Val Acurácia', color='tomato', linewidth=2)
    ax.set_xlabel('Época')
    ax.set_ylabel('Acurácia')
    ax.set_title('Acurácia e Val Acurácia ao Longo das Épocas')
    ax.legend()
    st.pyplot(fig)

    # Mensagem final
    st.markdown("""
    <div style="font-size:16px; color:#34495e; line-height:1.5;">
        Esses resultados ajudam a identificar se o modelo está aprendendo de forma eficiente
        ou se há sinais de overfitting ou underfitting. Continue ajustando e validando seu modelo
        para alcançar uma performance ótima.
    </div>
    """, unsafe_allow_html=True)

# Execute a função quando este script for executado diretamente
if __name__ == "__main__":
    show_results()
