import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

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
    light_mode = True if background_color and background_color.lower() in ["#ffffff", "white", "rgb(255, 255, 255)"] else False
    
    text_color = '#000000' if light_mode else '#ecf0f1'
    subtitle_color = '#2c3e50' if light_mode else '#bdc3c7'

    # Título da página
    st.markdown(f"<h1 style='color: {text_color};'>Resultados de Treinamento e Validação</h1>", unsafe_allow_html=True)

    # Descrição introdutória
    st.markdown(f"""
    <div style="font-size:18px; color:{subtitle_color}; line-height:1.5;">
        Bem-vindo à página de resultados! Aqui você pode visualizar os dados de treinamento e validação
        do modelo de classificação de imagens. Analisaremos a evolução das métricas ao longo das épocas,
        incluindo gráficos de perda (loss), acurácia, e mais. Use estas informações para avaliar o comportamento
        e a eficácia do modelo.
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

    # Matriz de Confusão
    st.markdown(f"<h2 style='color: {text_color};'>Matriz de Confusão</h2>", unsafe_allow_html=True)
    y_true = df['true_labels']  # Adapte conforme o nome da coluna no seu CSV
    y_pred = df['predicted_labels']  # Adapte conforme o nome da coluna no seu CSV
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Gato', 'Cachorro'], yticklabels=['Gato', 'Cachorro'])
    ax.set_xlabel('Predição', color=text_color)
    ax.set_ylabel('Real', color=text_color)
    ax.set_title('Matriz de Confusão', color=text_color)
    plt.setp(ax.get_xticklabels(), color=text_color)
    plt.setp(ax.get_yticklabels(), color=text_color)
    st.pyplot(fig)

    # Curva ROC
    st.markdown(f"<h2 style='color: {text_color};'>Curva ROC</h2>", unsafe_allow_html=True)
    # Adapte y_true e y_scores conforme suas colunas
    y_true = df['true_labels']  # Adapte conforme o nome da coluna no seu CSV
    y_scores = df['predicted_scores']  # Adapte conforme o nome da coluna no seu CSV
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos', color=text_color)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', color=text_color)
    ax.set_title('Curva ROC', color=text_color)
    ax.legend(loc='lower right')
    plt.setp(ax.get_xticklabels(), color=text_color)
    plt.setp(ax.get_yticklabels(), color=text_color)
    st.pyplot(fig)

    # Classificação Report
    st.markdown(f"<h2 style='color: {text_color};'>Relatório de Classificação</h2>", unsafe_allow_html=True)
    classification_rep = classification_report(y_true, y_pred, target_names=['Gato', 'Cachorro'])
    st.text(classification_rep)

# Execute a função quando este script for executado diretamente
if __name__ == "__main__":
    show_results()
