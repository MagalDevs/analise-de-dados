import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modelos de Classificação
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Métricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="Heart Disease Analytics",
    layout="wide"
)

st.title("❤️ Dashboard de Classificação - Heart Disease")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Painel de Configuração")

arquivo_postado = st.sidebar.file_uploader(
    "Upload do Dataset (CSV)",
    type="csv"
)

algoritmos_escolhidos = st.sidebar.multiselect(
    "Selecione os Algoritmos",
    [
        "KNN",
        "Regressão Logística",
        "Árvore de Decisão",
        "Random Forest",
        "SVM",
        "Naive Bayes"
    ],
    default=[
        "KNN",
        "Regressão Logística",
        "Random Forest"
    ]
)

# =========================
# PROCESSAMENTO
# =========================

if arquivo_postado is not None:

    # =========================
    # LEITURA DOS DADOS
    # =========================

    df = pd.read_csv(arquivo_postado)

    st.subheader("📄 Visualização do Dataset")
    st.dataframe(df.head())

    # =========================
    # REMOÇÃO DE NULOS
    # =========================

    df = df.dropna()

    # =========================
    # FEATURES E TARGET
    # =========================

    X = df.drop(columns=['target'])
    y = df['target']

    # =========================
    # DIVISÃO TREINO E TESTE
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # =========================
    # PADRONIZAÇÃO
    # =========================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tabela_resultados = []

    # =========================
    # ABAS
    # =========================

    tab_metricas, tab_graficos, tab_matriz = st.tabs([
        "📋 Métricas",
        "📈 Gráficos",
        "🧩 Matriz de Confusão"
    ])

    # =========================
    # MÉTRICAS
    # =========================

    with tab_metricas:

        st.subheader("📊 Performance dos Modelos")

        for algo in algoritmos_escolhidos:

            # =========================
            # ESCOLHA DO MODELO
            # =========================

            if algo == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
                usar_scaler = True

            elif algo == "Regressão Logística":
                model = LogisticRegression(max_iter=1000)
                usar_scaler = True

            elif algo == "Árvore de Decisão":
                model = DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
                usar_scaler = False

            elif algo == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                usar_scaler = False

            elif algo == "SVM":
                model = SVC(kernel='rbf')
                usar_scaler = True

            else:
                model = GaussianNB()
                usar_scaler = True

            # =========================
            # TREINAMENTO
            # =========================

            inicio = time.time()

            if usar_scaler:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            fim = time.time()

            tempo = (fim - inicio) * 1000

            # =========================
            # MÉTRICAS
            # =========================

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            tabela_resultados.append({
                "Algoritmo": algo,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
                "Tempo (ms)": round(tempo, 2),
                "y_pred": y_pred
            })

            # =========================
            # EXIBIÇÃO
            # =========================

            st.write(f"### {algo}")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", round(accuracy, 4))
            c2.metric("Precision", round(precision, 4))
            c3.metric("Recall", round(recall, 4))
            c4.metric("F1-Score", round(f1, 4))

            st.divider()

        # =========================
        # TABELA FINAL
        # =========================

        st.subheader("📑 Tabela Comparativa")

        df_final = pd.DataFrame(tabela_resultados).drop(columns=['y_pred'])

        st.dataframe(df_final, use_container_width=True)

    # =========================
    # GRÁFICOS
    # =========================

    with tab_graficos:

        st.subheader("📈 Comparação dos Modelos")

        if len(tabela_resultados) > 0:

            df_chart = pd.DataFrame(tabela_resultados)

            fig = px.bar(
                df_chart,
                x='Algoritmo',
                y='Accuracy',
                color='Algoritmo',
                title='Accuracy dos Modelos'
            )

            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                df_chart,
                x='Algoritmo',
                y='F1-Score',
                color='Algoritmo',
                title='F1-Score dos Modelos'
            )

            st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # MATRIZ DE CONFUSÃO
    # =========================

    with tab_matriz:

        st.subheader("🧩 Matrizes de Confusão")

        cols = st.columns(2)

        for idx, res in enumerate(tabela_resultados):

            with cols[idx % 2]:

                cm = confusion_matrix(y_test, res['y_pred'])

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title=f"Matriz de Confusão - {res['Algoritmo']}"
                )

                fig_cm.update_xaxes(
                    title='Predição'
                )

                fig_cm.update_yaxes(
                    title='Valor Real'
                )

                st.plotly_chart(fig_cm, use_container_width=True)

else:

    st.info(
        "⬅️ Faça upload do arquivo heart.csv para iniciar a análise."
    )