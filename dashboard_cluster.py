import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from pathlib import Path
import os

# ───────────────────────── Configuração global ─────────────────────────
st.set_page_config(
    page_title="Dashboard de Análise de Clusters para o Município de São Paulo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Esconde menu padrão e rodapé
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* título customizado */
        .dashboard-title {
            font-size: 36px;
            font-weight: 700;
            color: #C65534;  /* cor laranja-queimada da paleta */
            margin: 0;
            line-height: 1.1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Cabeçalho com logo + título ─────────────────────────
logo_path = Path(__file__).parent / "assets" / "logo_dash.png"  # coloque sua imagem aqui
col_logo, col_title, _ = st.columns([1, 3, 1])
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), use_column_width="auto")
with col_title:
    st.markdown(
        "<p class='dashboard-title'>DASHBOARD DE ANÁLISE DE CLUSTERS PARA O MUNICÍPIO DE SÃO PAULO</p>",
        unsafe_allow_html=True,
    )

PLOTLY_TEMPLATE = "plotly_white"
CLASSE_CORES = {
    0: "#F4DD63",
    1: "#B1BF7C",
    2: "#D58243",
    3: "#C65534",
    4: "#6FA097",
    5: "#14407D",
}

GROUP_COLS = ["KMeans_k5", "Spectral_k5", "KMedoids_k5"]

# ───────────────────────── Paths ─────────────────────────
BASE_DIR = Path(__file__).parent
PASTA_DADOS = BASE_DIR / "data" / "metricas"
PASTA_ANALISES = BASE_DIR / "data" / "merged"

# (restante do código permanece inalterado)
