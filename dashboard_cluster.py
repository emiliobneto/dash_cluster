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

# ───────────────────────── Funções utilitárias ─────────────────────────
@st.cache_data(show_spinner=False)
def carregar_todos_arquivos(pasta: Path):
    arquivos = {}
    for csv in pasta.rglob("*.csv"):
        try:
            df = pd.read_csv(csv)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            arquivos[csv.name] = df
        except Exception as e:
            st.warning(f"Erro ao carregar {csv.name}: {e}")
    return arquivos

def normalizar_df(df, estatisticas):
    df_norm = df.copy()
    for stat in estatisticas:
        if stat in df_norm.columns:
            for var in df_norm["Variável"].unique():
                mask = df_norm["Variável"] == var
                min_val = df_norm.loc[mask, stat].min()
                max_val = df_norm.loc[mask, stat].max()
                if min_val != max_val:
                    df_norm.loc[mask, stat] = (df_norm.loc[mask, stat] - min_val) / (
                        max_val - min_val
                    )
    return df_norm

# ──────────── Plot funções ────────────

def plot_barras(df, stat_col):
    fig = px.bar(
        df,
        x="Variável",
        y=stat_col,
        color="Classe",
        color_discrete_map=CLASSE_CORES,
        barmode="group",
        facet_col="Método",
        facet_col_wrap=2,
        title=f"{stat_col.capitalize()} por Variável",
        labels={stat_col: stat_col.capitalize()},
        height=480,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    return fig

def plot_radar(df, stat_col, metodos, classes):
    fig = go.Figure()
    for classe in classes:
        for metodo in metodos:
            df_tmp = df[(df["Classe"] == classe) & (df["Método"] == metodo)]
            if not df_tmp.empty:
                fig.add_trace(
                    go.Scatterpolar(
                        r=df_tmp[stat_col],
                        theta=df_tmp["Variável"],
                        fill="toself",
                        name=f"{metodo} - Cluster {classe}",
                        line_color=CLASSE_CORES.get(classe, None),
                    )
                )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=580,
        template=PLOTLY_TEMPLATE,
        title=f"Radar Chart - {stat_col.capitalize()} por Variável",
    )
    return fig

def plot_univariadas(df, estatistica, group_col):
    st.markdown("### Análises Univariadas")

    for var in sorted(df["Variável"].unique()):
        st.markdown(f"#### Variável: {var}")
        df_var = df[df["Variável"] == var]

        n_por_classe = df_var.groupby(group_col)[estatistica].count().min()
        col1, col2 = st.columns(2)

        # ----- Coluna 1: Distribuição -----
        with col1:
            if n_por_classe <= 15:
                fig_strip = px.strip(
                    df_var,
                    x=group_col,
                    y=estatistica,
                    color=group_col,
                    color_discrete_map=CLASSE_CORES,
                    stripmode="overlay",
                    template=PLOTLY_TEMPLATE,
                    title=f"Valores individuais - {var}",
                )
                fig_strip.update_traces(jitter=0.35, marker_size=8)
                st.plotly_chart(fig_strip, use_container_width=True)
            else:
                fig_hist = px.histogram(
                    df_var,
                    x=estatistica,
                    color=group_col,
                    color_discrete_map=CLASSE_CORES,
                    marginal="rug",
                    nbins=min(20, max(5, df_var.shape[0] // 3)),
                    template=PLOTLY_TEMPLATE,
                    title=f"Distribuição - {var}",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # ----- Coluna 2: Violin + Box -----
        with col2:
            fig_violin = px.violin(
                df_var,
                x=group_col,
                y=estatistica,
                color=group_col,
                color_discrete_map=CLASSE_CORES,
                box=True,
                points="all",
                template=PLOTLY_TEMPLATE,
                title=f"Violin/Box - {var}",
            )
            st.plotly_chart(fig_violin, use_container_width=True)

        # ----- Resumo estatístico -----
        resumo = (
            df_var.groupby(group_col)[estatistica]
            .agg(n="count", média="mean", mediana="median", mín="min", máx="max", desvio="std")
            .round(2)
            .reset_index()
        )
        st.dataframe(resumo, use_container_width=True)

def analise_estatistica_variavel(group_col):
    st.markdown("## 📐 Análise Estatística por Variável")
    arquivos_merged = carregar_todos_arquivos(PASTA_ANALISES)
    if not arquivos_merged:
        st.warning("Nenhum arquivo em data/merged.")
        return

    arquivo_merged = st.selectbox("Selecione o arquivo:", list(arquivos_merged.keys()), key="merged_file")
    df_var = arquivos_merged[arquivo_merged]

    colunas_num = df_var.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if group_col not in df_var.columns:
        st.error(f"Coluna '{group_col}' não existe nesse arquivo.")
        return

    col_var = st.selectbox("Variável numérica:", colunas_num, index=0)

    st.markdown("### ANOVA")
    grupos = [grupo[col_var].dropna().values for _, grupo in df_var.groupby(group_col)]
    if len(grupos) > 1:
        f_stat, p_value = stats.f_oneway(*grupos)
        st.write(f"F = {f_stat:.4f},  p = {p_value:.4f}")
        st.success("Diferença significativa." if p_value < 0.05 else "Sem diferença significativa.")
    else:
        st.warning("Não há grupos suficientes para ANOVA.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Histograma por grupo")
        fig_hist = px.histogram(
            df_var,
            x=col_var,
            color=group_col,
            color_discrete_map=CLASSE_CORES,
            marginal="box",
            nbins=20,
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        st.markdown("#### Boxplot por grupo")
        fig_box = px.box(
            df_var,
            x=group_col,
            y=col_var,
            color=group_col,
            color_discrete_map=CLASSE_CORES,
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ───────────────────────── Carregamento inicial ─────────────────────────
bases = carregar_todos_arquivos(PASTA_DADOS)
if not bases:
    st.error("Nenhum CSV encontrado em data/metricas.")
