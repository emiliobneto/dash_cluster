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

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .dashboard-title {
            font-size: 36px;
            font-weight: 700;
            color: #C65534;
            margin: 0;
            line-height: 1.1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Cabeçalho ─────────────────────────
logo_path = Path(__file__).parent / "assets" / "logo_dash.png"
col_logo, col_title, _ = st.columns([1, 4, 1])
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), use_column_width="auto")
with col_title:
    st.markdown(
        "<p class='dashboard-title'>DASHBOARD DE ANÁLISE DE CLUSTERS PARA O MUNICÍPIO DE SÃO PAULO</p>",
        unsafe_allow_html=True,
    )

# ───────────────────────── Constantes ─────────────────────────
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

BASE_DIR = Path(__file__).parent
PASTA_DADOS = BASE_DIR / "data" / "metricas"
PASTA_ANALISES = BASE_DIR / "data" / "merged"

# ───────────────────────── Utilidades ─────────────────────────
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
                    df_norm.loc[mask, stat] = (df_norm.loc[mask, stat] - min_val) / (max_val - min_val)
    return df_norm

# ───────────────────────── Funções de plot ─────────────────────────

def plot_barras(df, stat_col):
    fig = px.bar(
        df, x="Variável", y=stat_col, color="Classe", color_discrete_map=CLASSE_CORES,
        barmode="group", facet_col="Método", facet_col_wrap=2,
        template=PLOTLY_TEMPLATE, height=480,
        title=f"{stat_col.capitalize()} por Variável",
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
                        r=df_tmp[stat_col], theta=df_tmp["Variável"], fill="toself",
                        name=f"{metodo} - Cluster {classe}", line_color=CLASSE_CORES.get(classe),
                    )
                )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                      template=PLOTLY_TEMPLATE, height=580,
                      title=f"Radar Chart - {stat_col.capitalize()} por Variável")
    return fig

def plot_univariadas(df, estatistica, group_col):
    st.markdown("### Análises Univariadas")
    for var in sorted(df["Variável"].unique()):
        st.markdown(f"#### Variável: {var}")
        df_var = df[df["Variável"] == var]
        n_min = df_var.groupby(group_col)[estatistica].count().min()
        col1, col2 = st.columns(2)
        with col1:
            if n_min <= 15:
                fig = px.strip(df_var, x=group_col, y=estatistica, color=group_col,
                                color_discrete_map=CLASSE_CORES, stripmode="overlay",
                                template=PLOTLY_TEMPLATE, title=f"Valores individuais - {var}")
                fig.update_traces(jitter=0.35, marker_size=8)
            else:
                fig = px.histogram(df_var, x=estatistica, color=group_col,
                                   color_discrete_map=CLASSE_CORES, marginal="rug",
                                   nbins=min(20, max(5, df_var.shape[0] // 3)),
                                   template=PLOTLY_TEMPLATE, title=f"Distribuição - {var}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig_v = px.violin(df_var, x=group_col, y=estatistica, color=group_col,
                              color_discrete_map=CLASSE_CORES, box=True, points="all",
                              template=PLOTLY_TEMPLATE, title=f"Violin/Box - {var}")
            st.plotly_chart(fig_v, use_container_width=True)
        resumo = (df_var.groupby(group_col)[estatistica]
                  .agg(n="count", média="mean", mediana="median", mín="min", máx="max", desvio="std")
                  .round(2).reset_index())
        st.dataframe(resumo, use_container_width=True)

# ───────────────────────── ANOVA ─────────────────────────

def analise_estatistica_variavel(group_col):
    st.markdown("## 📐 Análise Estatística por Variável")
    arquivos_merged = carregar_todos_arquivos(PASTA_ANALISES)
    if not arquivos_merged:
        st.warning("Nenhum arquivo em data/merged.")
        return
    arquivo_merged = st.selectbox("Selecione o arquivo:", list(arquivos_merged.keys()), key="merged_file")
    df_var = arquivos_merged[arquivo_merged]
    if group_col not in df_var.columns:
        st.error(f"Coluna '{group_col}' não existe neste arquivo.")
        return
    col_num = st.selectbox("Variável numérica:", df_var.select_dtypes(include=["float64", "int64"]).columns)
    grupos = [g[col_num].dropna().values for _, g in df_var.groupby(group_col)]
    st.markdown("### ANOVA")
    if len(grupos) > 1:
        f_stat, p_val = stats.f_oneway(*grupos)
        st.write(f"F = {f_stat:.4f}, p = {p_val:.4f}")
        st.success("Diferença significativa." if p_val < 0.05 else "Sem diferença significativa.")
    else:
        st.warning("Não há grupos suficientes para ANOVA.")
    c1, c2 = st.columns(2)
    with c1:
        fig_h = px.histogram(df_var, x=col_num, color=group_col, color_discrete_map=CLASSE_CORES,
                             marginal="box", nbins=20, template=PLOTLY_TEMPLATE,
                             title="Histograma por grupo")
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        fig_b = px.box(df_var, x=group_col, y=col_num, color=group_col, color_discrete_map=CLASSE_CORES,
                        template=PLOTLY_TEMPLATE, title="Boxplot por grupo")
        st.plotly_chart(fig_b, use_container_width=True)

# ───────────────────────── Carregamento inicial ─────────────────────────

df_metricas_files = carregar_todos_arquivos(PASTA_DADOS)
if not df_metricas_files:
    st.error("Nenhum CSV encontrado em data/metricas.")
    st.stop()

file_metricas = st.selectbox("Selecione o arquivo de métricas:", list(df_metricas_files.keys()))
df = df_metricas_files[file_metricas]

# Sidebar extras ---------------------------------------------------------
with st.sidebar:
    st.subheader("🔧 Configurações gerais")
    group_col_sel = st.selectbox("Agrupamento (coluna do cluster):", GROUP_COLS)

# Variáveis e estatísticas disponíveis -----------------------------------
metodos = sorted(df["Método"].unique())
classes = sorted(df["Classe"].unique())
variaveis = sorted(df["Variável"].unique())
estat_cols = [col for col in df.columns if col not in ["Método", "Classe", "Variável"]]

# Filtros ----------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    met_sel = st.multiselect("Métodos:", metodos, default=metodos)
    cls_sel = st.multiselect("Classes:", classes, default=classes)
    var_sel = st.multiselect("Variáveis:", variaveis, default=variaveis)
    est_sel = st.multiselect("Estatísticas:", estat_cols, default=[estat_cols[0]])
    view_mode = st.radio("Visualização:", ["Escala Real", "Normalizado", "Ambos"], index=0)

# Filtra dataframe -------------------------------------------------------
df_filt = df[(df["Método"].isin(met_sel)) &
             (df["Classe"].isin(cls_sel)) &
             (df["Variável"].isin(var_sel))]

if df_filt.empty:
    st.warning("Filtros retornaram zero linhas.")
    st.stop()

# Normalização se necessário
if view_mode in ["Normalizado", "Ambos"]:
    df_norm = normalizar_df(df_filt, estat_cols)
else:
    df_norm = pd.DataFrame()

# Tabs -------------------------------------------------------------------
aba_metricas, aba_univ, aba_stats = st.tabs(["📊 Métricas", "🏷️ Univariadas", "📐 Estatísticas"])

with aba_metricas:
    metodo_radio = st.radio("Filtrar método:", ["Todos"] + met_sel, horizontal=True)
    for est in est_sel:
        st.header(f"Estatística: {est}")
        for mode, data in [("Escala Real", df_filt), ("Normalizado", df_norm)]:
            if view_mode in [mode, "Ambos"] and not data.empty:
                st.subheader(mode)
                col1, col2 = st.columns(2)
                with col1:
                    d = data if metodo_radio == "Todos" else data[data["Método"] == metodo_radio]
                    st.plotly_chart(plot_barras(d, est), use_container_width=True)
                with col2:
                    d = data if metodo_radio == "Todos" else data[data["Método"] == metodo_radio]
                    st.plotly_chart(plot_radar(d, est, d["Método"].unique(), cls_sel), use_container_width=True)

with aba_univ:
    estat_univ = st.selectbox("Estatística para univariadas:", estat_cols)
    plot_univariadas(df_filt, estat_univ, group_col_sel)

with aba_stats:
    analise_estatistica_variavel(group_col_sel)

# Download ---------------------------------------------------------------
st.markdown("---")
st.subheader("Tabela filtrada")
st.dataframe(df_filt, use_container_width=True)

csv_bytes = df_filt.to_csv(index=False).encode()
st.download_button("⬇️ Baixar CSV filtrado", csv_bytes, file_name="metricas_filtradas.csv")
