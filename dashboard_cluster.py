
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from pathlib import Path
import os



st.set_page_config(page_title="Análise de Clusters", layout="wide")

st.title("Dashboard de Análise de Clusters por Arquivo, Método e Classe")

BASE_DIR = Path(__file__).parent          # raiz do repo no deploy
PASTA_DADOS    = BASE_DIR / "data" / "metricas"
PASTA_ANALISES = BASE_DIR / "data" / "merged"

@st.cache_data
def carregar_todos_arquivos(pasta: Path):
    arquivos = {}
    for csv in pasta.rglob("*.csv"):      # percorre recursivamente
        try:
            df = pd.read_csv(csv)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            arquivos[csv.name] = df       # chave = só o nome do arquivo
        except Exception as e:
            st.warning(f"Erro em {csv}: {e}")
    return arquivos

def normalizar_df(df, estatisticas):
    df_norm = df.copy()
    for stat in estatisticas:
        if stat in df_norm.columns:
            for var in df_norm['Variável'].unique():
                mask = df_norm['Variável'] == var
                min_val = df_norm.loc[mask, stat].min()
                max_val = df_norm.loc[mask, stat].max()
                if min_val != max_val:
                    df_norm.loc[mask, stat] = (df_norm.loc[mask, stat] - min_val) / (max_val - min_val)
    return df_norm

def plot_barras(df, stat_col):
    fig = px.bar(df, x="Variável", y=stat_col,
                 color="Classe", barmode="group",
                 facet_col="Método", facet_col_wrap=2,
                 title=f"{stat_col.capitalize()} por Variável",
                 labels={stat_col: stat_col.capitalize(), "Variável": "Variável"},
                 height=500)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

def plot_radar(df, stat_col, metodos, classes):
    fig = go.Figure()
    for classe in classes:
        for metodo in metodos:
            df_tmp = df[(df["Classe"] == classe) & (df["Método"] == metodo)]
            if not df_tmp.empty:
                fig.add_trace(go.Scatterpolar(
                    r=df_tmp[stat_col],
                    theta=df_tmp["Variável"],
                    fill='toself',
                    name=f"{metodo} - Cluster {classe}"
                ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=600,
        title=f"Radar Chart - {stat_col.capitalize()} por Variável"
    )
    return fig

def plot_univariadas(df, estatistica):
    st.markdown("### Análises Univariadas")
    for var in sorted(df["Variável"].unique()):
        st.markdown(f"#### Variável: {var}")
        df_var = df[df["Variável"] == var]
        col1, col2 = st.columns(2)
        with col1:
            fig_dist = px.histogram(
                df_var, x=estatistica, color="Classe",
                title=f"Distribuição da {estatistica.capitalize()} - {var}",
                marginal="box",
                nbins=min(10, max(2, df_var.shape[0] // 2))
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            if df_var[estatistica].nunique() > 1:
                fig_box = px.box(df_var, x="Classe", y=estatistica,
                                 color="Classe",
                                 title=f"Boxplot da {estatistica.capitalize()} - {var}")
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning(f"Boxplot indisponível para {var} - valores constantes.")

def analise_estatistica_variavel():
    st.markdown("### Análise Estatística por Variável (ANOVA, Histogramas, Boxplots)")
    arquivos_merged = carregar_todos_arquivos(PASTA_ANALISES)
    arquivo_merged = st.selectbox("Selecione o arquivo da variável:", list(arquivos_merged.keys()), key="merged_file")
    df_var = arquivos_merged[arquivo_merged]
    colunas_num = df_var.select_dtypes(include=['float64', 'int64']).columns.tolist()
    col_classe = st.selectbox("Coluna de agrupamento (classe):", df_var.columns, index=0)
    col_var = st.selectbox("Variável numérica para análise:", colunas_num, index=1)

    st.markdown("#### ANOVA")
    grupos = [grupo[col_var].dropna().values for nome, grupo in df_var.groupby(col_classe)]
    if len(grupos) > 1:
        f_stat, p_value = stats.f_oneway(*grupos)
        st.write(f"Estatística F: {f_stat:.4f}, Valor-p: {p_value:.4f}")
        st.success("Diferença significativa encontrada." if p_value < 0.05 else "Sem diferença significativa.")
    else:
        st.warning("Não há grupos suficientes para ANOVA.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Histograma por grupo")
        fig_hist = px.histogram(df_var, x=col_var, color=col_classe, marginal="box", nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        st.markdown("#### Boxplot por grupo")
        fig_box = px.box(df_var, x=col_classe, y=col_var, color=col_classe)
        st.plotly_chart(fig_box, use_container_width=True)

bases = carregar_todos_arquivos(PASTA_DADOS)

if not bases:
    st.error("Nenhum arquivo CSV encontrado na pasta especificada.")
    st.stop()

arquivo_selecionado = st.selectbox("Selecione o arquivo de métricas:", list(bases.keys()))
df = bases[arquivo_selecionado]

metodos = sorted(df["Método"].unique())
classes = sorted(df["Classe"].unique())
variaveis_disponiveis = sorted(df["Variável"].unique())
estatisticas = ["média", "mediana", "desvio_relativo_média", "mínimo", "máximo", "primeiro_quartil", "terceiro_quartil"]

with st.sidebar:
    metodos_selecionados = st.multiselect("Métodos de Clusterização:", metodos, default=metodos[:1])
    classes_selecionadas = st.multiselect("Clusters:", classes, default=classes[:1])
    variaveis_selecionadas = st.multiselect("Variáveis:", variaveis_disponiveis, default=variaveis_disponiveis)
    estatisticas_selecionadas = st.multiselect("Estatísticas:", estatisticas, default=["média"])
    modo_visualizacao = st.radio("Visualização:", ["Escala Real", "Normalizado", "Ambos"], horizontal=True)
    exibir_referencias = st.checkbox("Exibir linha de referência da média geral", value=True)
    estatistica_univariada = st.selectbox("Estatística para análise univariada:", estatisticas, index=0)

# Filtro principal
df_filtrado = df[(df["Método"].isin(metodos_selecionados)) &
                 (df["Classe"].isin(classes_selecionadas)) &
                 (df["Variável"].isin(variaveis_selecionadas))]

if modo_visualizacao in ["Normalizado", "Ambos"] and not df_filtrado.empty:
    df_normalizado = normalizar_df(df_filtrado, estatisticas)
else:
    df_normalizado = pd.DataFrame()

if modo_visualizacao in ["Escala Real", "Ambos"]:
    df_nao_normalizado = df_filtrado
else:
    df_nao_normalizado = pd.DataFrame()

if not df_filtrado.empty:
    metodo_radio = st.radio("Filtrar método no gráfico: ", ["Todos"] + metodos_selecionados, horizontal=True)

    for stat_col in estatisticas_selecionadas:
        st.markdown(f"### Estatística: {stat_col.capitalize()}")

        if modo_visualizacao in ["Escala Real", "Ambos"]:
            st.markdown("#### Visualização em Escala Real")
            col1, col2 = st.columns(2)
            with col1:
                df_bar = df_nao_normalizado.copy()
                if metodo_radio != "Todos":
                    df_bar = df_bar[df_bar["Método"] == metodo_radio]
                st.plotly_chart(plot_barras(df_bar, stat_col), use_container_width=True)
            with col2:
                radar_data = df_nao_normalizado.copy()
                if metodo_radio != "Todos":
                    radar_data = radar_data[radar_data["Método"] == metodo_radio]
                st.plotly_chart(plot_radar(radar_data, stat_col, radar_data['Método'].unique(), classes_selecionadas), use_container_width=True)

        if modo_visualizacao in ["Normalizado", "Ambos"]:
            st.markdown("#### Visualização Normalizada")
            col1, col2 = st.columns(2)
            with col1:
                df_bar = df_normalizado.copy()
                if metodo_radio != "Todos":
                    df_bar = df_bar[df_bar["Método"] == metodo_radio]
                st.plotly_chart(plot_barras(df_bar, stat_col), use_container_width=True)
            with col2:
                radar_data = df_normalizado.copy()
                if metodo_radio != "Todos":
                    radar_data = radar_data[radar_data["Método"] == metodo_radio]
                st.plotly_chart(plot_radar(radar_data, stat_col, radar_data['Método'].unique(), classes_selecionadas), use_container_width=True)

        st.markdown(f"#### Resumo Estatístico")
        resumo = df_filtrado[df_filtrado["Variável"].isin(variaveis_selecionadas)]
        resumo_pivot = resumo.pivot_table(index=["Classe", "Método"],
                                          columns="Variável",
                                          values=stat_col)
        st.dataframe(resumo_pivot, use_container_width=True)

    # Gráficos Univariados
    st.markdown("---")
    plot_univariadas(df_filtrado, estatistica_univariada)

    # Análises estatísticas adicionais
    st.markdown("---")
    analise_estatistica_variavel()
else:
    st.warning("Nenhum dado disponível para os filtros selecionados.")

st.subheader("Tabela de Métricas Filtradas")
st.dataframe(df_filtrado, use_container_width=True)

df_csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button("Baixar dados filtrados CSV", df_csv, file_name="metricas_comparativas.csv")
