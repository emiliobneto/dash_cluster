import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import numpy as np
from pathlib import Path
from itertools import combinations
import statsmodels.stats.multitest as smm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List


try:
    import geopandas as gpd  # type: ignore
    _GPD_AVAILABLE = True
    _GPD_ERR_MSG = ""
except Exception as e:
    gpd = None  # type: ignore
    _GPD_AVAILABLE = False
    _GPD_ERR_MSG = str(e)

try:
    import pyogrio  # acelera leitura de GPKG se presente
    _HAVE_PYOGRIO = True
except Exception:
    _HAVE_PYOGRIO = False

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_AVAILABLE = True
except Exception:
    folium = None  # type: ignore
    st_folium = None  # type: ignore
    _FOLIUM_AVAILABLE = False


# ───────────────────────── Configuração global ─────────────────────────
st.set_page_config(
    page_title="Dashboard de Análise de Clusters para o Município de São Paulo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] {font-family:'Roboto',sans-serif;}
        #MainMenu, footer {visibility:hidden;}
        h1 {
            font-size:84px;
            font-weight:700;
            color:#C65534 !important;
            margin:0;
            line-height:1.05;
        }
        h2 {
            font-size:64px;
            font-weight:700;
            color:#C65534 !important;
            margin:6px 0 0 0;
        }
        .legend-box {display:flex; gap:8px; align-items:center; margin-right:16px;}
        .legend-color {width:14px; height:14px; border-radius:3px; display:inline-block; border:1px solid #bbb;}
        .legend-row {display:flex; flex-wrap:wrap; gap:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Cabeçalho ------------------------------------
BASE_DIR = Path(__file__).parent
logo_path = BASE_DIR / "data" / "assets" / "logo_dash.jpg"
col_logo, col_title = st.columns([2, 6])
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=160)
with col_title:
    st.markdown(
        """
        <h1>DASHBOARD DE ANÁLISE DE CLUSTERS PARA O MUNICÍPIO DE SÃO PAULO</h1>
        """,
        unsafe_allow_html=True,
    )

# ───────────────────────── Constantes ─────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
CLASSE_CORES = {0:'#F4DD63',1:'#B1BF7C',2:'#D58243',3:'#C65534',4:'#6FA097',5:'#14407D'}
GROUP_COLS = ["KMeans_k5","Spectral_k5","KMedoids_k5"]
PASTA_DADOS = BASE_DIR/"data"/"metricas"
PASTA_ANALISES = BASE_DIR/"data"/"merged"
PASTA_MAPA = BASE_DIR / "dash_cluster" / "mapa"

# Mapeamento de cores para a camada de clusters (GeoPackage)
MAP_CLUSTER_CORES = {
    'Periférico de Alta Densidade Populacional': '#bf7db2',
    'Residencial de Médio Padrão': '#f7bd6a',
    'Periférico de Média Densidade': '#cf651f',
    'Vertical de Uso Misto': '#ede4e6',
    'Comércio e Serviços': '#793393',
}

# ───────────────────────── Utilidades ─────────────────────────
@st.cache_data(show_spinner=False)
def carregar_todos_arquivos(pasta: Path):
    arquivos = {}
    for csv in pasta.rglob("*.csv"):
        try:
            df = pd.read_csv(csv)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            arquivos[csv.name] = df
        except Exception as e:
            st.warning(f"Erro ao carregar {csv.name}: {e}")
    return arquivos

@st.cache_data(show_spinner=False)
def carregar_geopackages(pasta: Path):
    """Lê todos os .gpkg na pasta e retorna dict nome->GeoDataFrame."""
    gdfs = {}
    if not _GPD_AVAILABLE:
        st.error(
            "GeoPandas indisponível. Erro: %s\nInclua geopandas, shapely, pyproj e (opcional) pyogrio no requirements.txt."
            % _GPD_ERR_MSG
        )
        return gdfs
    if not pasta.exists():
        return gdfs
    prefer_engine = "pyogrio" if _HAVE_PYOGRIO else None
    for gpkg in sorted(pasta.glob("*.gpkg")):
        try:
            if prefer_engine:
                gdf = gpd.read_file(gpkg, engine=prefer_engine)
            else:
                gdf = gpd.read_file(gpkg)
            gdfs[gpkg.stem.lower()] = gdf
        except Exception as e:
            st.warning(f"Erro ao carregar {gpkg.name}: {e}")
    return gdfs

@st.cache_data(show_spinner=False)
def normalizar_df(df, est_cols):
    df_n = df.copy()
    for est in est_cols:
        if est in df_n.columns:
            for var in df_n['Variável'].unique():
                mask = df_n['Variável'] == var
                mn, mx = df_n.loc[mask, est].min(), df_n.loc[mask, est].max()
                if mn != mx:
                    df_n.loc[mask, est] = (df_n.loc[mask, est] - mn) / (mx - mn)
    return df_n

def montar_matriz_pca(df_met: pd.DataFrame, stat_col: str, group_cols: List[str]) -> tuple[pd.DataFrame, bool]:
    """
    Constrói a matriz 'wide' para PCA a partir de df_met.
    Retorna (wide, is_long):
      • is_long=True  -> df está em formato longo e foi feito pivot por 'Variável'
      • is_long=False -> df já está em formato wide (usa colunas numéricas)
    """
    tem_variavel = "Variável" in df_met.columns
    tem_stat = stat_col in df_met.columns

    if tem_variavel and tem_stat:
        wide = (
            df_met.assign(_obs=df_met.index)
                  .pivot_table(index="_obs", columns="Variável", values=stat_col)
        )
        return wide, True

    # fallback: formato wide — usa todas as colunas numéricas que não são grupo
    drop_cols = set(group_cols + ["Classe", "Variável"])
    num_cols = [c for c in df_met.columns
                if c not in drop_cols and pd.api.types.is_numeric_dtype(df_met[c])]
    wide = df_met[num_cols].copy()
    return wide, False


def rodar_pca(wide: pd.DataFrame, max_components: int = 3):
    """
    Executa PCA em 'wide' (linhas=observações, colunas=features).
    Retorna (pc_df, pca, err_msg). Se houver problema, pc_df/pca= None e err_msg com texto.
    """
    if wide is None or wide.empty:
        return None, None, "Matriz vazia para PCA."
    if wide.shape[1] < 2 or wide.shape[0] < 2:
        return None, None, "Dados insuficientes (precisa de ≥2 colunas e ≥2 linhas)."

    # padronização
    X_std = StandardScaler().fit_transform(wide.values)
    ncomp = min(max_components, wide.shape[1], wide.shape[0])
    pca = PCA(n_components=ncomp)
    pcs = pca.fit_transform(X_std)

    cols = [f"PC{i+1}" for i in range(ncomp)]
    pc_df = pd.DataFrame(pcs, index=wide.index, columns=cols)
    return pc_df, pca, None


def pairwise_t_matrix_safe(df_vals: pd.DataFrame, grupo_col: str, valor_col: str, method: str | None = "bonferroni"):
    """
    Versão robusta do pairwise t-test (Welch), com checagens.
    Retorna (matriz | None, err_msg | None).
    """
    if grupo_col not in df_vals.columns or valor_col not in df_vals.columns:
        return None, f"Colunas ausentes: grupo='{grupo_col}' ou valor='{valor_col}'."

    counts = df_vals.groupby(grupo_col)[valor_col].count()
    clusters = counts[counts >= 2].index.tolist()
    if len(clusters) < 2:
        return None, "Grupos insuficientes para pairwise (precisa de ≥2 grupos com ≥2 observações)."

    mat = pd.DataFrame(np.nan, index=clusters, columns=clusters)
    pvals, pairs = [], []

    for c1, c2 in combinations(clusters, 2):
        a = df_vals.loc[df_vals[grupo_col] == c1, valor_col].dropna()
        b = df_vals.loc[df_vals[grupo_col] == c2, valor_col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        pvals.append(p)
        pairs.append((c1, c2))

    if not pvals:
        return None, "Sem pares válidos para o teste t (amostras muito pequenas)."

    if method:
        pvals = smm.multipletests(pvals, method=method)[1]

    for (c1, c2), p in zip(pairs, pvals):
        mat.loc[c1, c2] = mat.loc[c2, c1] = p

    return mat, None


def render_pairwise_por_variavel(df_ana: pd.DataFrame, metodo_col: str, estat_cols_ana: List[str]):
    """
    UI segura para pairwise por variável.
    • Só aparece se houver coluna 'Variável'
    """
    if "Variável" not in df_ana.columns:
        st.info("O arquivo selecionado em data/merged **não** possui coluna 'Variável'. "
                "A seção 'Pairwise por variável' fica indisponível para este arquivo.")
        return

    var_ana = sorted(df_ana["Variável"].dropna().unique())
    if not var_ana:
        st.info("Sem valores em 'Variável' após os filtros.")
        return

    var_pair = st.selectbox("Variável:", var_ana, key="var_pair")
    estat_pair = st.selectbox("Estatística:", estat_cols_ana, key="estat_pair")
    corr_var = st.radio("Correção múltiplos testes:", ["bonferroni", "fdr_bh", "nenhuma"],
                        key="corr_var", horizontal=True)
    meth_var = None if corr_var == "nenhuma" else corr_var

    df_pair = df_ana[df_ana["Variável"] == var_pair]
    grp_ok = metodo_col if metodo_col in df_pair.columns else "Classe"
    mat_var, err = pairwise_t_matrix_safe(df_pair, grp_ok, estat_pair, meth_var)

    if err:
        st.warning(err)
        return

    if st.radio("Visualização matriz:", ["Tabela", "Heatmap"], key="view_var", horizontal=True) == "Tabela":
        st.dataframe(mat_var.style.format("{:.3e}"), use_container_width=True)
    else:
        st.plotly_chart(
            px.imshow(mat_var, text_auto=".2e", zmin=0, zmax=0.05,
                      color_continuous_scale="RdBu_r", title=f"Pairwise – {var_pair}"),
            use_container_width=True,
        )
    
# ───────────────────────── Carregamento Inicial ─────────────────────────
@st.cache_data(show_spinner=False)
def ler_csv(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    return df.loc[:, ~df.columns.str.contains(r'^Unnamed')]

# MÉTRICAS (data/metricas)
metric_paths = sorted(PASTA_DADOS.rglob("*.csv"))
if not metric_paths:
    st.error("Nenhum CSV encontrado em data/metricas.")
    st.stop()

metric_names = [p.name for p in metric_paths]
sel_metric = st.selectbox("Selecione o arquivo de métricas:", metric_names, key="sel_metric_top")
df = ler_csv(str(metric_paths[metric_names.index(sel_metric)]))

# ANALISES MERGED (data/merged) – usado em PCA/pairwise
merged_paths = sorted(PASTA_ANALISES.rglob("*.csv"))
df_ana = None
if merged_paths:
    merged_names = [p.name for p in merged_paths]
    sel_merge = st.selectbox("Arquivo para PCA / pairwise:", merged_names, key="sel_merged_stats")
    df_ana = ler_csv(str(merged_paths[merged_names.index(sel_merge)]))
else:
    st.warning("Nenhum CSV em data/merged para análises estatísticas/pairwise.", icon="⚠️")

# Listas derivadas
metodos = sorted(df['Método'].unique())
classes = sorted(df['Classe'].unique())
variaveis = sorted(df['Variável'].unique())
estat_cols = [c for c in df.columns if c not in ['Método', 'Classe', 'Variável']]

# ───────────────────────── Estatísticas auxiliares ─────────────────────────
SIG_BINS = [-np.inf, 0.001, 0.01, 0.05, 1]
SIG_LABELS = ["***", "**", "*", "ns"]


def quadro_resumo_long(df_long: pd.DataFrame, grupo_col: str, variaveis: List[str], col_estat: str) -> pd.DataFrame:
    agg_dict = {
        "n": "count",
        "mean": "mean",
        "std": "std",
        "min": "min",
        "25%": lambda s: s.quantile(0.25),
        "median": "median",
        "75%": lambda s: s.quantile(0.75),
        "max": "max",
    }
    linhas = []
    clusters = sorted(df_long[grupo_col].unique())

    for var in variaveis:
        sub = df_long[df_long["Variável"] == var]
        agg_named = {nome: (col_estat, func) for nome, func in agg_dict.items()}
        stats_df = (
            sub.groupby(grupo_col).agg(**agg_named).T
        )
        row = {(estat, c): stats_df.loc[estat, c]
               for estat in stats_df.index
               for c in stats_df.columns}
        grupos = [sub.loc[sub[grupo_col] == c, col_estat].dropna() for c in clusters]
        p_val = stats.f_oneway(*grupos)[1] if len(grupos) >= 2 and all(len(g) > 1 for g in grupos) else np.nan
        row["p_value"] = p_val
        row["signif"] = (pd.cut([p_val], SIG_BINS, labels=SIG_LABELS).astype(str)[0] if not np.isnan(p_val) else "na")
        linhas.append(pd.Series(row, name=var))

    return pd.DataFrame(linhas)


def pairwise_t_matrix(df: pd.DataFrame, grupo_col: str, estat_col: str, method: str = "bonferroni") -> pd.DataFrame:
    counts = df.groupby(grupo_col)[estat_col].count()
    clusters = counts[counts >= 2].index.tolist()
    mat = pd.DataFrame(np.nan, index=clusters, columns=clusters)

    pvals, pairs = [], []
    for c1, c2 in combinations(clusters, 2):
        a = df.loc[df[grupo_col] == c1, estat_col].dropna()
        b = df.loc[df[grupo_col] == c2, estat_col].dropna()
        _, p = stats.ttest_ind(a, b, equal_var=False)
        pvals.append(p)
        pairs.append((c1, c2))

    if method and pvals:
        pvals = smm.multipletests(pvals, method=method)[1]

    for (c1, c2), p in zip(pairs, pvals):
        mat.loc[c1, c2] = mat.loc[c2, c1] = p

    return mat


def filtrar_outliers_iqr(df: pd.DataFrame, valor_col: str, agrupadores: List[str]) -> pd.DataFrame:
    def _apply(gr):
        q1 = gr[valor_col].quantile(0.25)
        q3 = gr[valor_col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return gr[(gr[valor_col] >= lo) & (gr[valor_col] <= hi)]

    return (
        df.groupby(agrupadores, group_keys=False)
        .apply(_apply)
        .reset_index(drop=True)
    )

# ───────────────────────── Funções de gráfico ─────────────────────────

def plot_barras(df, est):
    fig = px.bar(
        df,
        x='Variável',
        y=est,
        color='Classe',
        color_discrete_map=CLASSE_CORES,
        barmode='group',
        facet_col='Método',
        facet_col_wrap=2,
        template=PLOTLY_TEMPLATE,
        height=480,
        title=f"{est.capitalize()} por Variável",
    )
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

# 💣 REMOVIDO: plot_radar – conforme solicitação


def plot_univariadas(df, est, grp_requested):
    grp = grp_requested if grp_requested in df.columns else "Classe"
    if grp != grp_requested:
        st.info(f"Coluna '{grp_requested}' não existe neste arquivo – usando '{grp}'.")

    st.markdown("### Análises Univariadas")
    for var in sorted(df["Variável"].unique()):
        st.subheader(f"Variável: {var}")
        dv = df[df["Variável"] == var]
        grupos = [g[est].dropna() for _, g in dv.groupby(grp)]
        nmin = min(len(g) for g in grupos)

        c1, c2 = st.columns(2)
        with c1:
            if nmin <= 15:
                fg = px.strip(
                    dv, x=grp, y=est, color=grp, stripmode="overlay",
                    color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE,
                    title="Valores Individuais",
                )
                fg.update_traces(jitter=0.35, marker_size=8)
            else:
                fg = px.histogram(
                    dv, x=est, color=grp,
                    nbins=min(20, max(5, dv.shape[0] // 3)),
                    marginal="rug", color_discrete_map=CLASSE_CORES,
                    template=PLOTLY_TEMPLATE, title="Distribuição",
                )
            st.plotly_chart(fg, use_container_width=True)
        with c2:
            vg = px.violin(
                dv, x=grp, y=est, color=grp, box=True, points="all",
                color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE,
                title="Violin + Box",
            )
            st.plotly_chart(vg, use_container_width=True)

        st.markdown("#### Testes estatísticos")
        if len(grupos) == 2:
            t_stat, p_val = stats.ttest_ind(*grupos, equal_var=False)
            st.write(f"**t‑Student (duas amostras, variâncias não iguais)** → *t* = {t_stat:.4f}, *p* = {p_val:.4f}")
            n1, n2 = len(grupos[0]), len(grupos[1])
            rss = sum((grupos[0] - grupos[0].mean())**2) + sum((grupos[1] - grupos[1].mean())**2)
            n = n1 + n2
            k = 2
            aic = n * np.log(rss / n) + 2 * k
            st.write(f"AIC aproximado do modelo de 2 médias: {aic:.2f}")
        else:
            f_stat, p_anova = stats.f_oneway(*grupos)
            st.write(f"**ANOVA** → *F* = {f_stat:.4f}, *p* = {p_anova:.4f}")
        if len(grupos) >= 2:
            h_stat, p_kw = stats.kruskal(*grupos)
            st.write(f"**Kruskal‑Wallis** (não paramétrico) → *H* = {h_stat:.4f}, *p* = {p_kw:.4f}")

        resumo = (
            dv.groupby(grp)[est]
            .agg(n="count", média="mean", mediana="median", mín="min", máx="max", desvio="std")
            .round(2)
            .reset_index()
        )
        st.dataframe(resumo, use_container_width=True)


# ───────────────────────── ABA: VISUALIZAÇÕES (por sessão) ─────────────────────────
st.markdown("## 📊 Métricas (Sessões Independentes)")

n_sessoes = st.number_input("Número de sessões de visualização:", min_value=1, max_value=3, value=2, step=1)
sessoes_tabs = st.tabs([f"Sessão {i+1}" for i in range(n_sessoes)])

for i, tab in enumerate(sessoes_tabs, start=1):
    with tab:
        st.markdown("#### Filtros desta sessão")
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.8, 1.2])
        with c1:
            met_sel = st.multiselect(
                "Métodos:", metodos, default=metodos, key=f"met_sel_{i}"
            )
        with c2:
            cls_sel = st.multiselect(
                "Classes:", classes, default=classes, key=f"cls_sel_{i}"
            )
        with c3:
            var_sel = st.multiselect(
                "Variáveis:", variaveis, default=variaveis, key=f"var_sel_{i}"
            )
        with c4:
            est_sel = st.multiselect(
                "Estatísticas:", estat_cols, default=[estat_cols[0]] if estat_cols else [], key=f"est_sel_{i}"
            )

        view_mode = st.radio(
            "Visualização:", ["Escala Real", "Normalizado", "Ambos"], index=0, key=f"view_mode_{i}", horizontal=True
        )

        df_filt = df[(df["Método"].isin(met_sel)) & (df["Classe"].isin(cls_sel)) & (df["Variável"].isin(var_sel))]
        if df_filt.empty:
            st.warning("Filtros desta sessão retornaram zero linhas.")
            continue

        df_norm = normalizar_df(df_filt, estat_cols) if view_mode in ["Normalizado", "Ambos"] else pd.DataFrame()

        metodo_radio = st.radio("Filtrar método (opcional):", ["Todos"] + met_sel, horizontal=True, key=f"metodo_radio_{i}")

        # Gráficos
        for est in est_sel:
            st.subheader(f"Estatística: {est}")
            for mode, data in [("Escala Real", df_filt), ("Normalizado", df_norm)]:
                if view_mode in [mode, "Ambos"] and not data.empty:
                    st.caption(mode)
                    data_use = data if metodo_radio == "Todos" else data[data["Método"] == metodo_radio]
                    st.plotly_chart(plot_barras(data_use, est), use_container_width=True)

            # Tabela resumida
            st.markdown("**Tabela (clusters × variáveis)**")
            pivot = df_filt.pivot_table(index="Classe", columns="Variável", values=est)
            st.dataframe(pivot, use_container_width=True)

        csv_bytes = df_filt.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Baixar CSV desta sessão", csv_bytes, file_name=f"metricas_filtradas_sessao_{i}.csv", mime="text/csv",
            key=f"download_filtrado_metricas_{i}"
        )

# ───────────────────────── ABA: UNIVARIADAS ─────────────────────────
with st.expander("🏷️ Univariadas • clique para abrir", expanded=False):
    grp_sel = st.selectbox("Agrupamento (coluna de cluster):", GROUP_COLS + ["Classe"], index=len(GROUP_COLS), key="grp_sel_uni")

    met_uni = st.multiselect("Métodos:", metodos, default=metodos, key="met_uni")
    cls_uni = st.multiselect("Classes:", classes, default=classes, key="cls_uni")
    var_univ = st.selectbox("Variável:", variaveis, key="var_univ")
    estat_univ = st.selectbox("Estatística:", estat_cols, key="estat_univ")

    df_uni_base = df[(df["Método"].isin(met_uni)) & (df["Classe"].isin(cls_uni))]
    grp_active = grp_sel if grp_sel in df_uni_base.columns else "Classe"
    cls_options = sorted(df_uni_base[df_uni_base["Variável"] == var_univ][grp_active].unique())
    cls_pick = st.multiselect("Clusters a incluir:", cls_options, default=cls_options)
    df_uni = df_uni_base[(df_uni_base["Variável"] == var_univ) & (df_uni_base[grp_active].isin(cls_pick))]

    if df_uni.empty:
        st.warning("Nada para mostrar – verifique filtros.")
    else:
        plot_univariadas(df_uni, estat_univ, grp_active)

# ───────────────────────── ABA: ESTATÍSTICAS (globais e pairwise/PCA) ─────────────────────────
st.markdown("---")
st.markdown("## 📐 Estatísticas")

if df_ana is None or df_ana.empty:
    st.caption("Para liberar a aba Estatísticas, selecione um arquivo em **data/merged** no seletor do topo.")
else:
    tab_global, tab_t = st.tabs(["Testes globais", "t-Student & PCA"])

    with tab_global:
        estat_ref = st.selectbox("Estatística de referência:", estat_cols, key="estat_ref_global") if estat_cols else None
        if estat_ref is None:
            st.info("Não há estatísticas numéricas no arquivo de métricas selecionado.")
        else:
            grp_global = st.selectbox("Agrupamento:", GROUP_COLS + ["Classe"], index=len(GROUP_COLS), key="grp_global")
            var_sel_global = st.multiselect("Variáveis:", variaveis, default=variaveis, key="var_sel_global")

            df_filt_global = df[(df["Variável"].isin(var_sel_global))]
            grp_ok = grp_global if grp_global in df_filt_global.columns else "Classe"
            df_clean = filtrar_outliers_iqr(df_filt_global, estat_ref, ["Variável", grp_ok])
            tab_res = quadro_resumo_long(df_clean, grp_ok, var_sel_global, estat_ref)

            st.caption(f"Outliers removidos: {len(df_filt_global) - len(df_clean)} linhas")
            st.dataframe(tab_res.style.format({"p_value": "{:.3e}"}), use_container_width=True)

    with tab_t:
        metodo_pca = st.selectbox("Algoritmo de cluster:", GROUP_COLS, key="metodo_pca")
        if metodo_pca not in df_ana.columns:
            st.error(f"Coluna '{metodo_pca}' não existe no arquivo selecionado.")
        else:
            df_met = df_ana.copy()
            df_met["Classe"] = df_met[metodo_pca]
            cls_opts = sorted(df_met["Classe"].dropna().unique())
            cls_sel_pca = st.multiselect("Clusters (PCA):", cls_opts, default=cls_opts, key="cls_pca")
            df_met = df_met[df_met["Classe"].isin(cls_sel_pca)].reset_index(drop=True)

            estat_cols_ana = [c for c in df_ana.columns if c not in ["Variável"] + GROUP_COLS]
            if not estat_cols_ana:
                st.error("Nenhuma coluna numérica disponível para PCA no arquivo escolhido.")
            else:
                stat_col = st.selectbox("Estatística PCA:", estat_cols_ana, key="stat_pca")

                # ===== 2) Montagem robusta da matriz para PCA (aceita long ou wide) =====
                tem_variavel = ("Variável" in df_met.columns) and (stat_col in df_met.columns)
                if tem_variavel:
                    # formato LONGO -> pivota para WIDE
                    wide = (
                        df_met.assign(_obs=df_met.index)
                              .pivot_table(index="_obs", columns="Variável", values=stat_col)
                    )
                    is_long = True
                else:
                    # formato WIDE -> usa todas numéricas fora das colunas de grupo
                    drop_cols = set(GROUP_COLS + ["Classe", "Variável"])
                    num_cols = [c for c in df_met.columns
                                if c not in drop_cols and pd.api.types.is_numeric_dtype(df_met[c])]
                    wide = df_met[num_cols].copy()
                    is_long = False

                # Se veio de LONGO, opcionalmente filtramos pelas PCA_VARS (se existirem)
                if is_long:
                    PCA_VARS = [
                        "comp_res", "outros_usos", "fator_com",
                        "densidade_hec_norm", "mobilidade_norm", "equipamentos_norm",
                        "a_vl_m2_construcao_norm", "comp_res_norm",
                        "outros_usos_norm", "fator_com_norm"
                    ]
                    vars_disp = [v for v in PCA_VARS if v in wide.columns]
                    wide = wide[vars_disp].dropna(how="any") if vars_disp else pd.DataFrame()
                else:
                    wide = wide.dropna(how="any")

                # ===== PCA =====
                if wide.empty or wide.shape[1] < 2 or wide.shape[0] < 2:
                    st.warning("Dados insuficientes para PCA depois dos filtros/transformações.")
                else:
                    X_std = StandardScaler().fit_transform(wide.values)
                    ncomp = min(3, wide.shape[1], wide.shape[0])
                    pca = PCA(n_components=ncomp)
                    pcs = pca.fit_transform(X_std)

                    pc_cols = [f"PC{i+1}" for i in range(ncomp)]
                    pc_df = pd.DataFrame(pcs, index=wide.index, columns=pc_cols)
                    pc_df["Classe"] = df_met.loc[pc_df.index, "Classe"].values

                    if "PC1" in pc_df.columns and "PC2" in pc_df.columns:
                        st.write(f"PC1 explica {pca.explained_variance_ratio_[0]:.1%}; "
                                 f"PC2 {pca.explained_variance_ratio_[1]:.1%}")
                        st.plotly_chart(
                            px.scatter(pc_df, x="PC1", y="PC2", color="Classe",
                                       template=PLOTLY_TEMPLATE, color_discrete_map=CLASSE_CORES,
                                       title="PCA – PC1 × PC2"),
                            use_container_width=True,
                        )
                    else:
                        st.info("PCA gerou menos de 2 componentes — gráfico PC1×PC2 indisponível.")

                    # Pairwise em PC1 (com checagens)
                    if "PC1" in pc_df.columns:
                        corr_pca = st.radio("Correção múltiplos testes (PC1):",
                                            ["bonferroni", "fdr_bh", "nenhuma"],
                                            key="corr_pca", horizontal=True)
                        meth_corr = None if corr_pca == "nenhuma" else corr_pca
                        mat_pc1, err_pw = pairwise_t_matrix_safe(pc_df, "Classe", "PC1", meth_corr)

                        if err_pw:
                            st.warning(err_pw)
                        else:
                            if st.radio("Visualização PC1:", ["Tabela", "Heatmap"],
                                        key="view_pc1", horizontal=True) == "Tabela":
                                st.dataframe(mat_pc1.style.format("{:.3e}"), use_container_width=True)
                            else:
                                st.plotly_chart(
                                    px.imshow(mat_pc1, text_auto=".2e", zmin=0, zmax=0.05,
                                              color_continuous_scale="RdBu_r", title="Pairwise – PC1"),
                                    use_container_width=True,
                                )

                # ===== 3) Pairwise por variável (apenas se existir a coluna 'Variável') =====
                st.markdown("---")
                if "Variável" not in df_ana.columns:
                    st.info("O arquivo selecionado em data/merged **não** possui coluna 'Variável'; "
                            "‘Pairwise por variável’ indisponível para este arquivo.")
                else:
                    var_ana = sorted(df_ana["Variável"].dropna().unique())
                    if not var_ana:
                        st.info("Sem valores em 'Variável' após os filtros.")
                    else:
                        var_pair = st.selectbox("Variável:", var_ana, key="var_pair")
                        estat_pair = st.selectbox("Estatística:", estat_cols_ana, key="estat_pair")
                        corr_var = st.radio("Correção múltiplos testes:",
                                            ["bonferroni", "fdr_bh", "nenhuma"],
                                            key="corr_var", horizontal=True)
                        meth_var = None if corr_var == "nenhuma" else corr_var

                        df_pair = df_ana[df_ana["Variável"] == var_pair]
                        grp_ok = metodo_pca if metodo_pca in df_pair.columns else "Classe"
                        mat_var, err = pairwise_t_matrix_safe(df_pair, grp_ok, estat_pair, meth_var)

                        if err:
                            st.warning(err)
                        else:
                            if st.radio("Visualização matriz:", ["Tabela", "Heatmap"],
                                        key="view_var", horizontal=True) == "Tabela":
                                st.dataframe(mat_var.style.format("{:.3e}"), use_container_width=True)
                            else:
                                st.plotly_chart(
                                    px.imshow(mat_var, text_auto=".2e", zmin=0, zmax=0.05,
                                              color_continuous_scale="RdBu_r", title=f"Pairwise – {var_pair}"),
                                    use_container_width=True,
                                )

# ───────────────────────── ABA: MAPA ─────────────────────────
st.markdown("---")
st.markdown("## 🗺️ Mapa (dash_cluster/mapa)")

if not _GPD_AVAILABLE or not _FOLIUM_AVAILABLE:
    st.warning("Recursos de mapa indisponíveis. Instale `geopandas`, `pyproj`, `shapely`, `pyogrio`, `folium` e `streamlit-folium` (veja requirements.txt).")
else:
    with st.expander("Carregar dados do mapa (.gpkg)", expanded=False):
        load_map = st.checkbox("Ler agora os arquivos da pasta 'dash_cluster/mapa'", value=False)

    # Lê os GPKGs apenas sob demanda (evita travar o boot)
    map_gdfs = carregar_geopackages(PASTA_MAPA) if load_map else {}

    # defaults seguros
    gdf_rios = gdf_parque = gdf_cluster = None

    if not map_gdfs:
        st.info("Nenhum .gpkg encontrado em 'dash_cluster/mapa'. Coloque **rios.gpkg**, **parque.gpkg** e **cluster.gpkg**.")
    else:
        # Obtém camadas (chaves minúsculas)
        gdf_rios    = map_gdfs.get('rios')
        gdf_parque  = map_gdfs.get('parque')
        gdf_cluster = map_gdfs.get('cluster')

        # Acha a coluna 'cluster' (case-insensitive)
        cluster_col = None
        if gdf_cluster is not None:
            for c in gdf_cluster.columns:
                if str(c).lower() == 'cluster':
                    cluster_col = c
                    break

        # Controles UI
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        with c1:
            show_rios   = st.checkbox("Mostrar rios", value=(gdf_rios is not None))
            show_parque = st.checkbox("Mostrar parques", value=(gdf_parque is not None))
        with c2:
            if gdf_cluster is not None and cluster_col:
                categorias = sorted([c for c in gdf_cluster[cluster_col].dropna().unique()])
                cats_sel = st.multiselect("Clusters a mostrar:", categorias, default=categorias)
            else:
                categorias, cats_sel = [], []
        with c3:
            basemap = st.selectbox("Base cartográfica:", ["CartoDB positron", "OpenStreetMap", "Stamen Terrain"], index=0)

        # Bounds/centro (WGS84)
        def _total_bounds(gdfs):
            mins, maxs = [], []
            for g in gdfs:
                if g is not None and not g.empty:
                    try:
                        x1, y1, x2, y2 = g.to_crs(4326).total_bounds
                    except Exception:
                        x1, y1, x2, y2 = g.total_bounds
                    mins.append((x1, y1)); maxs.append((x2, y2))
            if not mins:
                return (-46.75, -23.80, -46.45, -23.45)  # approx SP
            x1 = min(m[0] for m in mins); y1 = min(m[1] for m in mins)
            x2 = max(m[0] for m in maxs); y2 = max(m[1] for m in maxs)
            return (x1, y1, x2, y2)

        x1, y1, x2, y2 = _total_bounds([gdf_rios, gdf_parque, gdf_cluster])
        center = ((y1 + y2) / 2.0, (x1 + x2) / 2.0)

        m = folium.Map(location=center, zoom_start=12, tiles=None)
        if basemap == "CartoDB positron":
            folium.TileLayer("CartoDB positron").add_to(m)
        elif basemap == "Stamen Terrain":
            folium.TileLayer("Stamen Terrain").add_to(m)
        else:
            folium.TileLayer("OpenStreetMap").add_to(m)

        # Cores
        cor_rios   = "#BBD2EC"
        cor_parque = "#77942E"
        MAP_CLUSTER_CORES = {
            'Periférico de Alta Densidade Populacional': '#bf7db2',
            'Residencial de Médio Padrão':               '#f7bd6a',
            'Periférico de Média Densidade':             '#cf651f',
            'Vertical de Uso Misto':                     '#ede4e6',
            'Comércio e Serviços':                       '#793393',
        }

        # Rios
        if show_rios and gdf_rios is not None and not gdf_rios.empty:
            try:
                folium.GeoJson(
                    gdf_rios.to_crs(4326),
                    name="Rios",
                    style_function=lambda feat: {"color": cor_rios, "weight": 2, "opacity": 0.9},
                    tooltip=folium.features.GeoJsonTooltip(fields=[c for c in gdf_rios.columns if c != gdf_rios.geometry.name][:5]),
                ).add_to(m)
            except Exception as e:
                st.warning(f"Falha ao desenhar rios: {e}")

        # Parques
        if show_parque and gdf_parque is not None and not gdf_parque.empty:
            try:
                folium.GeoJson(
                    gdf_parque.to_crs(4326),
                    name="Parques",
                    style_function=lambda feat: {"color": cor_parque, "fillColor": cor_parque, "fillOpacity": 0.35, "weight": 1},
                    tooltip=folium.features.GeoJsonTooltip(fields=[c for c in gdf_parque.columns if c != gdf_parque.geometry.name][:5]),
                ).add_to(m)
            except Exception as e:
                st.warning(f"Falha ao desenhar parques: {e}")

        # Clusters
        if gdf_cluster is not None and not gdf_cluster.empty and cluster_col:
            gplot = gdf_cluster.copy()
            if cats_sel:
                gplot = gplot[gplot[cluster_col].isin(cats_sel)]
            try:
                def style_cluster(feat):
                    cat = feat['properties'].get(cluster_col)
                    col = MAP_CLUSTER_CORES.get(cat, '#999999')
                    return {"color": col, "fillColor": col, "fillOpacity": 0.45, "weight": 0.5}

                folium.GeoJson(
                    gplot.to_crs(4326),
                    name="Clusters",
                    style_function=style_cluster,
                    tooltip=folium.features.GeoJsonTooltip(fields=[cluster_col]),
                ).add_to(m)
            except Exception as e:
                st.warning(f"Falha ao desenhar clusters: {e}")

        folium.LayerControl(collapsed=False).add_to(m)

        # Legenda
        legend_html = "<div class='legend-row'>" + "".join(
            [f"<div class='legend-box'><span class='legend-color' style='background:{cor}'></span> {nome}</div>"
             for nome, cor in MAP_CLUSTER_CORES.items()]
        ) + "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        st_folium(m, use_container_width=True, returned_objects=[])

        # Tabelas/atributos
        with st.expander("Tabelas de atributos das camadas visíveis"):
            if show_rios and gdf_rios is not None:
                st.markdown("**Rios**")
                st.dataframe(gdf_rios.drop(columns=gdf_rios.geometry.name, errors='ignore').head(500), use_container_width=True)
            if show_parque and gdf_parque is not None:
                st.markdown("**Parques**")
                st.dataframe(gdf_parque.drop(columns=gdf_parque.geometry.name, errors='ignore').head(500), use_container_width=True)
            if gdf_cluster is not None and cluster_col:
                st.markdown("**Clusters**")
                gtmp = gdf_cluster if not cats_sel else gdf_cluster[gdf_cluster[cluster_col].isin(cats_sel)]
                st.dataframe(gtmp.drop(columns=gtmp.geometry.name, errors='ignore').head(1000), use_container_width=True)







