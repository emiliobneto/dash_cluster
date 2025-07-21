import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np
from pathlib import Path
import os
from itertools import combinations
import statsmodels.stats.multitest as smm
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

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
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] {font-family:'Roboto',sans-serif;}
        #MainMenu, footer {visibility:hidden;}
        h1 {
            font-size:84px;
            font-weight:700;
            color:#C65534 !important;  /* garante cor sobre o tema streamlit */
            margin:0;
            line-height:1.05;
        }
        h2 {
            font-size:64px;
            font-weight:700;
            color:#C65534 !important;
            margin:6px 0 0 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Cabeçalho ------------------------------------
logo_path = Path(__file__).parent / "data" / "assets" / "logo_dash.jpg"
col_logo, col_title = st.columns([2, 6])
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=160)
with col_title:
    # Usamos HTML direto para garantir aplicação do CSS
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
BASE_DIR = Path(__file__).parent
PASTA_DADOS = BASE_DIR/"data"/"metricas"
PASTA_ANALISES = BASE_DIR/"data"/"merged"

# ───────────────────────── Utilidades ─────────────────────────
@st.cache_data(show_spinner=False)
def carregar_todos_arquivos(pasta: Path):
    arquivos = {}
    for csv in pasta.rglob("*.csv"):
        try:
            df = pd.read_csv(csv)
            df = df.loc[:,~df.columns.str.contains('^Unnamed')]
            arquivos[csv.name] = df
        except Exception as e:
            st.warning(f"Erro ao carregar {csv.name}: {e}")
    return arquivos

def normalizar_df(df, est_cols):
    df_n = df.copy()
    for est in est_cols:
        if est in df_n.columns:
            for var in df_n['Variável'].unique():
                mask = df_n['Variável']==var
                mn, mx = df_n.loc[mask,est].min(), df_n.loc[mask,est].max()
                if mn!=mx:
                    df_n.loc[mask,est] = (df_n.loc[mask,est]-mn)/(mx-mn)
    return df_n

merged_files = carregar_todos_arquivos(PASTA_ANALISES)
if not merged_files:
    st.error("Nenhum CSV em data/merged.")
    st.stop()

arq_ana = st.sidebar.selectbox(
    "Arquivo para PCA / pairwise:", list(merged_files.keys()),
    key="sel_merged"
)
df_ana = merged_files[arq_ana]

# ───────────────────────── Estatísticas auxiliares ─────────────────────────
SIG_BINS   = [-np.inf, 0.001, 0.01, 0.05, 1]
SIG_LABELS = ["***",   "**",    "*",  "ns"]

def quadro_resumo_long(df_long: pd.DataFrame,
                       grupo_col: str,
                       variaveis: list[str],
                       col_estat: str) -> pd.DataFrame:
    """Gera quadro (estatísticas por cluster + ANOVA) para DF em formato longo."""
    agg_dict = {
        "n":      "count",
        "mean":   "mean",
        "std":    "std",
        "min":    "min",
        "25%":    lambda s: s.quantile(0.25),
        "median": "median",
        "75%":    lambda s: s.quantile(0.75),
        "max":    "max",
    }
    linhas   = []
    clusters = sorted(df_long[grupo_col].unique())

    for var in variaveis:
        sub = df_long[df_long["Variável"] == var]

        agg_named = {nome: (col_estat, func) for nome, func in agg_dict.items()}

        stats_df = (
            sub.groupby(grupo_col)       # agora é DataFrameGroupBy
               .agg(**agg_named)         # sintaxe moderna ✓
               .T                        # linhas = estat, colunas = clusters
        )
        row = {(estat, c): stats_df.loc[estat, c]
               for estat in stats_df.index
               for c in stats_df.columns}

        grupos = [sub.loc[sub[grupo_col] == c, col_estat].dropna() for c in clusters]
        p_val  = stats.f_oneway(*grupos)[1] if len(grupos) >= 2 and all(len(g) > 1 for g in grupos) else np.nan
        row["p_value"] = p_val
        row["signif"]  = (pd.cut([p_val], SIG_BINS, labels=SIG_LABELS).astype(str)[0]
                          if not np.isnan(p_val) else "na")
        linhas.append(pd.Series(row, name=var))

    return pd.DataFrame(linhas)

def pairwise_t_matrix(
        df: pd.DataFrame,
        grupo_col: str,
        estat_col: str,
        method: str = "bonferroni"
    ) -> pd.DataFrame:
    """
    p-values t-Student (Welch) entre todos os pares de clusters.
    • Descarta clusters com < 2 observações
    • Ajusta múltiplos testes: bonferroni | fdr_bh | None
    """
    # garante que só entram grupos “testáveis”
    counts = df.groupby(grupo_col)[estat_col].count()
    clusters = counts[counts >= 2].index.tolist()

    # matriz (pré-preenchida) ────
    mat = pd.DataFrame(np.nan, index=clusters, columns=clusters)

    # coleta p-values
    pvals, pairs = [], []
    for c1, c2 in combinations(clusters, 2):
        a = df.loc[df[grupo_col] == c1, estat_col].dropna()
        b = df.loc[df[grupo_col] == c2, estat_col].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pvals.append(p)
        pairs.append((c1, c2))

    # correção de múltiplos testes ────
    if method and pvals:
        pvals = smm.multipletests(pvals, method=method)[1]

    # preenche matriz simétrica
    for (c1, c2), p in zip(pairs, pvals):
        mat.loc[c1, c2] = mat.loc[c2, c1] = p

    return mat

def filtrar_outliers_iqr(df: pd.DataFrame,
                         valor_col: str,
                         agrupadores: list[str]) -> pd.DataFrame:
    """
    Remove outliers usando a regra IQR×1,5.
    É aplicada independentemente para cada combinação em agrupadores.
    """
    def _apply(gr):
        q1 = gr[valor_col].quantile(0.25)
        q3 = gr[valor_col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        return gr[(gr[valor_col] >= lo) & (gr[valor_col] <= hi)]

    return (df.groupby(agrupadores, group_keys=False)
              .apply(_apply)
              .reset_index(drop=True))
# ───────────────────────── Funções de gráfico ─────────────────────────
def plot_barras(df,est):
    fig=px.bar(df,x='Variável',y=est,color='Classe',color_discrete_map=CLASSE_CORES,
               barmode='group',facet_col='Método',facet_col_wrap=2,template=PLOTLY_TEMPLATE,
               height=480,title=f"{est.capitalize()} por Variável")
    fig.update_layout(uniformtext_minsize=8,uniformtext_mode='hide')
    return fig

def plot_radar(df,est,metodos,classes):
    fig=go.Figure()
    for c in classes:
        for m in metodos:
            tmp=df[(df['Classe']==c)&(df['Método']==m)]
            if not tmp.empty:
                fig.add_trace(go.Scatterpolar(r=tmp[est],theta=tmp['Variável'],fill='toself',
                                              name=f"{m} - Cluster {c}",line_color=CLASSE_CORES.get(c)))
    fig.update_layout(template=PLOTLY_TEMPLATE,showlegend=True,height=580,
                      polar=dict(radialaxis=dict(visible=True)),
                      title=f"Radar Chart - {est.capitalize()} por Variável")
    return fig

def plot_univariadas(df, est, grp_requested):
    """Exibe gráficos univariados + testes estatísticos.
    Se a coluna escolhida não existir em df, cai automaticamente para 'Classe'."""

    grp = grp_requested if grp_requested in df.columns else "Classe"
    if grp != grp_requested:
        st.info(f"Coluna '{grp_requested}' não existe neste arquivo – usando '{grp}'.")

    st.markdown("### Análises Univariadas")
    for var in sorted(df["Variável"].unique()):
        st.subheader(f"Variável: {var}")
        dv = df[df["Variável"] == var]
        grupos = [g[est].dropna() for _, g in dv.groupby(grp)]
        nmin = min(len(g) for g in grupos)

        # Gráficos ------------------------------------------------------
        c1, c2 = st.columns(2)
        with c1:
            if nmin <= 15:
                fg = px.strip(dv, x=grp, y=est, color=grp, stripmode="overlay",
                               color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE,
                               title="Valores Individuais")
                fg.update_traces(jitter=0.35, marker_size=8)
            else:
                fg = px.histogram(dv, x=est, color=grp, nbins=min(20, max(5, dv.shape[0] // 3)),
                                  marginal="rug", color_discrete_map=CLASSE_CORES,
                                  template=PLOTLY_TEMPLATE, title="Distribuição")
            st.plotly_chart(fg, use_container_width=True)
        with c2:
            vg = px.violin(dv, x=grp, y=est, color=grp, box=True, points="all",
                           color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE,
                           title="Violin + Box")
            st.plotly_chart(vg, use_container_width=True)

        # Testes Estatísticos ------------------------------------------
        st.markdown("#### Testes estatísticos")
        if len(grupos) == 2:
            t_stat, p_val = stats.ttest_ind(*grupos, equal_var=False)
            st.write(f"**t‑Student (duas amostras, variâncias não iguais)** → *t* = {t_stat:.4f}, *p* = {p_val:.4f}")
            # AIC simples a partir do RSS do modelo reduzido vs completo
            n1, n2 = len(grupos[0]), len(grupos[1])
            rss = sum((grupos[0] - grupos[0].mean())**2) + sum((grupos[1] - grupos[1].mean())**2)
            n = n1 + n2
            k = 2  # média1, média2
            aic = n * np.log(rss / n) + 2 * k
            st.write(f"AIC aproximado do modelo de 2 médias: {aic:.2f}")
        else:
            f_stat, p_anova = stats.f_oneway(*grupos)
            st.write(f"**ANOVA** → *F* = {f_stat:.4f}, *p* = {p_anova:.4f}")
        # teste não paramétrico
        if len(grupos) >= 2:
            h_stat, p_kw = stats.kruskal(*grupos)
            st.write(f"**Kruskal‑Wallis** (não paramétrico) → *H* = {h_stat:.4f}, *p* = {p_kw:.4f}")

        # Resumo --------------------------------------------------------
        resumo = dv.groupby(grp)[est].agg(n="count", média="mean", mediana="median",
                                          mín="min", máx="max", desvio="std").round(2).reset_index()
        st.dataframe(resumo, use_container_width=True)


                           
# ───────────────────────── Função ANOVA ─────────────────────────
def analise_estatistica_variavel(grp):
    """Estatísticas multivariadas; se grp não existir, usa 'Classe'."""
    st.markdown("## 📐 Análise Estatística por Variável")
    arqs = carregar_todos_arquivos(PASTA_ANALISES)
    if not arqs:
        st.warning("Nenhum arquivo em data/merged.")
        return
    arq = st.selectbox("Selecione o arquivo:", list(arqs.keys()))
    dfv = arqs[arq]

    if grp not in dfv.columns:
        st.info(f"Coluna '{grp}' não existe em {arq} – usando 'Classe'.")
        grp = "Classe"

    num_cols = dfv.select_dtypes(include=["float64", "int64"]).columns.tolist()
    col = st.selectbox("Variável numérica:", num_cols)
    grupos = [g[col].dropna() for _, g in dfv.groupby(grp)]

    st.markdown("### ANOVA")
    if len(grupos) > 1:
        f, p = stats.f_oneway(*grupos)
        st.write(f"F = {f:.4f}, p = {p:.4f}")
        st.success("Diferença significativa." if p < 0.05 else "Sem diferença significativa.")
    else:
        st.warning("Não há grupos suficientes para ANOVA.")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(px.histogram(dfv, x=col, color=grp, nbins=20, marginal='box',
                                     color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE,
                                     title="Histograma"), use_container_width=True)
    with ch2:
        st.plotly_chart(px.box(dfv, x=grp, y=col, color=grp,
                               color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE,
                               title="Boxplot"), use_container_width=True)

# ───────────────────────── Carregamento Inicial ─────────────────────────
metric_files = carregar_todos_arquivos(PASTA_DADOS)
if not metric_files:
    st.error("Nenhum CSV encontrado em data/metricas.")
    st.stop()
sel_metric = st.selectbox("Selecione o arquivo de métricas:", list(metric_files.keys()), key="sel_metric")
df = metric_files[sel_metric]

# Listas derivadas uma única vez ----------------------------------------
metodos   = sorted(df['Método'].unique())
classes   = sorted(df['Classe'].unique())
variaveis = sorted(df['Variável'].unique())
estat_cols = [c for c in df.columns if c not in ['Método','Classe','Variável']]

with st.sidebar:
    st.subheader("🔧 Configurações gerais")
    grp_sel = st.selectbox("Agrupamento (coluna de cluster):", GROUP_COLS, key="grp_sel")

    st.markdown("---")
    met_sel = st.multiselect("Métodos:",    metodos,   default=metodos,   key="met_sel")
    cls_sel = st.multiselect("Classes:",    classes,   default=classes,   key="cls_sel")
    var_sel = st.multiselect("Variáveis:",  variaveis, default=variaveis, key="var_sel")
    est_sel = st.multiselect("Estatísticas:", estat_cols,
                              default=[estat_cols[0]], key="est_sel")
    view_mode = st.radio("Visualização:",
                         ["Escala Real", "Normalizado", "Ambos"],
                         index=0, key="view_mode")

# aplica filtro ----------------------------------------------------------

df_filt = df[(df["Método"].isin(met_sel)) &
             (df["Classe"].isin(cls_sel)) &
             (df["Variável"].isin(var_sel))]

if df_filt.empty:
    st.warning("Filtros retornaram zero linhas.")
    st.stop()

# Normalização -----------------------------------------------------------
if view_mode in ["Normalizado", "Ambos"]:
    df_norm = normalizar_df(df_filt, estat_cols)
else:
    df_norm = pd.DataFrame()

# ───────────────────────── Layout em Abas ─────────────────────────
aba_metricas, aba_univ, aba_stats = st.tabs(
    ["📊 Métricas", "🏷️ Univariadas", "📐 Estatísticas"]
)

# Aba Métricas -----------------------------------------------------------
with aba_metricas:
    metodo_radio = st.radio(
        "Filtrar método:", ["Todos"] + met_sel, horizontal=True
    )

    # -------- Gráficos --------
    for est in est_sel:
        st.header(f"Estatística: {est}")
        for mode, data in [("Escala Real", df_filt), ("Normalizado", df_norm)]:
            if view_mode in [mode, "Ambos"] and not data.empty:
                st.subheader(mode)
                col1, col2 = st.columns(2)
                with col1:
                    d = (
                        data
                        if metodo_radio == "Todos"
                        else data[data["Método"] == metodo_radio]
                    )
                    st.plotly_chart(plot_barras(d, est), use_container_width=True)
                with col2:
                    d = (
                        data
                        if metodo_radio == "Todos"
                        else data[data["Método"] == metodo_radio]
                    )
                    st.plotly_chart(
                        plot_radar(d, est, d["Método"].unique(), cls_sel),
                        use_container_width=True,
                    )

    # ───────── Pivot-table por estatística ─────────
    st.markdown("---")
    st.subheader("Tabelas resumidas (clusters × variáveis)")

    for est in est_sel:
        st.markdown(f"### {est.capitalize()}")
        pivot = df_filt.pivot_table(
            index="Classe", columns="Variável", values=est
        )
        st.dataframe(pivot, use_container_width=True)

    # Botão de download — key exclusiva desta aba
    csv_bytes = df_filt.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Baixar CSV filtrado",
        csv_bytes,
        file_name="metricas_filtradas.csv",
        mime="text/csv",
        key="download_filtrado_metricas",
    )

# Aba Univariadas --------------------------------------------------------
with aba_univ:
    # Seleciona variável
    var_univ = st.selectbox("Variável:", variaveis, key="var_univ")

    # Determina coluna de agrupamento realmente presente
    grp_active = grp_sel if grp_sel in df_filt.columns else "Classe"
    if grp_active != grp_sel:
        st.info(f"Coluna '{grp_sel}' não existe neste arquivo – usando '{grp_active}'.")

    # Opções de clusters disponíveis para a variável escolhida
    cls_options = sorted(df_filt[df_filt["Variável"] == var_univ][grp_active].unique())
    cls_univ = st.multiselect("Clusters a incluir:", cls_options, default=cls_options)

    estat_univ = st.selectbox("Estatística:", estat_cols, key="estat_univ")

    # Filtra dataframe
    df_uni = df_filt[(df_filt["Variável"] == var_univ) & (df_filt[grp_active].isin(cls_univ))]

    if df_uni.empty:
        st.warning("Nada para mostrar – verifique filtros de classe.")
    else:
        plot_univariadas(df_uni, estat_univ, grp_active)

        with st.expander("Legenda de significância (p-value)"):
                st.markdown("""
#### Como interpretar
* ***p ≤ 0.05*** – rejeitamos H₀: pelo menos dois clusters diferem na estatística analisada.
* ***p > 0.05*** – não há evidência suficiente para afirmar diferença entre clusters.
""")
    
# Aba Estatísticas -------------------------------------------------------
PCA_VARS = [
    "comp_res", "outros_usos", "fator_com",
    "densidade_hec_norm", "mobilidade_norm", "equipamentos_norm",
    "a_vl_m2_construcao_norm", "comp_res_norm",
    "outros_usos_norm", "fator_com_norm"
]


# =============================================================================
#                             ABA  ESTATÍSTICAS
# =============================================================================
with aba_stats:
    tab_global, tab_t = st.tabs(["Testes globais", "t-Student pairwise"])

    # ---------- 1) Testes globais -----------------------------------
    with tab_global:
        estat_ref = est_sel[0]
        st.subheader(f"Resumo por cluster – {estat_ref}")

        grp_ok = grp_sel if grp_sel in df_filt.columns else "Classe"
        df_clean = filtrar_outliers_iqr(df_filt, estat_ref, ["Variável", grp_ok])
        tab = quadro_resumo_long(df_clean, grp_ok, var_sel, estat_ref)

        st.caption(f"Outliers removidos: {len(df_filt) - len(df_clean)} linhas")
        st.dataframe(tab.style.format({"p_value": "{:.3e}"}), use_container_width=True)

    # ---------- 2) t‑Student pairwise + PCA -------------------------
    with tab_t:
        # =================== SELEÇÃO DO ALGORITMO =====================
        metodo_pca = st.selectbox("Algoritmo de cluster:", GROUP_COLS, key="metodo_pca")
        if metodo_pca not in df_ana.columns:
            st.error(f"Coluna '{metodo_pca}' não existe no arquivo selecionado.")
            st.stop()

        # cria coluna "Classe" a partir da coluna do algoritmo escolhido
        df_met = df_ana.copy()
        df_met["Classe"] = df_met[metodo_pca]

        # ---------------- CLUSTERS A INCLUIR --------------------------
        cls_opts = sorted(df_met["Classe"].dropna().unique())
        cls_sel = st.multiselect("Clusters (PCA):", cls_opts, default=cls_opts, key="cls_pca")
        df_met = df_met[df_met["Classe"].isin(cls_sel)].reset_index(drop=True)

        # =========================== PCA ==============================
        estat_cols_ana = [c for c in df_ana.columns if c not in ["Variável"] + GROUP_COLS]
        if not estat_cols_ana:
            st.error("Nenhuma coluna numérica disponível para PCA no arquivo escolhido.")
            st.stop()

        stat_col = st.selectbox("Estatística PCA:", estat_cols_ana, key="stat_pca")

        # Formato wide: linha = observação, colunas = variáveis em PCA_VARS
        wide = (
            df_met.reset_index()
                  .pivot_table(index="index", columns="Variável", values=stat_col)
        )

        vars_disp = [v for v in PCA_VARS if v in wide.columns]
        wide = wide[vars_disp].dropna(how="any")
        if len(vars_disp) < 2 or wide.shape[0] < 2:
            st.warning("Dados insuficientes para PCA depois dos filtros.")
            st.stop()

        X_std = StandardScaler().fit_transform(wide)
        pca   = PCA(n_components=3).fit(X_std)
        pcs   = pca.transform(X_std)

        pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"], index=wide.index)
        pc_df["Classe"] = df_met.loc[wide.index, "Classe"].values

        st.write(f"PC1 explica {pca.explained_variance_ratio_[0]:.1%}; "
                 f"PC2 {pca.explained_variance_ratio_[1]:.1%}")
        st.plotly_chart(
            px.scatter(pc_df, x="PC1", y="PC2", color="Classe",
                       template=PLOTLY_TEMPLATE, color_discrete_map=CLASSE_CORES,
                       title="PCA – PC1 × PC2"),
            use_container_width=True
        )

        # ---------------- pairwise em PC1 -----------------------------
        corr_pca = st.radio(
            "Correção múltiplos testes (PC1):", ["bonferroni", "fdr_bh", "nenhuma"],
            key="corr_pca")
        meth_corr = None if corr_pca == "nenhuma" else corr_pca
        mat_pc1 = pairwise_t_matrix(pc_df, "Classe", "PC1", meth_corr)

        if st.radio("Visualização PC1:", ["Tabela", "Heatmap"], key="view_pc1") == "Tabela":
            st.dataframe(mat_pc1.style.format("{:.3e}"), use_container_width=True)
        else:
            st.plotly_chart(px.imshow(mat_pc1, text_auto=".2e", zmin=0, zmax=0.05,
                                      color_continuous_scale="RdBu_r", title="Pairwise – PC1"),
                            use_container_width=True)

        st.markdown("---")

        # ================= pairwise por variável ======================
        var_ana = sorted(df_ana["Variável"].unique())
        var_pair = st.selectbox("Variável:", var_ana, key="var_pair")
        estat_pair = st.selectbox("Estatística:", estat_cols_ana, key="estat_pair")
        corr_var = st.radio("Correção múltiplos testes:", ["bonferroni", "fdr_bh", "nenhuma"], key="corr_var")
        meth_var = None if corr_var == "nenhuma" else corr_var

        df_pair = df_ana[df_ana["Variável"] == var_pair]
        grp_ok  = grp_sel if grp_sel in df_pair.columns else "Classe"
        mat_var = pairwise_t_matrix(df_pair, grp_ok, estat_pair, meth_var)

        if st.radio("Visualização matriz:", ["Tabela", "Heatmap"], key="view_var") == "Tabela":
            st.dataframe(mat_var.style.format("{:.3e}"), use_container_width=True)
        else:
            st.plotly_chart(px.imshow(mat_var, text_auto=".2e", zmin=0, zmax=0.05,
                                      color_continuous_scale="RdBu_r",
                                      title=f"Pairwise – {var_pair}"),
                            use_container_width=True)
