# ───────────────────────── Configuração global ─────────────────────────
st.set_page_config(
    page_title="Dashboard de Análise de Clusters para o Município de São Paulo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS rápido ------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] {font-family:'Roboto',sans-serif;}
        #MainMenu, footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Constantes ─────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
CLASSE_CORES = {0: "#F4DD63", 1: "#B1BF7C", 2: "#D58243", 3: "#C65534", 4: "#6FA097", 5: "#14407D"}
GROUP_COLS = ["KMeans_k5", "Spectral_k5", "KMedoids_k5"]
BASE_DIR = Path(__file__).parent
PASTA_DADOS = BASE_DIR / "data" / "metricas"
PASTA_ANALISES = BASE_DIR / "data" / "merged"

# Variáveis usadas no PCA ------------------------------------------------
PCA_VARS = [
    "comp_res", "outros_usos", "fator_com",
    "densidade_hec_norm", "mobilidade_norm", "equipamentos_norm",
    "a_vl_m2_construcao_norm", "comp_res_norm",
    "outros_usos_norm", "fator_com_norm",
]

# ───────────────────────── Utilidades --------- ------------------------
@st.cache_data(show_spinner=False)
def carregar_todos_arquivos(pasta: Path) -> dict[str, pd.DataFrame]:
    """Lê todos os CSV de uma pasta recursivamente e devolve dict nome→DataFrame"""
    d = {}
    for csv in pasta.rglob("*.csv"):
        try:
            df = pd.read_csv(csv).loc[:, lambda x: ~x.columns.str.contains("^Unnamed")]
            d[csv.name] = df
        except Exception as e:
            st.warning(f"Erro ao carregar {csv.name}: {e}")
    return d


def normalizar_df(df: pd.DataFrame, est_cols: list[str]) -> pd.DataFrame:
    df_n = df.copy()
    for est in est_cols:
        if est in df_n.columns:
            for var in df_n["Variável"].unique():
                mask = df_n["Variável"] == var
                mn, mx = df_n.loc[mask, est].min(), df_n.loc[mask, est].max()
                if mn != mx:
                    df_n.loc[mask, est] = (df_n.loc[mask, est] - mn) / (mx - mn)
    return df_n

# ───────── Significância ------------------------------------------------
SIG_BINS = [-np.inf, 0.001, 0.01, 0.05, 1]
SIG_LABELS = ["***", "**", "*", "ns"]

# ───────── Funções estatísticas ----------------------------------------

def quadro_resumo_long(df_long: pd.DataFrame, grupo: str, variaveis: list[str], col_estat: str) -> pd.DataFrame:
    agg_dict = {
        "n": "count", "mean": "mean", "std": "std", "min": "min",
        "25%": lambda s: s.quantile(0.25), "median": "median",
        "75%": lambda s: s.quantile(0.75), "max": "max",
    }
    linhas = []
    clusters = sorted(df_long[grupo].unique())
    for var in variaveis:
        sub = df_long[df_long["Variável"] == var]
        stats_df = (
            sub.groupby(grupo)[col_estat].agg(**{k: v for k, v in agg_dict.items()}).T
        )
        row = {(estat, c): stats_df.loc[estat, c] for estat in stats_df.index for c in stats_df.columns}
        grupos = [sub.loc[sub[grupo] == c, col_estat].dropna() for c in clusters]
        p_val = stats.f_oneway(*grupos)[1] if all(len(g) > 1 for g in grupos) else np.nan
        row["p_value"] = p_val
        row["signif"] = pd.cut([p_val], SIG_BINS, labels=SIG_LABELS).astype(str)[0] if not np.isnan(p_val) else "na"
        linhas.append(pd.Series(row, name=var))
    return pd.DataFrame(linhas)


def pairwise_t_matrix(df: pd.DataFrame, grupo: str, valor: str, method: str | None = "bonferroni") -> pd.DataFrame:
    counts = df.groupby(grupo)[valor].count()
    clusters = counts[counts >= 2].index.tolist()
    mat = pd.DataFrame(np.nan, index=clusters, columns=clusters)
    pvals, pairs = [], []
    for c1, c2 in combinations(clusters, 2):
        a = df.loc[df[grupo] == c1, valor].dropna()
        b = df.loc[df[grupo] == c2, valor].dropna()
        _, p = stats.ttest_ind(a, b, equal_var=False)
        pvals.append(p)
        pairs.append((c1, c2))
    if method and pvals:
        pvals = smm.multipletests(pvals, method=method)[1]
    for (c1, c2), p in zip(pairs, pvals):
        mat.loc[c1, c2] = mat.loc[c2, c1] = p
    return mat


def filtrar_outliers_iqr(df: pd.DataFrame, col: str, grupos: list[str]) -> pd.DataFrame:
    def _cut(g):
        q1, q3 = g[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return g[(g[col] >= lo) & (g[col] <= hi)]
    return df.groupby(grupos, group_keys=False).apply(_cut).reset_index(drop=True)

# ───────── Carregamento de dados ---------------------------------------
metric_files = carregar_todos_arquivos(PASTA_DADOS)
merged_files = carregar_todos_arquivos(PASTA_ANALISES)

if not metric_files:
    st.error("Nenhum CSV em data/metricas"); st.stop()
if not merged_files:
    st.error("Nenhum CSV em data/merged"); st.stop()

sel_metric = st.selectbox("Arquivo de métricas:", list(metric_files.keys()), key="sel_metric")
df = metric_files[sel_metric]

sel_merged = st.sidebar.selectbox("Arquivo para PCA / pairwise:", list(merged_files.keys()), key="sel_merged")
df_ana = merged_files[sel_merged]

# Listas derivadas ------------------------------------------------------
metodos = sorted(df["Método"].unique())
classes = sorted(df["Classe"].unique())
variaveis = sorted(df["Variável"].unique())
estat_cols = [c for c in df.columns if c not in ["Método", "Classe", "Variável"]]

with st.sidebar:
    st.subheader("🔧 Configurações gerais")
    grp_sel = st.selectbox("Agrupamento (cluster):", GROUP_COLS, key="grp_sel")
    st.markdown("---")
    met_sel = st.multiselect("Métodos:", metodos, default=metodos, key="met_sel")
    cls_sel = st.multiselect("Classes:", classes, default=classes, key="cls_sel")
    var_sel = st.multiselect("Variáveis:", variaveis, default=variaveis, key="var_sel")
    est_sel = st.multiselect("Estatísticas:", estat_cols, default=[estat_cols[0]], key="est_sel")
    view_mode = st.radio("Visualização:", ["Escala Real", "Normalizado", "Ambos"], index=0, key="view_mode")

# Filtros em df ---------------------------------------------------------
df_filt = df.query("Método in @met_sel and Classe in @cls_sel and Variável in @var_sel")
if df_filt.empty:
    st.warning("Filtros retornaram zero linhas."); st.stop()

df_norm = normalizar_df(df_filt, estat_cols) if view_mode in ["Normalizado", "Ambos"] else pd.DataFrame()

# ───────── Abas de layout ----------------------------------------------
aba_metricas, aba_univ, aba_stats = st.tabs(["📊 Métricas", "🏷️ Univariadas", "📐 Estatísticas"])

# ----------------------------------------------------------------------
# Aba Estatísticas > tab_t (PCA + pairwise)
# ----------------------------------------------------------------------
with aba_stats:
    tab_global, tab_t = st.tabs(["Testes globais", "t-Student pairwise"])

    # ---------- Pairwise ----------
    with tab_t:
        # 1) PCA em PCA_VARS -----------------------------------------
        metodo_pca = st.selectbox("Método de cluster:", GROUP_COLS, key="metodo_pca")
        df_met = df_ana[df_ana["Método"] == metodo_pca].copy()
        cls_opts = sorted(df_met["Classe"].unique())
        cls_sel_pca = st.multiselect("Clusters a incluir (PCA):", cls_opts, default=cls_opts, key="cls_sel_pca")
        df_met = df_met[df_met["Classe"].isin(cls_sel_pca)]

        estat_cols_ana = [c for c in df_ana.columns if c not in ["Método", "Classe", "Variável"]]
        stat_col = st.selectbox("Estatística usada no PCA:", estat_cols_ana, key="pca_stat")

        df_met = df_met.reset_index(drop=True)  # índice único
        wide = df_met.pivot_table(index="index", columns="Variável", values=stat_col)
        avail_vars = [v for v in PCA_VARS if v in wide.columns]
        wide = wide[avail_vars].dropna()
        if len(avail_vars) < 2 or wide.shape[0] < 2:
            st.warning("Dados insuficientes para PCA."); st.stop()

        X_std = StandardScaler().fit_transform(wide)
        pca = PCA(n_components=3).fit(X_std)
        scores = pca.transform(X_std)
        pc_df = pd.DataFrame(scores, index=wide.index, columns=["PC1", "PC2", "PC3"])
        pc_df["Classe"] = df_met.loc[wide.index, "Classe"].values

        st.write(f"Variância explicada – PC1 {pca.explained_variance_ratio_[0]:.1%} · PC2 {pca.explained_variance_ratio_[1]:.1%}")
        st.plotly_chart(px.scatter(pc_df, x="PC1", y="PC2", color="Classe", color_discrete_map=CLASSE_CORES, template=PLOTLY_TEMPLATE, title="PCA – PC1 × PC2"), use_container_width=True)

        # ANOVA em PC1
        grupos_pc1 = [g["PC1"] for _, g in pc_df.groupby("Classe")]
        if len(grupos_pc1) >= 2:
            f, p = stats.f_oneway(*grupos_pc1); st.write(f"ANOVA PC1 → F={f:.2f}, p={p:.3e}")

        corr_pca = st.radio("Correção múltiplos testes (PCA):", ["bonferroni", "fdr_bh", "nenhuma"], key="corr_pca")
        method_corr = None if corr_pca == "nenhuma" else corr_pca
        mat_pc1 = pairwise_t_matrix(pc_df, "Classe", "PC1", method_corr)
        view_pca = st.radio("Visualização matriz (PC1):", ["Tabela", "Heatmap"], key="view_pca")
        if view_pca == "Tabela":
            st.dataframe(mat_pc1.style.format("{:.3e}"), use_container_width=True)
        else:
            st.plotly_chart(px.imshow(mat_pc1, text_auto=".2e", zmin=0, zmax=0.05, color_continuous_scale="RdBu_r", title="p-values t-Student – PC1"), use_container_width=True)

        st.markdown("---")

        # 2) Pairwise por variável -----------------------------------
        var_sel_ana = sorted(df_ana["Variável"].unique())
        var_pair = st.selectbox("Variável:", var_sel_ana, key="pair_var")
        estat_pair = st.selectbox("Estatística:", estat_cols_ana, key="pair_est")
        corr_label = st.radio("Correção múltiplos testes:", ["bonferroni", "fdr_bh", "nenhuma"], key="corr_var")
        method = None if corr_label == "nenhuma" else corr_label

        df_pair = df_ana[df_ana["Variável"] == var_pair]
        grp_ok = grp_sel if grp_sel in df_pair.columns else "Classe"
        mat = pairwise_t_matrix(df_pair, grp_ok, estat_pair, method)

        view_var = st.radio("Visualização:", ["Tabela", "Heatmap"], key="view_var")
        if view_var == "Tabela":
            st.dataframe(mat.style.format("{:.3e}"), use_container_width=True)
        else:
            st.plotly_chart(px.imshow(mat, text_auto=".2e", zmin=0, zmax=0.05, color_continuous_scale="RdBu_r", title=f"p-values – {var_pair}"), use_container_width=True)
