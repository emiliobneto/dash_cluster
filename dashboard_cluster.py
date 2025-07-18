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

        stats_df = (
            sub.groupby(grupo_col)[col_estat]
               .agg(agg_dict).T                    # linhas = estat, colunas = clusters
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

def pairwise_t_matrix(df: pd.DataFrame,
                  grupo_col: str,
                  var: str,
                  estat_col: str,
                  method: str = "bonferroni") -> pd.DataFrame:
    """
    Devolve DataFrame (clusters × clusters) com p-values t-Student.
    `method`: bonferroni | fdr_bh | None
    """
    clusters = sorted(df[grupo_col].unique())
    pvals, idx = [], []

    for c1, c2 in combinations(clusters, 2):
        a = df.loc[df[grupo_col] == c1, estat_col].dropna()
        b = df.loc[df[grupo_col] == c2, estat_col].dropna()
        if len(a) > 1 and len(b) > 1:
            _, p = stats.ttest_ind(a, b, equal_var=False)
            pvals.append(p)
            idx.append((c1, c2))

    # correção múltiplos testes
    if method is not None and pvals:
        pvals = smm.multipletests(pvals, method=method)[1]

    # monta matriz simétrica
    mat = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    for (c1, c2), p in zip(idx, pvals):
        mat.loc[c1, c2] = mat.loc[c2, c1] = p
    np.fill_diagonal(mat.values, np.nan)
    return mat

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

 # ------------------------------------------------------------------
    # Quadro resumido por cluster (todas as variáveis) + ANOVA
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"Resumo por cluster – {estat_univ}")

    tabela_resumo = quadro_resumo_long(
        df_filt,           # usa o dataframe já filtrado
        grp_active,        # coluna de agrupamento (Classe ou outra)
        var_sel,           # lista de variáveis selecionadas no sidebar
        estat_univ         # estatística que o usuário escolheu p/ análise
    )

    st.dataframe(
        tabela_resumo.style.format({"p_value": "{:.3e}"}),
        use_container_width=True
    )

    with st.expander("Legenda de significância (p-value)"):
        st.markdown(
            """
| Estrelas | p ≤ | Interpretação |
|:---:|:---:|:---|
| *** | 0.001 | diferença **muito** significativa |
| **  | 0.01  | diferença **significativa** |
| *   | 0.05  | diferença moderada |
| ns  | > 0.05 | sem diferença significativa |
            """
        )

# Aba Estatísticas -------------------------------------------------------
with aba_stats:
    # ---------------- tabs internas ----------------
    tab_global, tab_t = st.tabs(["Testes globais", "t-Student pairwise"])

    # ---------- 1) Quadro global (ANOVA/Kruskal) ----------
    with tab_global:
        estat_ref = est_sel[0]          # primeira estatística escolhida
        st.subheader(f"Resumo por cluster – {estat_ref}")

        tabela_resumo = quadro_resumo_long(
            df_filt, grp_sel if grp_sel in df_filt.columns else "Classe",
            var_sel, estat_ref
        )
        st.dataframe(
            tabela_resumo.style.format({"p_value": "{:.3e}"}),
            use_container_width=True
        )
        with st.expander("Legenda de significância (p-value)"):
            st.markdown(
                """
| Estrelas | p ≤ | Interpretação |
|:---:|:---:|:---|
| *** | 0.001 | diferença **muito** significativa |
| **  | 0.01  | diferença **significativa** |
| *   | 0.05  | diferença moderada |
| ns  | > 0.05 | sem diferença significativa |
                """
            )

    # ---------- 2) Matriz pairwise t-Student ----------
    with tab_t:
        st.subheader("Matriz de p-values t-Student")
        var_pair = st.selectbox("Variável:", var_sel, key="pair_var")
        estat_pair = st.selectbox("Estatística:", estat_cols, key="pair_est")
        corr_method = st.radio(
            "Correção múltiplos testes:",
            ["bonferroni", "fdr_bh", "nenhuma"],
            index=0
        )
        method = None if corr_method == "nenhuma" else corr_method

        mat = pairwise_t_matrix(
            df_filt[df_filt["Variável"] == var_pair],
            grp_sel if grp_sel in df_filt.columns else "Classe",
            var_pair,
            estat_pair,
            method
        )

        # escolha: mostrar como tabela ou heatmap
        view = st.radio("Visualização:", ["Tabela", "Heatmap"], horizontal=True)
        if view == "Tabela":
            st.dataframe(mat.style.format("{:.3e}"), use_container_width=True)
        else:
            fig = px.imshow(
                mat,
                text_auto=".2e",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title=f"p-values t-Student – {var_pair} ({estat_pair})"
            )
            st.plotly_chart(fig, use_container_width=True)
