from typing import Any
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import html as pyhtml
import numpy as np
import textwrap
import torch
from scipy import sparse
import scipy

PALETTE = [
    "#E6194B",  # vivid red
    "#3CB44B",  # bright green
    "#4363D8",  # bold blue
    "#F58231",  # strong orange
    "#911EB4",  # deep purple
    "#46F0F0",  # cyan
    "#F032E6",  # magenta
    "#BCF60C",  # lime
    "#FABEBE",  # soft pink
    "#008080"   # teal
]

@st.cache_data
def get_data(dataset: str):
    return pd.read_feather(f"data/{dataset}_df.feather")

@st.cache_data
def get_items(dataset: str):
    return pd.read_feather(f"data/{dataset}_items.feather")

@st.cache_data
def load_gb_segments():
    return np.load("data/gb10_final_segments_20_2048_16nnz.npy", allow_pickle=True).item()

@st.cache_data
def load_idx():
    return np.load("data/gb10_book_id_index.npy", allow_pickle=True)

@st.cache_data
def get_test_users():
    X = sparse.load_npz("data/gb10_test_users.npz")
    return X[::-1]

@st.cache_data
def get_A():
    A = sparse.load_npz("data/gb10_weights_2048_16nnz_sparse.npz")
    return A

def get_B(segments):
    arrays=[v["similar"]for v in segments.values()]
    n_rows = len(arrays)
    n_cols = 2048 
    
    data = []
    rows = []
    cols = []
    
    for i, arr in enumerate(arrays):
        data.extend(np.sign(arr))           # +1 or -1 depending on sign
        rows.extend([i] * len(arr))         # row index
        cols.extend(np.abs(arr))            # column index (absolute value)
    
    # Build sparse matrix
    B = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
    
    return B

def topk(arr, k): # todo rewrite without torch
    ten=torch.from_numpy(arr)
    k,v = torch.topk(ten, k)
    return k.numpy(), v.numpy()

def norm_csr(mat):
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1  # row-wise L2 norms
    norms[norms == 0] = 1  # avoid division by zero
    inv_norms = 1.0 / norms
    return mat.multiply(inv_norms[:, np.newaxis])

# table style for results
def highlight_cells(x):
    df_styled = x.copy()
    # Default: no style
    df_styled = pd.DataFrame('', index=x.index, columns=x.columns, dtype=object)
    
    # Highlight specific cells (by condition or coordinates)
    for i in range(len(df_styled)):
        j = (i-1)//3
        if j%2 == 0:
            df_styled.iloc[i, :] += 'background-color: #f0f8ff;'
        else:
            df_styled.iloc[i, :] += 'background-color: white;'

        if (i)%3 == 1:
            for c in x.columns:
                if x[c].dtype=="float":
                    sub_series = df_styled[c].iloc[i:i+3]
                    sub_series.iloc[sub_series.argmax()]+= 'font-weight: bold;'
    return df_styled

# plot three graphs
def plot_three_side_by_side(df1, df2, df3, titles=("Recall@20","Recall@50","NDCG@100"), baselines=[None, None, None], legend_title_text="Method", xcol="nnz", baselines_2=None):
    # Collect method names (all columns except 'nnz')
    # normalize once
    (df1, df2, df3), cat_str = normalize_x_as_reversed_categories(
        df1, df2, df3, xcol=xcol
    )
    def methods(df): return [c for c in df.columns if c not in [xcol, "nnz_cat"]]
    all_methods = list(dict.fromkeys(methods(df1) + methods(df2) + methods(df3)))  # stable order

    # Consistent colors across all panels
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.D3
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(all_methods)}

    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=titles)

    for i, df in enumerate([df1, df2, df3], start=1):
        for m in all_methods:
            if m not in df.columns:
                continue  # method not present in this DF
            fig.add_trace(
                go.Scatter(
                    x=df["nnz_cat"], y=df[m],
                    mode="lines+markers",
                    name=m if i == 1 else m,     # name is required for legendgroup
                    showlegend=(i == 1),         # ONLY show legend in the first subplot
                    legendgroup=m,               # link traces across subplots
                    line=dict(color=color_map[m]),
                    marker=dict(color=color_map[m]),
                    hovertemplate = f"{xcol}=%{{x}}<br>%{{y:.4f}}<extra>{m}</extra>"
                ),
                row=1, col=i
            )
            
        if baselines[i-1] is not None:
            fig.add_hline(y=baselines[i-1], annotation_text=f"dense ELSA ({baselines[i-1]:.3f})", line_color="black", line_dash="dash",row=1, col=i),
        
        fig.update_yaxes(title_text="", row=1, col=i)
        fig.update_xaxes(type="category", categoryorder="array",
                     categoryarray=cat_str, tickangle=0, title_text=xcol,
                     row=1, col=i)

    fig.update_layout(
        height=420, width=1200,
        legend_title_text=legend_title_text,
        title_text="",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

# to maintain stable order in the table
def normalize_x_as_reversed_categories(*dfs, xcol="nnz"):
    # collect all x values across dfs, make one global reversed order
    cats = sorted(
        {int(v) for df in dfs for v in [int(float(x)) for x in dfs[0][xcol].unique()]},
        reverse=True #if xcol=="nnz" else False
    )
    cat_str = [str(v) for v in cats]

    out = []
    for df in dfs:
        df = df.copy()
        df["nnz_cat"] = pd.Categorical(
            df[xcol].astype(float).astype(int).astype(str),
            categories=cat_str,
            ordered=True
        )
        out.append(df)
    return out, cat_str

# ---------- Global CSS + optional scroll restore ----------
def init_rows_rendering():
    st.markdown("""
    <style>
    /* Row */
    .rec-row { margin: 0 0 0 0; }
    .rec-title { font-weight: 700; font-size: 1.1rem; margin: 4px 0 8px 0; }
    .rec-rail { display: flex; gap: 14px; overflow-x: auto; overflow-y: hidden;
                scroll-snap-type: x mandatory; padding: 4px 2px; scroll-behavior: smooth; }

    /* Card */
    .card { flex: 0 0 auto; width: 220px; scroll-snap-align: start;
            border-radius: 12px; background: #fff; color: #222; text-decoration: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
    .card img { width: 100%; height: 124px; object-fit: cover;
                border-top-left-radius: 12px; border-top-right-radius: 12px; }
    .card-body { padding: 8px 10px 10px 10px; }
    .card-title { font-weight: 600; font-size: .95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .card-meta { font-size: 0.6rem; color: #888; margin-top: 4px; }
    .card-sub { font-size: .82rem; color: #666; }

    /* Little arrow buttons (optional) */
    .rail-wrap { position: relative; }
    .rail-arrow { position: absolute; top: 50%; transform: translateY(-50%); z-index: 2;
                width: 28px; height: 28px; border-radius: 50%; border: 0;
                background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,.2);
                display:flex; align-items:center; justify-content:center; cursor:pointer; }
    .rail-left { left: 0; } .rail-right { right: 0; }
    </style>

    <!-- restore last page scroll after rerun -->
    <script>
    (function() {
    try {
        const y = sessionStorage.getItem('page_scroll_y');
        if (y !== null) window.scrollTo(0, parseInt(y, 10));
    } catch (_) {}
    })();
    </script>
    """, unsafe_allow_html=True)

# ---------- Renderer (pure Streamlit DOM) ----------
def render_row(title: str, items: list, row_id: str):
    # JS helpers once (fine)
    st.markdown("""
<script>
function savePageScroll(){try{sessionStorage.setItem('page_scroll_y', String(window.scrollY));}catch(_){}} 
function scrollRail(id,dx){const el=document.getElementById(id); if(el) el.scrollBy({left:dx,behavior:'smooth'});}
</script>
""", unsafe_allow_html=True)

    cards = []
    for it in items:
        t = pyhtml.escape("" if it["title"] is None else it["title"])
        a = pyhtml.escape(it.get("authors",""))
        cards.append(
            f'<div class="card" href="?open={pyhtml.escape(it["id"])}" '
            f'onclick="savePageScroll()" title="{t}">'
            f'<img src="{pyhtml.escape(it["image"])}" alt="{t}">'
            f'<div class="card-body"><div class="card-title">{t}</div>'
            f'<div class="card-sub">{a}</div>'
            f'<div class="card-meta">{it['description']}</div></div></div>'
        )
    row_html = f"""<div class="rec-row">
  <div class="rail-wrap">
    <div class="rec-title">{pyhtml.escape(title)}</div>
    <div class="rec-rail" id="{row_id}">
    {''.join(cards)}
    </div>
  </div>
</div>"""

    # ensure no leading indentation sneaks in:
    st.markdown(textwrap.dedent(row_html), unsafe_allow_html=True)

def transform_items_for_row(items, show_scores=False, show_ids=False):
    return [
        {
            "id": str(row.book_id),
            "image": row.image_url,
            "title": f"{'['+str(row.book_id)+'] ' if show_ids else ''}{row.original_title if row.original_title is not None or "title" not in row else row.title} ({row.recomm_scores:.2f})" if "recomm_scores" in row and show_scores else f"{'['+str(row.book_id)+'] ' if show_ids else ''}{row.original_title}",
            "authors": row.authors,
            "description": row.mistral_7b[:225]+"...",
        } for i,row in items.iterrows()
    ]

class TOC:
    def __init__(self, object, md: str | None = None, num_sep=2) -> None:
        self.num_sep = num_sep
        self.object = object
        if md:
            headers = [(x.split(" ")[0], " ".join(x.split(" ")[1:]), " ".join(x.split(" ")[1:]).lower().replace(" ", "-"))  for x in md.split("\n") if x.startswith("#")]
            self.toc = [f"{'&nbsp;'*self.num_sep*len(x)}[{y}](#{z})" for x,y,z in headers]
        else:
            self.toc = []

    def __call__(self, sep: str = "<br>") -> str:
        self.object.markdown(sep.join(self.toc), unsafe_allow_html=True)

    def append_md(self, md: str) -> None:
        headers = [(x.split(" ")[0], " ".join(x.split(" ")[1:]), " ".join(x.split(" ")[1:]).lower().replace(" ", "-"))  for x in md.split("\n") if x.startswith("#")]
        self.toc += [f"{'&nbsp;'*self.num_sep*len(x)}[{y}](#{z})" for x,y,z in headers]

    def append(self, object: Any, name:str, anchor:str, indent:int=0):
        if object:
            if indent==0:
                object.header(name, anchor=anchor)
            else:
                object.subheader(name, anchor=anchor)
        self.toc.append(f"{'&nbsp;'*self.num_sep*indent}[{name}](#{anchor})")


class CompressedSparseElsaRecommender:
    def __init__(self, A, items, items_idx, B, segments):
        self.A = norm_csr(A)
        self.B = norm_csr(B)
        self.items = items
        self.items_idx = items_idx
        self.segments=segments
        self.segment_names = np.array(list(segments.keys()))

    def get_items_by_indices(self, idxs):
        return self.items.loc[self.items_idx].iloc[idxs]

    def get_user_history(self, x, metadata=None):
        if isinstance(x, scipy.sparse.spmatrix):
            x=x.toarray()
        indices = np.where(x>0)[1]
        if metadata is None:
            ret = self.get_items_by_indices(indices)
        else:
            ret = self.get_items_by_indices(indices)[metadata]
        return ret
    
    def recommend_items(self, x, k=10, metadata=None, segment=None):
        if isinstance(x, scipy.sparse.spmatrix):
            x=x.toarray()
        if segment is not None:
            inds = pd.Series(self.items_idx).to_frame().reset_index().set_index(0).loc[self.segments[segment]['precomputed_segment_items'].book_id.astype(str).to_numpy()].to_numpy().squeeze()
            x_seg = np.zeros([1, 10000], dtype="float32")
            x_seg[0,inds]=1.
        else:
            x_seg = np.ones([1, 10000], dtype="float32")
        # (1-x)*preedictions => ban reminder
        scores, indices = topk((1-x)*(x@self.A@self.A.T)*x_seg, k)
        recomms = self.get_items_by_indices(indices[0])
        recomms["recomm_scores"]=scores[0]
        if metadata is not None:
            recomms=recomms[metadata]
        return recomms

    def recommend_segments(self, x, k=5):
        if isinstance(x, scipy.sparse.spmatrix):
            x=x.toarray()
        scores, indices = topk((x@self.A@self.B.T), k)
        return {k:v for k,v in dict(zip(list(self.segment_names[indices[0]]), scores[0])).items() if v>0}

    def explore_user_segments(self, x, segids, userid, scores=None, only_active=False, sort_latents=True):
        colors = PALETTE[:len(segids)]

        # user latents
        U=x@self.A
        _,i=np.where(U.toarray()!=0)
        U.toarray()[0,i]
        df=pd.DataFrame(U.toarray()[0,i]).reset_index()
        df["index"]=i
        df=df[["index",0]]
        df.columns=["latent", "val"]
    
    
        # prepare segment dataframes
        seg_dfs = []
        lat = []    
        for i in range(len(segids)):
            S=self.B.toarray()[np.where(self.segment_names==segids[i])[0]]
            _,j=np.where(S!=0)

            df_seg=pd.DataFrame(S[0,j]).reset_index()
            df_seg["index"]=j
            df_seg=df_seg[["index",0]]
            df_seg.columns=["latent", "val"]
            seg_dfs.append(df_seg)
            lat.append(df_seg[df_seg.latent.isin(df.latent)].latent.to_numpy())
        # user latents

        if only_active:
            all_dfs = pd.concat(seg_dfs)
            df = df[df.latent.isin(all_dfs.latent)]
            if sort_latents:
                df = df.set_index("latent", drop=True)
                df = df.loc[np.hstack(lat)].reset_index(drop=False)

        fig = px.bar(df, x='latent', y='val',
                title=f"Latent factors for the test user {userid}",
                labels={'latent': 'Latent Dimension', 'val': 'Value'},
                color_discrete_sequence=['lightgray'])  # fixed gray color

        # remove colorbar (since we’re not using continuous color)
        fig.update_traces(marker_color='lightgray', showlegend=False)
        fig.update_layout(coloraxis_showscale=False)

        # optional: cleaner look
        fig.update_layout(
            xaxis=dict(type='category'),
            plot_bgcolor='white',
            yaxis_title='Value'
        )
        
        for i in range(len(segids)):
            df_seg=seg_dfs[i]
            df_seg=df_seg[df_seg.latent.isin(df.latent)]
            # Example scatter data (you can replace this with your own)
            scatter_x = df_seg.latent.to_list()
            scatter_y = df_seg.val.to_list()
            
            seg=segids[i]
            xs = df_seg['latent'].tolist()
            ys = df_seg['val'].tolist()

            # draw vertical lines
            for x, y in zip(xs, ys):
                fig.add_trace(go.Scatter(
                    x=[x, x],
                    y=[0, y],
                    mode='lines',
                    line=dict(color=colors[i], width=2),
                    name=f'{seg}' + (f' ({scores[i]:.2f})' if scores else ''),
                    legendgroup=f'seg{i}',
                    showlegend=False  # legend only on marker trace
                ))

            # triangle tips (up for +, down for -)
            symbols = ['triangle-up' if v >= 0 else 'triangle-down' for v in ys]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(
                    color=colors[i],
                    symbol=symbols,
                    size=15,
                    #line=dict(width=1, color='white')  # crisp tips
                ),
                name=f'{segids[i]}' + (f'({scores[i]:.2f})' if scores else ''),  # Legend label
                legendgroup=f'seg{i}',
                #showlegend=False                    # keep legend from the bar trace
            ))
        
        # Layout tweaks
        fig.update_layout(
            title=f'Latent factors analysis for the test user "{userid}"',
            xaxis_title='Latent Dimension',
            yaxis_title='Value',
            plot_bgcolor='white',
            xaxis=dict(type='category'),
            showlegend=True
        )
        return fig
