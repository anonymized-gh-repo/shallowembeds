import streamlit as st
from pathlib import Path
from utils import TOC

# prepare placeholder for table of contents
toc = TOC(st.sidebar.empty(), num_sep=0)

md = Path("supplementary_streamlit.md").read_text(encoding="utf-8")

toc.append_md(md)
st.markdown(md, unsafe_allow_html=True)


toc()