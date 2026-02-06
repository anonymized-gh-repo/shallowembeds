import streamlit as st
from utils import TOC

# prepare placeholder for table of contents
toc = TOC(st.sidebar.empty())

st.markdown ("""## Language Embeddings Meet Shallow Autoencoders

This mini-page accompanies our paper [link TBA] and provides additional materials that 
could not fit into the manuscript. The source code is at 
https://github.com/anonymized-gh-repo/shallowembeds. 
The source code for this site is in the branch "demo".
""")

st.markdown("""### Abstract

Shallow autoencoders are appealing recommenders due to their simplicity, 
scalability, and competitive retrieval quality, but they struggle in strict 
cold-start settings where new items have no interactions. We propose an 
inductive shallow autoencoder that leverages item side information 
(language embeddings) by fixing the decoder to item features and learning 
only an encoder in the same semantic space. To prevent trivial self-reconstruction 
without enforcing a hard zero diagonal, we introduce diagonal gating: a leave-one-item-out 
objective that blocks the self-copy shortcut only for the item being updated while 
retaining context from the rest of the user history. An efficient ALS-style optimization 
trains the model. Experiments on three real-world benchmarks show consistent gains 
over strong cold-start baselines, including other shallow autoencoders, and support 
lightweight (cross-domain) semantic user modeling.

""")

st.markdown("""### Appendix

We provide additional experimental results, detailed descriptions of the datasets and baselines, implementation details, and a complexity analysis in the appendix.

[<img src="https://raw.githubusercontent.com/anonymized-gh-repo/shallowembeds/main/appendix_thumbnail.jpg" width="400px">](https://raw.githubusercontent.com/anonymized-gh-repo/shallowembeds/main/appendix.pdf)

""", unsafe_allow_html=True)

st.markdown("""### Live Demo""")  

st.markdown("""### Citation""")  
st.markdown("""TBA""")  

st.markdown("""### Acknowledgments""")
st.markdown("""
Book cover images are provided by the Open Library Covers API. © Open Library. https://openlibrary.org
""")
st.markdown("""
Movie poster images are provided by The Movies Database API. © TMDB https://www.themoviedb.org 
""")
toc()

