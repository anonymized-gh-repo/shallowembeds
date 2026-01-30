import streamlit as st
from utils import TOC

# prepare placeholder for table of contents
toc = TOC(st.sidebar.empty())

st.markdown ("""## Efficient Learning of Sparse Representations from Interactions

This mini-page accompanies our paper [link TBA] and provides additional materials that 
could not fit into the manuscript. The source code is at https://github.com/anonymized-gh-repo/shallowembeds. 
The source code for this site is in the branch "demo".
""")

st.markdown("""### Abstract""")

st.markdown("""### Navigation""")  

st.markdown("""### Citation""")  
st.markdown("""TBA""")  

st.markdown("""### Acknowledgments""")
st.markdown("""
Book cover images are provided by the Open Library Covers API. © Open Library. https://openlibrary.org
""")

toc()

