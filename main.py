import streamlit as st

st.set_page_config(page_title="Paper Demo", page_icon=":rocket:", layout="wide")

pages = [
    st.Page("introduction.py", title="Home", icon="🏠"),
    st.Page("results.py", title="Appendix", icon="📈"),
    st.Page("demo.py", title="Live Demo", icon="🖥️"),
]

nav = st.navigation(pages)   # select current page
nav.run()