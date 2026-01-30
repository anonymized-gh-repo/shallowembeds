import streamlit as st

st.set_page_config(page_title="Paper Demo", page_icon=":rocket:", layout="wide")

home = st.Page("introduction.py", title="Home", icon="🏠")
report = st.Page("results.py", title="Results", icon="📈")
admin = st.Page("demo.py", title="Live Demo", icon="🖥️")

nav = st.navigation([home, report, admin])   # select current page
nav.run()