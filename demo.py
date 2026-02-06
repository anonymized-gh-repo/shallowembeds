import streamlit as st
import pandas as pd
from utils import *

# insert css for recommendation rows
init_rows_rendering()

# prepare placeholder for table of contents in the sidebar
toc = TOC(st.sidebar.empty())

#################### read data ######################
@st.cache_data
def read_data():
    CL=pd.read_feather("data/CLUSTERS.feather")
    UBT=pd.read_feather("data/UBT.feather")
    UMT=pd.read_feather("data/UMT.feather")
    UIF=pd.read_feather("data/UIF.feather")
    UIWOP=pd.read_feather("data/UIWOP.feather")
    UBR=pd.read_feather("data/UBR.feather") 
    UMR=pd.read_feather("data/UMR.feather")
    UC=pd.read_feather("data/UC.feather")
    items=pd.read_feather("data/gb10_items.feather")
    items["id"]=items["book_id"]
    items=items.set_index("id")
    movies=pd.read_feather("data/movies.feather")
    movies["id"]=movies["movieId"].astype(int)
    movies["original_title"]=movies["title"]
    movies["mistral_7b"]=movies["description"]
    movies["book_id"]=movies["movieId"].astype(int)
    movies=movies.set_index("id")
    UMR=UMR[UMR.MR.astype(int).isin(movies.index)] # we dont have the metadata for all movies 
    return {
        "Clusters": CL,
        "UserCluster": UC,
        "BookTopics": UBT,
        "MovieTopics": UMT,
        "InteractionsFull": UIF,
        "InteractionsWOP": UIWOP,
        "BookRecomms": UBR,
        "MovieRecomms": UMR,
        "Books": items,
        "Movies": movies,
    }

data = read_data()

#################### get cluster ######################
with st.sidebar:
    # pick a user from the list
    cluster_picked = st.selectbox(
        label="Select a cluster", 
        options=data["Clusters"].apply(lambda row: f"{row.ClusterID} - {row.books_label}", axis=1).to_list(), 
        #index=1011
    )
    cluster_picked = int(cluster_picked.split(" - ")[0])
#################### get user ######################
with st.sidebar:
    # pick a user from the list
    user = st.selectbox(
        label="Select a user", 
        options=data["UserCluster"][data["UserCluster"].Cluster==cluster_picked].UserID.to_numpy(), 
        #index=1011
    )

#################### cluster info ######################

st.markdown(f"## User {user} - analysis")
st.image("data/clusters.png", width=800)

cluster = data["UserCluster"][data["UserCluster"]["UserID"]==user]["Cluster"].item()
_, books_label, movies_label, books_topic, movies_topic = data["Clusters"][data["Clusters"]["ClusterID"]==cluster].iloc[0]
books_topic = books_topic.split("|")
movies_topic = movies_topic.split("|")

bcol, mcol = st.columns(2)
bcol.markdown(f"#### Books: {books_label}")
bcol.markdown(f"Books topics: \n - {"\n - ".join(books_topic)}")
mcol.markdown(f"#### Movies: {movies_label}")
mcol.markdown(f"Movies topics: \n - {"\n - ".join(movies_topic)}")


interacted = data["Books"].loc[data["InteractionsWOP"][data["InteractionsWOP"]["UserID"]==user].IWOP.astype(int).to_list()]
render_row(
    "Book interactions (without popular items)", 
    transform_items_for_row(
        interacted,
        show_scores=False,
        show_ids=True,
    ), 
    row_id=f"history_rail_{0}"
)

book_recomms = data["Books"].loc[data["BookRecomms"][data["BookRecomms"]["UserID"]==user].BR.astype(int).to_list()]
render_row(
    "Book recommendations", 
    transform_items_for_row(
        book_recomms,
        show_scores=False,
        show_ids=True,
    ), 
    row_id=f"book_recomms_rail_{0}"
)

print("Generating")
movie_recomms = data["Movies"].loc[data["MovieRecomms"][data["MovieRecomms"]["UserID"]==user].MR.astype(int).to_list()].iloc[::-1]
movie_recomms["image_url"]=movie_recomms.title.apply(get_movie_poster).fillna(" ")
print(movie_recomms["image_url"])
render_row(
    "Movie recommendations", 
    transform_items_for_row(
        movie_recomms,
        show_scores=False,
        show_ids=True,
    ), 
    row_id=f"movie_recomms_rail_{0}"
)