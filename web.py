import streamlit as st
import pandas as pd
import pickle
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import altair as alt

# Load data and model
@st.cache_resource

def load_data():
    df = pd.read_csv("dataset_fir.csv")
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data()

st.title("🔍 Crime Case Similarity Finder")

# User input with example suggestions
suggestions = ["robbery at bank", "missing person", "vehicle theft", "assault at night"]
user_input = st.text_area("📝 Enter FIR Summary or Crime Description", placeholder="E.g., robbery at bank")

# Sidebar filters
st.sidebar.header("🔧 Filter Similar Cases")
status_filter = st.sidebar.multiselect("Status", options=df['status'].unique(), default=list(df['status'].unique()))
location_filter = st.sidebar.multiselect("Location", options=df['location'].unique())

# Initialize session state
if 'df_similar' not in st.session_state:
    st.session_state.df_similar = None
if 'similarities' not in st.session_state:
    st.session_state.similarities = None

if st.button("🔎 Recommend Similar Cases"):
    if user_input.strip() == "":
        st.warning("Please enter a case summary to get recommendations.")
    else:
        user_vector = vectorizer.transform([user_input])
        cos_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
        similar_indices = cos_sim.argsort()[-10:][::-1]
        df_similar = df.iloc[similar_indices].copy()
        df_similar['similarity_score'] = cos_sim[similar_indices] * 100
        st.session_state.df_similar = df_similar
        st.session_state.similarities = cos_sim[similar_indices]

# Show similar cases if available
if st.session_state.df_similar is not None:
    df_similar = st.session_state.df_similar

    # Apply filters
    if status_filter:
        df_similar = df_similar[df_similar['status'].isin(status_filter)]
    if location_filter:
        df_similar = df_similar[df_similar['location'].isin(location_filter)]


    st.subheader("🔗 Top 10 Similar Cases")

    for i, row in df_similar.iterrows():
        with st.expander(f"📁 Case ID: {row['case_id']} - {row['status']}"):
            st.markdown(f"**Similarity Score:** {row['similarity_score']:.2f}%")
            st.markdown(f"**Summary:** {row['summary']}")
            st.markdown(f"**Status:** {'🟢 Closed' if row['status'].lower() == 'closed' else '🔴 Open' if row['status'].lower() == 'open' else '🟡 In Progress'}")
            st.markdown(f"**Reported Date:** {row['date_reported']}")
            st.markdown(f"**Closed Date:** {row['closed_date']}")
            st.markdown(f"**Time to Solve:** {row['time_to_solve']} days")
            st.markdown(f"**Location:** {row['location']}")

    # Timeline Visualization
    st.subheader("⏳ Time to Solve (Days)")
    if not df_similar.empty:
        chart_data = df_similar[['case_id', 'time_to_solve']].dropna()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='case_id:N',
            y='time_to_solve:Q',
            tooltip=['case_id', 'time_to_solve']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    # Map visualization
    st.subheader("🗺️ Case Locations Map")
    map_center = [df_similar.iloc[0]['lat'], df_similar.iloc[0]['lon']]
    case_map = folium.Map(location=map_center, zoom_start=5)
    marker_cluster = MarkerCluster().add_to(case_map)

    heat_data = []
    for _, row in df_similar.iterrows():
        location = [row['lat'], row['lon']]
        heat_data.append(location)
        folium.Marker(
            location=location,
            popup=(f"<b>Case ID:</b> {row['case_id']}<br><b>Status:</b> {row['status']}<br><b>Summary:</b> {row['summary'][:100]}..."),
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    HeatMap(heat_data).add_to(case_map)
    st_folium(case_map, width=700, height=450)

    # Download Option
    st.subheader("📥 Download Results")
    st.download_button("Download as CSV", data=df_similar.to_csv(index=False), file_name="similar_cases.csv", mime="text/csv")

# Show search history
if 'history' not in st.session_state:
    st.session_state.history = []

if user_input and st.button("💾 Save Search"):
    st.session_state.history.insert(0, user_input)
    st.session_state.history = st.session_state.history[:3]

if st.session_state.history:
    st.sidebar.subheader("🕘 Recent Searches")
    for i, past in enumerate(st.session_state.history):
        st.sidebar.write(f"{i+1}. {past}")
