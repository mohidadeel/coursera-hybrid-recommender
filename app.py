import streamlit as st
import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import nltk

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Coursera Recommender", page_icon="🎓", layout="wide")

# Ensure VADER is downloaded
nltk.download('vader_lexicon', quiet=True)

# --- BACKEND ENGINE (CACHED) ---
@st.cache_resource(show_spinner="Initializing Hybrid Engine (This takes ~15 seconds on startup)...")
def load_and_train_system():
    # 1. Load the Pre-Joined Mini Dataset
    # We skip the heavy joining process because it is already done!
    master_df = pd.read_csv('coursera_mini_master.csv')

    # 2. Sentiment Adjustments
    sia = SentimentIntensityAnalyzer()
    master_df['reviews'] = master_df['reviews'].fillna('').astype(str)
    master_df['sentiment_score'] = master_df['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
    master_df['adjusted_rating'] = (master_df['rating'] + (0.15 * master_df['sentiment_score'])).clip(1.0, 5.0)

    # 3. Train SVD (Collaborative Filtering)
    reader = Reader(rating_scale=(1.0, 5.0))
    data_for_svd = Dataset.load_from_df(master_df[['reviewers', 'course_id', 'adjusted_rating']], reader)
    trainset = data_for_svd.build_full_trainset()
    svd_model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02)
    svd_model.fit(trainset)

    # 4. Build Content Matrix (TF-IDF)
    unique_courses = master_df.drop_duplicates(subset=['course_id']).copy().reset_index(drop=True)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(unique_courses['name'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return master_df, unique_courses, svd_model, cosine_sim

# Initialize the engine
master_df, unique_courses, svd_model, cosine_sim = load_and_train_system()

# Extract lists for the UI dropdowns
all_users = master_df['reviewers'].dropna().unique().tolist()
all_courses = unique_courses['name'].dropna().unique().tolist()

# --- FRONTEND UI ---
st.title("🎓 Sentiment-Aware Coursera Recommender")
st.markdown("Welcome to the Hybrid Engine. This system uses Natural Language Processing and Matrix Factorization to recommend courses based on true pedagogical value.")

# --- SIDEBAR (User Inputs) ---
st.sidebar.header("🔍 User Profile & Search")
selected_user = st.sidebar.selectbox("Select User ID (For Collaborative Prediction):", all_users[:100]) # Limiting to 100 for UI speed
selected_course = st.sidebar.selectbox("What course or topic did you enjoy previously?", all_courses)
selected_difficulty = st.sidebar.selectbox("Pedagogical Difficulty Filter:", ["Any", "Beginner", "Mixed", "Intermediate", "Advanced"])
top_n_slider = st.sidebar.slider("Number of Recommendations:", min_value=3, max_value=15, value=5)

# --- MAIN DISPLAY TABS ---
tab1, tab2 = st.tabs(["🎯 Top Recommendations", "📊 Analytics & Data Dashboard"])

# TAB 1: RECOMMENDATION LOGIC
with tab1:
    st.subheader(f"Generating tailored recommendations for {selected_user}...")
    
    if st.button("Generate Recommendations", type="primary"):
        try:
            # 1. Content Similarity
            idx = unique_courses.index[unique_courses['name'] == selected_course].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top 30 semantic matches
            course_indices = [i[0] for i in sim_scores[1:31]]
            candidate_courses = unique_courses.iloc[course_indices].copy()
            
            # 2. Difficulty Filter
            if selected_difficulty != "Any":
                candidate_courses = candidate_courses[candidate_courses['course_difficulty'] == selected_difficulty]
                
            # 3. SVD Prediction
            if candidate_courses.empty:
                st.warning(f"No courses found matching the difficulty level: {selected_difficulty}. Try changing the filter.")
            else:
                candidate_courses['Predicted Score (out of 5)'] = candidate_courses['course_id'].apply(
                    lambda x: round(svd_model.predict(selected_user, x).est, 2)
                )
                
                # 4. Final Formatting
                final_recs = candidate_courses.sort_values(by='Predicted Score (out of 5)', ascending=False)
                display_df = final_recs[['name', 'course_organization', 'course_difficulty', 'Predicted Score (out of 5)']].head(top_n_slider)
                display_df.columns = ['Course Title', 'Institution', 'Difficulty Level', 'Match Strength (Predicted Rating)']
                
                # Render beautiful dataframe
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.success("Recommendations generated successfully using SBERT/TF-IDF and SVD Hybridization!")
                
        except IndexError:
            st.error("Error generating recommendations. Please try a different course.")

# TAB 2: ANALYTICS DASHBOARD
with tab2:
    st.subheader("System Diagnostics & Exploratory Data Analysis")
    st.markdown("Explore the 22 rigorous data science visualizations that prove the efficacy of the Sentiment-Rating Gap and the Hybrid Model.")
    
    image_folder = 'report_images_extended'
    
    if os.path.exists(image_folder):
        # Section 1: EDA
        st.markdown("### 1. Market Overview & Difficulty Profiling")
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"{image_folder}/Fig_4_01_Difficulty.png", use_container_width=True)
        with col2:
            st.image(f"{image_folder}/Fig_4_02_Institutions.png", use_container_width=True)

        # Section 2: The Sentiment Gap
        st.markdown("### 2. The Sentiment-Rating Gap Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st.image(f"{image_folder}/Fig_4_08_SentimentGap.png", use_container_width=True)
        with col4:
            st.image(f"{image_folder}/Fig_4_10_RatingShift.png", use_container_width=True)
            
        # Section 3: ML Performance
        st.markdown("### 3. Machine Learning Evaluation (SVD)")
        col5, col6 = st.columns(2)
        with col5:
            st.image(f"{image_folder}/Fig_5_01_Accuracy.png", use_container_width=True)
        with col6:
            st.image(f"{image_folder}/Fig_5_04_NDCG.png", use_container_width=True)
            
        # Expander for the rest of the graphs to keep the UI clean
        with st.expander("View Full Analytics Appendix (All 22 Graphs)"):
            all_images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
            for img in all_images:
                st.image(f"{image_folder}/{img}", caption=img, use_container_width=True)
    else:
        st.warning("The 'report_images_extended' folder was not found. Please ensure it is in the same directory as this app.py script.")
