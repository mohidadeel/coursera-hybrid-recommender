import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import nltk

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Coursera Search & Recommendation Dashboard", page_icon="🎓", layout="wide")

# Ensure VADER is downloaded
nltk.download('vader_lexicon', quiet=True)

# --- BACKEND ENGINE (CACHED) ---
@st.cache_resource(show_spinner="Optimizing Search Engine and ML Pipelines...")
def load_and_train_system():
    # 1. Load Pre-Joined Mini Dataset
    master_df = pd.read_csv('coursera_mini_master.csv')

    # 2. Sentiment Profiling
    sia = SentimentIntensityAnalyzer()
    master_df['reviews'] = master_df['reviews'].fillna('').astype(str)
    master_df['sentiment_score'] = master_df['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
    master_df['adjusted_rating'] = (master_df['rating'] + (0.15 * master_df['sentiment_score'])).clip(1.0, 5.0)

    # 3. Hidden SVD Model Baseline Training (Removes need for User ID input)
    reader = Reader(rating_scale=(1.0, 5.0))
    data_for_svd = Dataset.load_from_df(master_df[['reviewers', 'course_id', 'adjusted_rating']], reader)
    trainset = data_for_svd.build_full_trainset()
    svd_model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
    svd_model.fit(trainset)

    # 4. Build Content Matrix (TF-IDF) over Unique Course Corpus
    unique_courses = master_df.drop_duplicates(subset=['course_id']).copy().reset_index(drop=True)
    
    # Pre-cleaning enrollment fields for summary metrics inside unique_courses
    def clean_enrollment(val):
        if pd.isna(val): return 0
        val = str(val).lower().replace(',', '')
        if 'm' in val: return float(val.replace('m', '')) * 1000000
        elif 'k' in val: return float(val.replace('k', '')) * 1000
        try: return float(val)
        except ValueError: return 0
    
    unique_courses['enrolled_numeric'] = unique_courses['course_students_enrolled'].apply(clean_enrollment)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(unique_courses['name'].fillna(''))

    return master_df, unique_courses, svd_model, tfidf, tfidf_matrix

# Initialize backend pipeline
master_df, unique_courses, svd_model, tfidf, tfidf_matrix = load_and_train_system()

# --- FRONTEND UI ---
st.title("🎓 Smart Coursera Discovery & Analytics Platform")
st.markdown("This app uses a Sentiment-Aware Hybrid Recommender to bridge the Sentiment-Rating Gap, ensuring you find courses based on true pedagogical value.")

# --- SIDEBAR (Upgraded Layout) ---
st.sidebar.header("🔍 Course Discovery Controls")

# Modification 1: Replaced dropdown selection with an open-ended concept search input box
search_query = st.sidebar.text_input(
    "What topic do you want to learn today?", 
    value="Data Science", 
    help="Type keywords like 'Python', 'Machine Learning', 'History', or 'Business'"
)

selected_difficulty = st.sidebar.selectbox(
    "Pedagogical Difficulty Filter:", 
    ["Any", "Beginner", "Mixed", "Intermediate", "Advanced"]
)

top_n_slider = st.sidebar.slider(
    "Maximum Recommendations to Surface:", 
    min_value=3, max_value=15, value=5
)

# --- MAIN DISPLAY TABS ---
tab1, tab2 = st.tabs(["🎯 Live Search & Recommendations", "📊 Deep System Analytics Dashboard"])

# TAB 1: ADVANCED RECOMMENDATION SEARCH ENGINE
with tab1:
    if search_query.strip() == "":
        st.warning("Please enter a keyword or learning topic in the sidebar to begin searching.")
    else:
        st.subheader(f"Analyzing Corpus for Topic: '{search_query}'")
        
        # FEATURE 1: DYNAMIC SEARCH VIA VECTOR TRANSFORMATION
        # Transforms the user's raw text input directly into the TF-IDF space
        query_vector = tfidf.transform([search_query])
        content_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Map similarity arrays back to the candidate course frame
        candidates = unique_courses.copy()
        candidates['content_match_score'] = content_similarities
        
        # Sort initially by raw textual keyword relevance
        candidates = candidates.sort_values(by='content_match_score', ascending=False)
        
        # Apply Pedagogical Difficulty Gating
        if selected_difficulty != "Any":
            candidates = candidates[candidates['course_difficulty'] == selected_difficulty]
            
        if candidates.empty or candidates['content_match_score'].max() == 0:
            st.error(f"No courses matched your query or difficulty tier. Try adjusting your sidebar entries.")
        else:
            # FEATURE 2: HIDDEN COLLABORATIVE INFERENCE PREDICTION
            # Evaluates candidate rows against the SVD matrix using an anonymous baseline profile
            hidden_user = "anonymous_learner"
            candidates['predicted_score'] = candidates['course_id'].apply(
                lambda x: svd_model.predict(hidden_user, x).est
            )
            
            # Combine content match and latent SVD sorting weights
            candidates['hybrid_rank_metric'] = (candidates['content_match_score'] * 0.5) + (candidates['predicted_score'] / 5.0 * 0.5)
            final_sorted_recs = candidates.sort_values(by='hybrid_rank_metric', ascending=False).head(top_n_slider)
            
            # FEATURE 3: NEW DYNAMIC HIGHLIGHT KPI CARDS FOR THE #1 MATCH
            top_match = final_sorted_recs.iloc[0]
            st.markdown("### 🏆 Top Algorithmic Match Overview")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.metric("Course Title", f"{top_match['name'][:25]}...")
            with kpi_col2:
                st.metric("Provider Institution", top_match['course_organization'])
            with kpi_col3:
                st.metric("Pedagogical Difficulty", top_match['course_difficulty'])
            with kpi_col4:
                st.metric("Adjusted Rating Profile", f"{round(top_match['adjusted_rating'], 2)} / 5.0")
                
            st.write("---")
            
            # Display Primary Recommendations Dataframe
            st.markdown("### 🎯 Curated Best-Fit Learning Path Suggestions")
            display_df = final_sorted_recs[['name', 'course_organization', 'course_difficulty', 'course_Certificate_type', 'course_students_enrolled', 'predicted_score']].copy()
            display_df['predicted_score'] = display_df['predicted_score'].apply(lambda x: f"{x:.2f} ★")
            display_df.columns = ['Course Name', 'Offered By', 'Difficulty', 'Credential Type', 'Total Registrations', 'Predicted Quality Rank']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.write("---")
            
            # FEATURE 4: NEW INTERACTIVE SIDE-BY-SIDE COURSE COMPARATOR
            st.markdown("### 📊 Interactive Peer-Course Comparator")
            st.markdown("Select specific courses from the recommended outputs below to view an immediate side-by-side descriptive metric audit.")
            
            comparison_selection = st.multiselect(
                "Choose up to 3 courses to cross-evaluate:",
                options=final_sorted_recs['name'].tolist(),
                default=final_sorted_recs['name'].head(2).tolist()
            )
            
            if comparison_selection:
                comp_data = final_sorted_recs[final_sorted_recs['name'].isin(comparison_selection)]
                
                # Reshape data into a highly readable vertical matrix comparison view
                comp_table = pd.DataFrame({
                    "Metric Parameter": [
                        "Partner Institution", 
                        "Difficulty Level", 
                        "Baseline Course Rating", 
                        "Text Sentiment Adjusted Rating", 
                        "Raw Review Sentiment Polarity",
                        "Platform Enrollment Footprint"
                    ]
                })
                
                for idx, row in comp_data.iterrows():
                    short_name = f"{row['name'][:30]}..."
                    comp_table[short_name] = [
                        row['course_organization'],
                        row['course_difficulty'],
                        f"{row['course_rating']} / 5.0",
                        f"{row['adjusted_rating']:.2f} / 5.0",
                        f"{row['sentiment_score']:.4f}",
                        row['course_students_enrolled']
                    ]
                    
                st.table(comp_table)

# TAB 2: SYSTEM VALIDATION & COMPREHENSIVE APPENDIX
with tab2:
    st.subheader("System Diagnostics & Exploratory Data Analysis")
    st.markdown("Review the analytical framework verifying the performance parameters of the system.")
    
    image_folder = 'report_images_extended'
    
    if os.path.exists(image_folder):
        st.markdown("### 📈 Core System Foundations & The Sentiment Gap")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(f"{image_folder}/Fig_4_01_Difficulty.png", caption="Figure 4.1: Corpus Difficulty Breakdown", use_container_width=True)
            st.image(f"{image_folder}/Fig_4_08_SentimentGap.png", caption="Figure 4.2: Boxplot Verification of the Sentiment-Rating Gap", use_container_width=True)
        with col_g2:
            st.image(f"{image_folder}/Fig_4_02_Institutions.png", caption="Figure 4.3: Top Institutional Distribution", use_container_width=True)
            st.image(f"{image_folder}/Fig_4_10_RatingShift.png", caption="Figure 4.4: Density Analysis of Rating Calibration Curves", use_container_width=True)
            
        st.write("---")
        st.markdown("### 🔬 Machine Learning Evaluation & Predictive Rigor")
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            st.image(f"{image_folder}/Fig_5_01_Accuracy.png", caption="Figure 4.5: Error Metric Comparison (RMSE/MAE)", use_container_width=True)
            st.image(f"{image_folder}/Fig_5_04_NDCG.png", caption="Figure 4.6: Ranking Quality Evaluation", use_container_width=True)
        with col_g4:
            st.image(f"{image_folder}/Fig_5_02_ErrorDist.png", caption="Figure 4.7: SVD Model Prediction Error Spread", use_container_width=True)
            st.image(f"{image_folder}/Fig_5_08_Latency.png", caption="Figure 4.8: Query System Latency Analysis", use_container_width=True)
            
        with st.expander("📂 Click to Expand Complete Technical Report Visual Appendix (All 22 Graphs)"):
            all_images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
            for img in all_images:
                st.image(f"{image_folder}/{img}", caption=f"System Verification Artifact: {img}", use_container_width=True)
    else:
        st.warning("The 'report_images_extended' image assets directory could not be resolved locally.")
