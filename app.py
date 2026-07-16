import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from collections import defaultdict
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
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

    # 3. Model Training & Evaluation
    reader = Reader(rating_scale=(1.0, 5.0))
    data_for_svd = Dataset.load_from_df(master_df[['reviewers', 'course_id', 'adjusted_rating']], reader)
    
    # 3.1 Train/Test Split for Metrics Calculation
    trainset, testset = train_test_split(data_for_svd, test_size=0.20, random_state=42)
    eval_model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
    eval_model.fit(trainset)
    
    # 3.2 Compute Latency, RMSE, and MAE
    start_time = time.time()
    predictions = eval_model.test(testset)
    inference_time = time.time() - start_time
    avg_latency_ms = (inference_time / len(testset)) * 1000 if len(testset) > 0 else 0
    
    rmse_val = accuracy.rmse(predictions, verbose=False)
    mae_val = accuracy.mae(predictions, verbose=False)
    
    # 3.3 Compute NDCG (Normalized Discounted Cumulative Gain)
    # Group true and predicted ratings by user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
        
    ndcg_scores = []
    for uid, user_ratings in user_est_true.items():
        # NDCG requires at least 2 items to compare ranking
        if len(user_ratings) > 1: 
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            true_ratings = [x[1] for x in user_ratings]
            predicted_ratings = [x[0] for x in user_ratings]
            # sklearn ndcg_score expects 2D arrays
            score = ndcg_score([true_ratings], [predicted_ratings])
            ndcg_scores.append(score)
            
    ndcg_val = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    # 3.4 Compile Metrics Dictionary
    live_metrics = {
        "RMSE (Root Mean Squared Error)": round(rmse_val, 4),
        "MAE (Mean Absolute Error)": round(mae_val, 4),
        "NDCG (Ranking Quality)": round(ndcg_val, 4),
        "Avg Inference Latency (ms)": round(avg_latency_ms, 4)
    }

    # 3.5 Train Final Production Model on 100% of data for actual app use
    full_trainset = data_for_svd.build_full_trainset()
    final_svd_model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
    final_svd_model.fit(full_trainset)

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

    return master_df, unique_courses, final_svd_model, tfidf, tfidf_matrix, live_metrics

# Initialize backend pipeline
master_df, unique_courses, svd_model, tfidf, tfidf_matrix, live_metrics = load_and_train_system()

# --- FRONTEND UI ---
st.title("🎓 Smart Coursera Discovery & Analytics Platform")
st.markdown("This app uses a Sentiment-Aware Hybrid Recommender to bridge the Sentiment-Rating Gap, ensuring you find courses based on true pedagogical value.")

# --- SIDEBAR (Upgraded Layout) ---
st.sidebar.header("🔍 Course Discovery Controls")

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
        
        # DYNAMIC SEARCH VIA VECTOR TRANSFORMATION
        query_vector = tfidf.transform([search_query])
        content_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        candidates = unique_courses.copy()
        candidates['content_match_score'] = content_similarities
        candidates = candidates.sort_values(by='content_match_score', ascending=False)
        
        # Apply Pedagogical Difficulty Gating
        if selected_difficulty != "Any":
            candidates = candidates[candidates['course_difficulty'] == selected_difficulty]
            
        if candidates.empty or candidates['content_match_score'].max() == 0:
            st.error(f"No courses matched your query or difficulty tier. Try adjusting your sidebar entries.")
        else:
            # HIDDEN COLLABORATIVE INFERENCE PREDICTION
            hidden_user = "anonymous_learner"
            candidates['predicted_score'] = candidates['course_id'].apply(
                lambda x: svd_model.predict(hidden_user, x).est
            )
            
            # Combine content match and latent SVD sorting weights
            candidates['hybrid_rank_metric'] = (candidates['content_match_score'] * 0.5) + (candidates['predicted_score'] / 5.0 * 0.5)
            final_sorted_recs = candidates.sort_values(by='hybrid_rank_metric', ascending=False).head(top_n_slider)
            
            # DYNAMIC HIGHLIGHT KPI CARDS
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
            
            # INTERACTIVE SIDE-BY-SIDE COURSE COMPARATOR
            st.markdown("### 📊 Interactive Peer-Course Comparator")
            st.markdown("Select specific courses from the recommended outputs below to view an immediate side-by-side descriptive metric audit.")
            
            comparison_selection = st.multiselect(
                "Choose up to 3 courses to cross-evaluate:",
                options=final_sorted_recs['name'].tolist(),
                default=final_sorted_recs['name'].head(2).tolist()
            )
            
            if comparison_selection:
                comp_data = final_sorted_recs[final_sorted_recs['name'].isin(comparison_selection)]
                
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
    
    # NEW: Live Computed Algorithm Metrics
    st.markdown("### 🧮 Live Algorithm Performance Metrics")
    st.markdown("The following metrics are dynamically calculated in real-time using an 80/20 train/test split of the ingested Coursera dataset.")
    
    metrics_df = pd.DataFrame([live_metrics])
    st.table(metrics_df)
    
    st.write("---")
    
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
            
        with st.expander("📂 Click to Expand Complete Technical Report Visual Appendix (All 22 Graphs)"):
            all_images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
            for img in all_images:
                st.image(f"{image_folder}/{img}", caption=f"System Verification Artifact: {img}", use_container_width=True)
    else:
        st.warning("The 'report_images_extended' image assets directory could not be resolved locally.")
