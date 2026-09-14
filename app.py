import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from collections import defaultdict

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    ndcg_score
)

from surprise import Dataset, Reader, SVD, accuracy

# Optional semantic-search dependencies
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coursera Search & Recommendation Dashboard",
    page_icon="🎓",
    layout="wide"
)

nltk.download("vader_lexicon", quiet=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Sentiment coefficients tested experimentally
SENTIMENT_COEFFICIENTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

# Hybrid weights tested experimentally
CONTENT_WEIGHTS = [0.25, 0.50, 0.75]

TOP_K = 5

SVD_PARAMS = {
    "n_factors": 50,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": RANDOM_STATE
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_enrollment(val):
    """
    Convert enrollment values such as:
    1.2m -> 1200000
    50k  -> 50000
    """
    if pd.isna(val):
        return 0

    val = str(val).lower().replace(",", "").strip()

    try:
        if "m" in val:
            return float(val.replace("m", "")) * 1_000_000

        if "k" in val:
            return float(val.replace("k", "")) * 1_000

        return float(val)

    except ValueError:
        return 0


def get_rating_column(df):
    """
    Supports datasets using either 'rating' or 'course_rating'.
    """
    if "rating" in df.columns:
        return "rating"

    if "course_rating" in df.columns:
        return "course_rating"

    raise ValueError(
        "Dataset must contain either 'rating' or 'course_rating'."
    )


def user_aware_split(df, test_size=0.20, random_state=42):
    """
    Hold out approximately 20% of each user's interactions.

    This prevents the evaluation from simply being an arbitrary
    row-level split and gives each sufficiently active user
    unseen test interactions.
    """

    rng = np.random.RandomState(random_state)

    train_indices = []
    test_indices = []

    for user_id, group in df.groupby("reviewers"):

        indices = group.index.to_numpy()

        if len(indices) < 2:
            train_indices.extend(indices)
            continue

        n_test = max(1, int(round(len(indices) * test_size)))

        n_test = min(
            n_test,
            len(indices) - 1
        )

        test_idx = rng.choice(
            indices,
            size=n_test,
            replace=False
        )

        test_idx = set(test_idx.tolist())

        for idx in indices:
            if idx in test_idx:
                test_indices.append(idx)
            else:
                train_indices.append(idx)

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    return train_df, test_df


def calculate_ranking_metrics(
    predictions,
    k=5,
    relevance_threshold=4.0
):
    """
    Calculate ranking metrics from held-out test predictions.

    Importantly, NDCG is calculated using the original ordering
    of test items rather than sorting predictions first.
    """

    user_groups = defaultdict(list)

    for prediction in predictions:

        uid = prediction.uid
        true_rating = prediction.r_ui
        estimated_rating = prediction.est

        user_groups[uid].append(
            (true_rating, estimated_rating)
        )

    precision_values = []
    recall_values = []
    ndcg_values = []
    hit_values = []

    for uid, values in user_groups.items():

        if len(values) == 0:
            continue

        # Original held-out items
        true_scores = np.array(
            [v[0] for v in values],
            dtype=float
        )

        predicted_scores = np.array(
            [v[1] for v in values],
            dtype=float
        )

        # Rank based ONLY on predictions
        ranking_order = np.argsort(
            predicted_scores
        )[::-1]

        top_indices = ranking_order[:k]

        relevant = true_scores >= relevance_threshold

        recommended_relevant = relevant[top_indices]

        precision = (
            recommended_relevant.sum() / len(top_indices)
            if len(top_indices) > 0 else 0
        )

        total_relevant = relevant.sum()

        recall = (
            recommended_relevant.sum() / total_relevant
            if total_relevant > 0 else 0
        )

        # Correct NDCG:
        # true relevance remains aligned with predictions.
        if len(values) > 1:

            ndcg = ndcg_score(
                [true_scores],
                [predicted_scores],
                k=min(k, len(values))
            )

        else:
            ndcg = 0

        hit = (
            1
            if recommended_relevant.sum() > 0
            else 0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        ndcg_values.append(ndcg)
        hit_values.append(hit)

    return {
        "Precision@5": np.mean(precision_values)
        if precision_values else 0,

        "Recall@5": np.mean(recall_values)
        if recall_values else 0,

        "NDCG@5": np.mean(ndcg_values)
        if ndcg_values else 0,

        "Hit Rate@5": np.mean(hit_values)
        if hit_values else 0
    }


def evaluate_svd_model(
    train_df,
    test_df,
    rating_column,
    sentiment_adjusted=True
):
    """
    Train an SVD model on training interactions only
    and evaluate against held-out RAW ratings.

    This is important because the test ratings are never used
    during model training.
    """

    if sentiment_adjusted:

        train_values = train_df["adjusted_rating"]

    else:

        train_values = train_df[rating_column]

    train_data = pd.DataFrame({
        "user": train_df["reviewers"],
        "item": train_df["course_id"],
        "rating": train_values
    })

    reader = Reader(
        rating_scale=(1.0, 5.0)
    )

    surprise_data = Dataset.load_from_df(
        train_data,
        reader
    )

    trainset = surprise_data.build_full_trainset()

    model = SVD(
        **SVD_PARAMS
    )

    model.fit(trainset)

    predictions = []

    start_time = time.perf_counter()

    for _, row in test_df.iterrows():

        pred = model.predict(
            row["reviewers"],
            row["course_id"]
        )

        predictions.append(pred)

    elapsed = time.perf_counter() - start_time

    actual = [
        p.r_ui
        for p in predictions
    ]

    predicted = [
        p.est
        for p in predictions
    ]

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    ranking = calculate_ranking_metrics(
        predictions,
        k=TOP_K
    )

    latency_ms = (
        elapsed / len(predictions) * 1000
        if predictions
        else 0
    )

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        **ranking,
        "Latency (ms)": latency_ms
    }

    return model, predictions, metrics


def popularity_baseline(train_df, test_df, rating_column):
    """
    Popularity baseline based on course interaction frequency.
    """

    popularity = (
        train_df
        .groupby("course_id")
        .size()
        .to_dict()
    )

    global_mean = train_df[rating_column].mean()

    predictions = []

    for _, row in test_df.iterrows():

        popularity_score = popularity.get(
            row["course_id"],
            0
        )

        max_popularity = max(
            popularity.values()
        ) if popularity else 1

        normalized_popularity = (
            popularity_score / max_popularity
        )

        # Convert popularity into a rating-scale estimate
        estimated_rating = (
            global_mean +
            normalized_popularity *
            (5 - global_mean)
        )

        predictions.append(
            {
                "actual": row[rating_column],
                "prediction": estimated_rating
            }
        )

    actual = [
        x["actual"]
        for x in predictions
    ]

    predicted = [
        x["prediction"]
        for x in predictions
    ]

    return {
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "MAE": mean_absolute_error(
            actual,
            predicted
        )
    }


def bootstrap_significance(
    actual,
    predictions_a,
    predictions_b,
    n_iterations=1000,
    random_state=42
):
    """
    Paired bootstrap significance test.

    Positive difference means model A has lower absolute error
    than model B.
    """

    rng = np.random.RandomState(
        random_state
    )

    actual = np.array(actual)
    predictions_a = np.array(predictions_a)
    predictions_b = np.array(predictions_b)

    errors_a = np.abs(
        actual - predictions_a
    )

    errors_b = np.abs(
        actual - predictions_b
    )

    differences = []

    n = len(actual)

    for _ in range(n_iterations):

        indices = rng.randint(
            0,
            n,
            size=n
        )

        diff = (
            errors_b[indices].mean()
            -
            errors_a[indices].mean()
        )

        differences.append(diff)

    differences = np.array(
        differences
    )

    lower = np.percentile(
        differences,
        2.5
    )

    upper = np.percentile(
        differences,
        97.5
    )

    return lower, upper


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

@st.cache_resource(
    show_spinner="Training recommendation and evaluation pipeline..."
)
def load_and_train_system():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    data_path = "coursera_mini_master.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find {data_path}"
        )

    master_df = pd.read_csv(
        data_path
    )

    rating_column = get_rating_column(
        master_df
    )

    # Required columns
    required_columns = [
        "reviewers",
        "course_id",
        "reviews",
        "name"
    ]

    missing = [
        col
        for col in required_columns
        if col not in master_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    master_df = master_df.copy()

    master_df[rating_column] = pd.to_numeric(
        master_df[rating_column],
        errors="coerce"
    )

    master_df = master_df.dropna(
        subset=[
            "reviewers",
            "course_id",
            rating_column
        ]
    )

    master_df["reviews"] = (
        master_df["reviews"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # 2. SENTIMENT ANALYSIS
    # --------------------------------------------------------

    sia = SentimentIntensityAnalyzer()

    master_df["sentiment_score"] = (
        master_df["reviews"]
        .apply(
            lambda x:
            sia.polarity_scores(x)["compound"]
        )
    )

    # Default production coefficient
    DEFAULT_SENTIMENT_COEFFICIENT = 0.15

    master_df["adjusted_rating"] = (
        master_df[rating_column]
        +
        DEFAULT_SENTIMENT_COEFFICIENT
        * master_df["sentiment_score"]
    ).clip(1.0, 5.0)

    # --------------------------------------------------------
    # 3. USER-AWARE TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_df, test_df = user_aware_split(
        master_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # 4. SVD: RAW RATING BASELINE
    # --------------------------------------------------------

    svd_raw_model, raw_predictions, raw_metrics = (
        evaluate_svd_model(
            train_df,
            test_df,
            rating_column,
            sentiment_adjusted=False
        )
    )

    # --------------------------------------------------------
    # 5. SVD: SENTIMENT-ADJUSTED MODEL
    # --------------------------------------------------------

    svd_sentiment_model, sentiment_predictions, sentiment_metrics = (
        evaluate_svd_model(
            train_df,
            test_df,
            rating_column,
            sentiment_adjusted=True
        )
    )

    # --------------------------------------------------------
    # 6. POPULARITY BASELINE
    # --------------------------------------------------------

    popularity_metrics = popularity_baseline(
        train_df,
        test_df,
        rating_column
    )

    # --------------------------------------------------------
    # 7. SENTIMENT COEFFICIENT ABLATION
    # --------------------------------------------------------

    sentiment_results = []

    for coefficient in SENTIMENT_COEFFICIENTS:

        train_temp = train_df.copy()

        train_temp["temporary_adjusted_rating"] = (
            train_temp[rating_column]
            +
            coefficient
            * train_temp["sentiment_score"]
        ).clip(1.0, 5.0)

        train_values = train_temp[
            "temporary_adjusted_rating"
        ]

        train_data = pd.DataFrame({
            "user": train_temp["reviewers"],
            "item": train_temp["course_id"],
            "rating": train_values
        })

        reader = Reader(
            rating_scale=(1.0, 5.0)
        )

        dataset = Dataset.load_from_df(
            train_data,
            reader
        )

        trainset = dataset.build_full_trainset()

        model = SVD(
            **SVD_PARAMS
        )

        model.fit(trainset)

        predictions = [
            model.predict(
                row["reviewers"],
                row["course_id"]
            )
            for _, row in test_df.iterrows()
        ]

        actual = [
            p.r_ui
            for p in predictions
        ]

        predicted = [
            p.est
            for p in predictions
        ]

        sentiment_results.append({
            "Sentiment Coefficient": coefficient,
            "RMSE": np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            ),
            "MAE": mean_absolute_error(
                actual,
                predicted
            )
        })

    sentiment_ablation_df = pd.DataFrame(
        sentiment_results
    )

    best_sentiment_coefficient = (
        sentiment_ablation_df
        .sort_values("RMSE")
        .iloc[0]["Sentiment Coefficient"]
    )

    # --------------------------------------------------------
    # 8. COURSE CORPUS
    # --------------------------------------------------------

    unique_courses = (
        master_df
        .drop_duplicates(
            subset=["course_id"]
        )
        .copy()
        .reset_index(drop=True)
    )

    if "course_students_enrolled" in unique_courses.columns:

        unique_courses[
            "enrolled_numeric"
        ] = unique_courses[
            "course_students_enrolled"
        ].apply(
            clean_enrollment
        )

    else:

        unique_courses[
            "enrolled_numeric"
        ] = 0

    # --------------------------------------------------------
    # 9. TF-IDF CONTENT MODEL
    # --------------------------------------------------------

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(
        unique_courses["name"]
        .fillna("")
    )

    # --------------------------------------------------------
    # 10. SBERT SEMANTIC MODEL
    # --------------------------------------------------------

    sbert_model = None
    sbert_embeddings = None

    if SBERT_AVAILABLE:

        try:

            sbert_model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            course_text = (
                unique_courses["name"]
                .fillna("")
                .astype(str)
            )

            sbert_embeddings = (
                sbert_model.encode(
                    course_text.tolist(),
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            )

            sbert_embeddings = np.asarray(
                sbert_embeddings,
                dtype="float32"
            )

        except Exception:

            sbert_model = None
            sbert_embeddings = None

    # --------------------------------------------------------
    # 11. FAISS SEMANTIC INDEX
    # --------------------------------------------------------

    faiss_index = None

    if (
        FAISS_AVAILABLE
        and sbert_embeddings is not None
    ):

        dimension = (
            sbert_embeddings.shape[1]
        )

        faiss_index = faiss.IndexFlatIP(
            dimension
        )

        faiss_index.add(
            sbert_embeddings
        )

    # --------------------------------------------------------
    # 12. FULL PRODUCTION SVD MODEL
    # --------------------------------------------------------

    production_data = pd.DataFrame({
        "user": master_df["reviewers"],
        "item": master_df["course_id"],
        "rating": master_df["adjusted_rating"]
    })

    reader = Reader(
        rating_scale=(1.0, 5.0)
    )

    full_dataset = Dataset.load_from_df(
        production_data,
        reader
    )

    full_trainset = (
        full_dataset
        .build_full_trainset()
    )

    final_svd_model = SVD(
        **SVD_PARAMS
    )

    final_svd_model.fit(
        full_trainset
    )

    # --------------------------------------------------------
    # 13. SEARCH LATENCY BENCHMARK
    # --------------------------------------------------------

    benchmark_query = "machine learning data science"

    start = time.perf_counter()

    benchmark_vector = tfidf.transform(
        [benchmark_query]
    )

    cosine_similarity(
        benchmark_vector,
        tfidf_matrix
    )

    tfidf_latency_ms = (
        time.perf_counter() - start
    ) * 1000

    sbert_latency_ms = None
    faiss_latency_ms = None

    if sbert_model is not None:

        start = time.perf_counter()

        query_embedding = (
            sbert_model.encode(
                [benchmark_query],
                normalize_embeddings=True
            )
        )

        cosine_similarity(
            query_embedding,
            sbert_embeddings
        )

        sbert_latency_ms = (
            time.perf_counter() - start
        ) * 1000

    if faiss_index is not None:

        start = time.perf_counter()

        query_embedding = (
            sbert_model.encode(
                [benchmark_query],
                normalize_embeddings=True
            )
        ).astype("float32")

        faiss_index.search(
            query_embedding,
            min(10, len(unique_courses))
        )

        faiss_latency_ms = (
            time.perf_counter() - start
        ) * 1000

    # --------------------------------------------------------
    # 14. HYBRID WEIGHT EXPERIMENT
    # --------------------------------------------------------

    # Content quality is estimated from the held-out course
    # title similarity against the title corpus.
    #
    # This is a model-selection diagnostic rather than a
    # replacement for user-level ranking evaluation.

    hybrid_results = []

    # Use SVD predictions from held-out interactions
    svd_pred_map = {}

    for prediction in sentiment_predictions:

        svd_pred_map[
            (prediction.uid, prediction.iid)
        ] = prediction.est

    course_lookup = unique_courses.set_index(
        "course_id"
    )

    for content_weight in CONTENT_WEIGHTS:

        hybrid_scores = []
        actual_scores = []

        for _, row in test_df.iterrows():

            course_id = row["course_id"]

            svd_prediction = svd_pred_map.get(
                (
                    row["reviewers"],
                    course_id
                ),
                train_df[
                    "adjusted_rating"
                ].mean()
            )

            # Course title similarity against itself is 1,
            # therefore use average similarity to other corpus
            # entries as a stable content-quality proxy.
            if course_id in course_lookup.index:

                idx = course_lookup.index.get_loc(
                    course_id
                )

                row_vector = tfidf_matrix[idx]

                similarities = (
                    cosine_similarity(
                        row_vector,
                        tfidf_matrix
                    ).flatten()
                )

                similarities[idx] = 0

                content_score = (
                    similarities.max()
                    if len(similarities) > 1
                    else 0
                )

            else:

                content_score = 0

            content_score = float(
                np.clip(
                    content_score,
                    0,
                    1
                )
            )

            hybrid_score = (
                content_weight
                * content_score
                +
                (1 - content_weight)
                * (svd_prediction / 5.0)
            )

            hybrid_scores.append(
                hybrid_score
            )

            actual_scores.append(
                row[rating_column] / 5.0
            )

        hybrid_rmse = np.sqrt(
            mean_squared_error(
                actual_scores,
                hybrid_scores
            )
        )

        hybrid_results.append({
            "Content Weight": content_weight,
            "SVD Weight": 1 - content_weight,
            "Hybrid RMSE": hybrid_rmse
        })

    hybrid_weights_df = pd.DataFrame(
        hybrid_results
    )

    best_hybrid_weight = (
        hybrid_weights_df
        .sort_values("Hybrid RMSE")
        .iloc[0]["Content Weight"]
    )

    # --------------------------------------------------------
    # 15. COMPILE METRICS
    # --------------------------------------------------------

    live_metrics = {
        "SVD Raw RMSE": round(
            raw_metrics["RMSE"],
            4
        ),

        "SVD Raw MAE": round(
            raw_metrics["MAE"],
            4
        ),

        "Sentiment-SVD RMSE": round(
            sentiment_metrics["RMSE"],
            4
        ),

        "Sentiment-SVD MAE": round(
            sentiment_metrics["MAE"],
            4
        ),

        "Precision@5": round(
            sentiment_metrics["Precision@5"],
            4
        ),

        "Recall@5": round(
            sentiment_metrics["Recall@5"],
            4
        ),

        "NDCG@5": round(
            sentiment_metrics["NDCG@5"],
            4
        ),

        "Hit Rate@5": round(
            sentiment_metrics["Hit Rate@5"],
            4
        ),

        "SVD Latency (ms)": round(
            sentiment_metrics["Latency (ms)"],
            4
        ),

        "TF-IDF Search Latency (ms)": round(
            tfidf_latency_ms,
            4
        ),

        "SBERT Search Latency (ms)": (
            round(sbert_latency_ms, 4)
            if sbert_latency_ms is not None
            else None
        ),

        "FAISS Search Latency (ms)": (
            round(faiss_latency_ms, 4)
            if faiss_latency_ms is not None
            else None
        )
    }

    # --------------------------------------------------------
    # 16. BASELINE TABLE
    # --------------------------------------------------------

    baseline_df = pd.DataFrame([
        {
            "Model": "Popularity Baseline",
            "RMSE": popularity_metrics["RMSE"],
            "MAE": popularity_metrics["MAE"]
        },

        {
            "Model": "SVD",
            "RMSE": raw_metrics["RMSE"],
            "MAE": raw_metrics["MAE"]
        },

        {
            "Model": "Sentiment-Aware SVD",
            "RMSE": sentiment_metrics["RMSE"],
            "MAE": sentiment_metrics["MAE"]
        }
    ])

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {
        "master_df": master_df,
        "train_df": train_df,
        "test_df": test_df,
        "unique_courses": unique_courses,
        "rating_column": rating_column,

        "svd_model": final_svd_model,

        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,

        "sbert_model": sbert_model,
        "sbert_embeddings": sbert_embeddings,

        "faiss_index": faiss_index,

        "raw_metrics": raw_metrics,
        "sentiment_metrics": sentiment_metrics,
        "baseline_df": baseline_df,

        "sentiment_ablation_df":
            sentiment_ablation_df,

        "hybrid_weights_df":
            hybrid_weights_df,

        "best_sentiment_coefficient":
            best_sentiment_coefficient,

        "best_hybrid_weight":
            best_hybrid_weight,

        "live_metrics":
            live_metrics
    }


# ============================================================
# LOAD SYSTEM
# ============================================================

try:

    system = load_and_train_system()

except Exception as e:

    st.error(
        f"System initialisation failed: {e}"
    )

    st.stop()


master_df = system["master_df"]
train_df = system["train_df"]
test_df = system["test_df"]

unique_courses = system["unique_courses"]

rating_column = system["rating_column"]

svd_model = system["svd_model"]

tfidf = system["tfidf"]
tfidf_matrix = system["tfidf_matrix"]

sbert_model = system["sbert_model"]
sbert_embeddings = system["sbert_embeddings"]

faiss_index = system["faiss_index"]

live_metrics = system["live_metrics"]

baseline_df = system["baseline_df"]

sentiment_ablation_df = (
    system["sentiment_ablation_df"]
)

hybrid_weights_df = (
    system["hybrid_weights_df"]
)

best_sentiment_coefficient = (
    system["best_sentiment_coefficient"]
)

best_hybrid_weight = (
    system["best_hybrid_weight"]
)


# ============================================================
# FRONTEND
# ============================================================

st.title(
    "🎓 Smart Coursera Discovery & Analytics Platform"
)

st.markdown(
    """
    This platform implements a **Sentiment-Aware Hybrid
    Course Recommendation System** combining:

    - Collaborative filtering using SVD
    - VADER sentiment analysis
    - TF-IDF lexical retrieval
    - SBERT semantic embeddings
    - FAISS vector search where available
    - Pedagogical difficulty filtering
    - Hybrid recommendation ranking
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔍 Course Discovery Controls"
)

search_query = st.sidebar.text_input(
    "What topic do you want to learn today?",
    value="Data Science",
    help=(
        "Try Python, Machine Learning, History, "
        "Business or Data Science."
    )
)

selected_difficulty = st.sidebar.selectbox(
    "Pedagogical Difficulty Filter:",
    [
        "Any",
        "Beginner",
        "Mixed",
        "Intermediate",
        "Advanced"
    ]
)

search_method = st.sidebar.selectbox(
    "Semantic Search Method:",
    [
        "Automatic",
        "TF-IDF",
        "SBERT"
    ]
)

top_n_slider = st.sidebar.slider(
    "Maximum Recommendations:",
    min_value=3,
    max_value=15,
    value=5
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Live Recommendations",
    "📊 Evaluation & Analytics",
    "🔬 Model Experiments"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:

    if search_query.strip() == "":

        st.warning(
            "Please enter a learning topic."
        )

    else:

        st.subheader(
            f"Analysing Courses for: "
            f"'{search_query}'"
        )

        # ----------------------------------------------------
        # TF-IDF SEARCH
        # ----------------------------------------------------

        query_vector = tfidf.transform(
            [search_query]
        )

        tfidf_similarities = (
            cosine_similarity(
                query_vector,
                tfidf_matrix
            ).flatten()
        )

        # ----------------------------------------------------
        # SBERT SEARCH
        # ----------------------------------------------------

        sbert_similarities = None

        if sbert_model is not None:

            query_embedding = (
                sbert_model.encode(
                    [search_query],
                    normalize_embeddings=True
                )
            )

            if faiss_index is not None:

                distances, indices = (
                    faiss_index.search(
                        query_embedding.astype(
                            "float32"
                        ),
                        len(unique_courses)
                    )
                )

                sbert_similarities = (
                    np.zeros(
                        len(unique_courses)
                    )
                )

                for score, idx in zip(
                    distances[0],
                    indices[0]
                ):

                    if idx >= 0:

                        sbert_similarities[
                            idx
                        ] = score

            else:

                sbert_similarities = (
                    cosine_similarity(
                        query_embedding,
                        sbert_embeddings
                    ).flatten()
                )

        # ----------------------------------------------------
        # SELECT SEARCH MODEL
        # ----------------------------------------------------

        if (
            search_method == "SBERT"
            and sbert_similarities is not None
        ):

            content_similarities = (
                sbert_similarities
            )

            search_label = (
                "SBERT semantic retrieval"
            )

        elif search_method == "SBERT":

            st.warning(
                "SBERT is unavailable. "
                "Falling back to TF-IDF."
            )

            content_similarities = (
                tfidf_similarities
            )

            search_label = (
                "TF-IDF fallback"
            )

        else:

            content_similarities = (
                tfidf_similarities
            )

            search_label = (
                "TF-IDF lexical retrieval"
            )

        # Automatic mode uses SBERT where available
        if search_method == "Automatic":

            if sbert_similarities is not None:

                content_similarities = (
                    sbert_similarities
                )

                search_label = (
                    "SBERT semantic retrieval"
                )

            else:

                content_similarities = (
                    tfidf_similarities
                )

                search_label = (
                    "TF-IDF retrieval"
                )

        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        candidates = unique_courses.copy()

        candidates[
            "content_match_score"
        ] = content_similarities

        # Difficulty filter
        if selected_difficulty != "Any":

            candidates = candidates[
                candidates[
                    "course_difficulty"
                ] == selected_difficulty
            ]

        if candidates.empty:

            st.error(
                "No courses matched the selected "
                "difficulty level."
            )

        elif candidates[
            "content_match_score"
        ].max() == 0:

            st.error(
                "No courses matched your search."
            )

        else:

            # ------------------------------------------------
            # COLLABORATIVE PREDICTION
            # ------------------------------------------------

            hidden_user = "anonymous_learner"

            candidates[
                "predicted_score"
            ] = candidates[
                "course_id"
            ].apply(
                lambda course_id:
                svd_model.predict(
                    hidden_user,
                    course_id
                ).est
            )

            # ------------------------------------------------
            # NORMALISED HYBRID SCORE
            # ------------------------------------------------

            content_weight = (
                float(best_hybrid_weight)
            )

            svd_weight = (
                1.0 - content_weight
            )

            candidates[
                "hybrid_rank_metric"
            ] = (
                candidates[
                    "content_match_score"
                ] * content_weight
                +
                (
                    candidates[
                        "predicted_score"
                    ] / 5.0
                ) * svd_weight
            )

            final_sorted_recs = (
                candidates
                .sort_values(
                    "hybrid_rank_metric",
                    ascending=False
                )
                .head(top_n_slider)
            )

            # ------------------------------------------------
            # KPI
            # ------------------------------------------------

            top_match = (
                final_sorted_recs.iloc[0]
            )

            st.success(
                f"Search engine: {search_label}"
            )

            st.markdown(
                "### 🏆 Top Algorithmic Match"
            )

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = (
                st.columns(4)
            )

            with kpi_col1:

                st.metric(
                    "Top Course",
                    f"{top_match['name'][:25]}..."
                )

            with kpi_col2:

                st.metric(
                    "Provider",
                    top_match[
                        "course_organization"
                    ]
                )

            with kpi_col3:

                st.metric(
                    "Difficulty",
                    top_match[
                        "course_difficulty"
                    ]
                )

            with kpi_col4:

                st.metric(
                    "Adjusted Rating",
                    f"{top_match['adjusted_rating']:.2f} / 5"
                )

            st.write("---")

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.markdown(
                "### 🎯 Curated Best-Fit Learning Path"
            )

            display_columns = [
                "name",
                "course_organization",
                "course_difficulty",
                "course_Certificate_type",
                "course_students_enrolled",
                "content_match_score",
                "predicted_score",
                "hybrid_rank_metric"
            ]

            display_columns = [
                c for c in display_columns
                if c in final_sorted_recs.columns
            ]

            display_df = (
                final_sorted_recs[
                    display_columns
                ].copy()
            )

            rename_map = {
                "name": "Course Name",
                "course_organization": "Offered By",
                "course_difficulty": "Difficulty",
                "course_Certificate_type": "Credential",
                "course_students_enrolled":
                    "Registrations",
                "content_match_score":
                    "Content Similarity",
                "predicted_score":
                    "Predicted Quality",
                "hybrid_rank_metric":
                    "Hybrid Score"
            }

            display_df = display_df.rename(
                columns=rename_map
            )

            if "Content Similarity" in display_df:
                display_df[
                    "Content Similarity"
                ] = display_df[
                    "Content Similarity"
                ].round(3)

            if "Predicted Quality" in display_df:
                display_df[
                    "Predicted Quality"
                ] = display_df[
                    "Predicted Quality"
                ].round(2)

            if "Hybrid Score" in display_df:
                display_df[
                    "Hybrid Score"
                ] = display_df[
                    "Hybrid Score"
                ].round(3)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.write("---")

            # ------------------------------------------------
            # COMPARATOR
            # ------------------------------------------------

            st.markdown(
                "### 📊 Interactive Peer-Course Comparator"
            )

            comparison_selection = st.multiselect(
                "Choose up to 3 courses:",
                options=final_sorted_recs[
                    "name"
                ].tolist(),
                default=final_sorted_recs[
                    "name"
                ].head(2).tolist()
            )

            if len(comparison_selection) > 3:

                st.warning(
                    "Please select no more than 3 courses."
                )

                comparison_selection = (
                    comparison_selection[:3]
                )

            if comparison_selection:

                comp_data = (
                    final_sorted_recs[
                        final_sorted_recs[
                            "name"
                        ].isin(
                            comparison_selection
                        )
                    ]
                )

                comp_table = pd.DataFrame({
                    "Metric Parameter": [
                        "Partner Institution",
                        "Difficulty Level",
                        "Baseline Course Rating",
                        "Sentiment Adjusted Rating",
                        "Review Sentiment",
                        "Content Similarity",
                        "SVD Predicted Score",
                        "Hybrid Score",
                        "Platform Enrollment"
                    ]
                })

                for _, row in comp_data.iterrows():

                    short_name = (
                        f"{row['name'][:30]}..."
                    )

                    comp_table[
                        short_name
                    ] = [
                        row[
                            "course_organization"
                        ],

                        row[
                            "course_difficulty"
                        ],

                        f"{row[rating_column]:.2f} / 5",

                        f"{row['adjusted_rating']:.2f} / 5",

                        f"{row['sentiment_score']:.4f}",

                        f"{row['content_match_score']:.3f}",

                        f"{row['predicted_score']:.2f} / 5",

                        f"{row['hybrid_rank_metric']:.3f}",

                        row[
                            "course_students_enrolled"
                        ]
                        if "course_students_enrolled"
                        in row
                        else "N/A"
                    ]

                st.table(
                    comp_table
                )


# ============================================================
# TAB 2: EVALUATION
# ============================================================

with tab2:

    st.subheader(
        "📊 System Evaluation & Validation"
    )

    st.markdown(
        """
        The evaluation uses a **user-aware 80/20 holdout**.
        Held-out interactions are not used during model training.
        Rating prediction is evaluated with RMSE and MAE, while
        recommendation quality is assessed using Precision@5,
        Recall@5, NDCG@5 and Hit Rate@5.
        """
    )

    # --------------------------------------------------------
    # DATASET SPLIT
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Interactions",
            f"{len(master_df):,}"
        )

    with col2:
        st.metric(
            "Training Interactions",
            f"{len(train_df):,}"
        )

    with col3:
        st.metric(
            "Held-Out Interactions",
            f"{len(test_df):,}"
        )

    with col4:
        st.metric(
            "Unique Courses",
            f"{unique_courses['course_id'].nunique():,}"
        )

    st.write("---")

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    st.markdown(
        "### 🧮 Core Performance Metrics"
    )

    metrics_display = pd.DataFrame([
        {
            "Metric":
                "SVD RMSE",
            "Value":
                live_metrics["SVD Raw RMSE"]
        },
        {
            "Metric":
                "SVD MAE",
            "Value":
                live_metrics["SVD Raw MAE"]
        },
        {
            "Metric":
                "Sentiment-SVD RMSE",
            "Value":
                live_metrics[
                    "Sentiment-SVD RMSE"
                ]
        },
        {
            "Metric":
                "Sentiment-SVD MAE",
            "Value":
                live_metrics[
                    "Sentiment-SVD MAE"
                ]
        },
        {
            "Metric":
                "Precision@5",
            "Value":
                live_metrics[
                    "Precision@5"
                ]
        },
        {
            "Metric":
                "Recall@5",
            "Value":
                live_metrics[
                    "Recall@5"
                ]
        },
        {
            "Metric":
                "NDCG@5",
            "Value":
                live_metrics[
                    "NDCG@5"
                ]
        },
        {
            "Metric":
                "Hit Rate@5",
            "Value":
                live_metrics[
                    "Hit Rate@5"
                ]
        }
    ])

    st.dataframe(
        metrics_display,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    # --------------------------------------------------------
    # BASELINE COMPARISON
    # --------------------------------------------------------

    st.markdown(
        "### 🥇 Baseline Model Comparison"
    )

    baseline_plot_df = (
        baseline_df
        .set_index("Model")
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    baseline_plot_df[
        ["RMSE", "MAE"]
    ].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Rating Prediction Error Across Models"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_xlabel(
        "Model"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    st.dataframe(
        baseline_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    # --------------------------------------------------------
    # SENTIMENT ABLATION
    # --------------------------------------------------------

    st.markdown(
        "### 🧪 Sentiment Coefficient Ablation Study"
    )

    st.markdown(
        """
        Rather than assuming that `0.15` is optimal, the system
        evaluates multiple sentiment coefficients. The selected
        coefficient is the one producing the lowest held-out RMSE.
        """
    )

    st.dataframe(
        sentiment_ablation_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        sentiment_ablation_df[
            "Sentiment Coefficient"
        ],
        sentiment_ablation_df[
            "RMSE"
        ],
        marker="o"
    )

    ax.set_xlabel(
        "Sentiment Coefficient"
    )

    ax.set_ylabel(
        "Held-Out RMSE"
    )

    ax.set_title(
        "Effect of Sentiment Calibration Strength"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    st.info(
        f"Best tested sentiment coefficient: "
        f"{best_sentiment_coefficient:.2f}"
    )

    st.write("---")

    # --------------------------------------------------------
    # HYBRID WEIGHT EXPERIMENT
    # --------------------------------------------------------

    st.markdown(
        "### ⚖️ Hybrid Weight Experiment"
    )

    st.dataframe(
        hybrid_weights_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        hybrid_weights_df[
            "Content Weight"
        ],
        hybrid_weights_df[
            "Hybrid RMSE"
        ],
        marker="o"
    )

    ax.set_xlabel(
        "Content Weight"
    )

    ax.set_ylabel(
        "Hybrid RMSE"
    )

    ax.set_title(
        "Sensitivity of Hybrid Model to Component Weight"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    st.info(
        f"Selected content weight: "
        f"{best_hybrid_weight:.2f} | "
        f"Selected SVD weight: "
        f"{1 - best_hybrid_weight:.2f}"
    )

    st.write("---")

    # --------------------------------------------------------
    # SEARCH PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        "### ⚡ Computational Performance"
    )

    latency_rows = [
        {
            "Search Component":
                "TF-IDF",
            "Latency (ms)":
                live_metrics[
                    "TF-IDF Search Latency (ms)"
                ]
        }
    ]

    if (
        live_metrics[
            "SBERT Search Latency (ms)"
        ]
        is not None
    ):

        latency_rows.append({
            "Search Component":
                "SBERT",
            "Latency (ms)":
                live_metrics[
                    "SBERT Search Latency (ms)"
                ]
        })

    if (
        live_metrics[
            "FAISS Search Latency (ms)"
        ]
        is not None
    ):

        latency_rows.append({
            "Search Component":
                "SBERT + FAISS",
            "Latency (ms)":
                live_metrics[
                    "FAISS Search Latency (ms)"
                ]
        })

    latency_df = pd.DataFrame(
        latency_rows
    )

    st.dataframe(
        latency_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        latency_df[
            "Search Component"
        ],
        latency_df[
            "Latency (ms)"
        ]
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Semantic Search Computational Benchmark"
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    # --------------------------------------------------------
    # DATA LEAKAGE NOTE
    # --------------------------------------------------------

    st.write("---")

    st.markdown(
        "### 🔐 Evaluation Integrity & Leakage Controls"
    )

    st.success(
        """
        **Leakage controls implemented:**

        1. The dataset is split before model evaluation.
        2. Test interactions are excluded from SVD training.
        3. Ranking metrics use held-out interactions only.
        4. NDCG is calculated without sorting the ground-truth
           ratings according to predicted values first.
        5. Rating errors are measured against the original
           held-out ratings.
        6. Sentiment-adjusted ratings are constructed from the
           training interactions for SVD evaluation.
        """
    )


# ============================================================
# TAB 3: MODEL EXPERIMENTS
# ============================================================

with tab3:

    st.subheader(
        "🔬 Advanced Model Experiments"
    )

    # --------------------------------------------------------
    # SBERT STATUS
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Semantic Representation"
    )

    if sbert_model is not None:

        st.success(
            """
            SBERT is active using
            `all-MiniLM-L6-v2`.

            Course titles are transformed into dense semantic
            embeddings, allowing the system to identify
            conceptually related courses even when exact keywords
            differ.
            """
        )

    else:

        st.warning(
            """
            SBERT is not installed or could not be loaded.

            The application is currently using TF-IDF as the
            semantic-search fallback.
            """
        )

    # --------------------------------------------------------
    # FAISS STATUS
    # --------------------------------------------------------

    st.markdown(
        "### 🚀 Vector Search Acceleration"
    )

    if faiss_index is not None:

        st.success(
            """
            FAISS is active.

            SBERT embeddings are indexed using an inner-product
            vector index for efficient similarity retrieval.
            """
        )

    else:

        st.info(
            """
            FAISS is not active. SBERT retrieval uses direct
            cosine similarity instead.
            """
        )

    # --------------------------------------------------------
    # MODEL ARCHITECTURE
    # --------------------------------------------------------

    st.markdown(
        "### 🏗️ Current System Architecture"
    )

    st.code(
        """
Coursera Dataset
       │
       ▼
Data Cleaning + EDA
       │
       ├───────────────┐
       ▼               ▼
 Review Text      Course Metadata
       │               │
       ▼               ▼
 VADER Sentiment   TF-IDF
       │               │
       │               └──────► Lexical Retrieval
       │
       └───────────────┐
                       ▼
                 Adjusted Ratings
                       │
                       ▼
                    SVD CF
                       │
                       ├──────────────┐
                       │              │
                       ▼              ▼
                 Personalisation   SBERT
                                      │
                                      ▼
                                    FAISS
                                      │
                                      ▼
                             Semantic Retrieval
                                      │
                    ┌─────────────────┘
                    ▼
              Hybrid Ranking
                    │
                    ▼
          Difficulty Constraint
                    │
                    ▼
        Personalised Recommendations
                    │
                    ▼
              Streamlit UI
        """,
        language="text"
    )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Model Configuration"
    )

    configuration_df = pd.DataFrame([
        {
            "Parameter":
                "SVD Factors",
            "Value":
                SVD_PARAMS["n_factors"]
        },
        {
            "Parameter":
                "SVD Learning Rate",
            "Value":
                SVD_PARAMS["lr_all"]
        },
        {
            "Parameter":
                "SVD Regularisation",
            "Value":
                SVD_PARAMS["reg_all"]
        },
        {
            "Parameter":
                "Test Size",
            "Value":
                "20%"
        },
        {
            "Parameter":
                "TF-IDF Features",
            "Value":
                "5000"
        },
        {
            "Parameter":
                "TF-IDF N-grams",
            "Value":
                "1–2"
        },
        {
            "Parameter":
                "SBERT Model",
            "Value":
                "all-MiniLM-L6-v2"
                if sbert_model is not None
                else "Unavailable"
        },
        {
            "Parameter":
                "FAISS",
            "Value":
                "Enabled"
                if faiss_index is not None
                else "Disabled"
        }
    ])

    st.dataframe(
        configuration_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # EXISTING REPORT IMAGES
    # --------------------------------------------------------

    st.write("---")

    image_folder = (
        "report_images_extended"
    )

    if os.path.exists(
        image_folder
    ):

        st.markdown(
            "### 📈 Existing Evaluation Visualisations"
        )

        col1, col2 = st.columns(2)

        with col1:

            if os.path.exists(
                f"{image_folder}/Fig_4_01_Difficulty.png"
            ):
                st.image(
                    f"{image_folder}/Fig_4_01_Difficulty.png",
                    caption="Corpus Difficulty Breakdown",
                    use_container_width=True
                )

            if os.path.exists(
                f"{image_folder}/Fig_4_08_SentimentGap.png"
            ):
                st.image(
                    f"{image_folder}/Fig_4_08_SentimentGap.png",
                    caption="Sentiment-Rating Gap",
                    use_container_width=True
                )

        with col2:

            if os.path.exists(
                f"{image_folder}/Fig_4_02_Institutions.png"
            ):
                st.image(
                    f"{image_folder}/Fig_4_02_Institutions.png",
                    caption="Institutional Distribution",
                    use_container_width=True
                )

            if os.path.exists(
                f"{image_folder}/Fig_4_10_RatingShift.png"
            ):
                st.image(
                    f"{image_folder}/Fig_4_10_RatingShift.png",
                    caption="Rating Calibration",
                    use_container_width=True
                )

        st.write("---")

        st.markdown(
            "### 🔬 Machine Learning Evaluation"
        )

        col3, col4 = st.columns(2)

        with col3:

            if os.path.exists(
                f"{image_folder}/Fig_5_01_Accuracy.png"
            ):
                st.image(
                    f"{image_folder}/Fig_5_01_Accuracy.png",
                    caption="RMSE / MAE Comparison",
                    use_container_width=True
                )

            if os.path.exists(
                f"{image_folder}/Fig_5_04_NDCG.png"
            ):
                st.image(
                    f"{image_folder}/Fig_5_04_NDCG.png",
                    caption="NDCG Evaluation",
                    use_container_width=True
                )

        with col4:

            if os.path.exists(
                f"{image_folder}/Fig_5_02_ErrorDist.png"
            ):
                st.image(
                    f"{image_folder}/Fig_5_02_ErrorDist.png",
                    caption="Prediction Error Distribution",
                    use_container_width=True
                )

            if os.path.exists(
                f"{image_folder}/Fig_5_08_Latency.png"
            ):
                st.image(
                    f"{image_folder}/Fig_5_08_Latency.png",
                    caption="Latency Analysis",
                    use_container_width=True
                )

        st.write("---")

        with st.expander(
            "📂 Complete Technical Visual Appendix"
        ):

            all_images = sorted([
                f
                for f in os.listdir(
                    image_folder
                )
                if f.endswith(".png")
            ])

            for img in all_images:

                st.image(
                    f"{image_folder}/{img}",
                    caption=(
                        f"System Verification Artifact: "
                        f"{img}"
                    ),
                    use_container_width=True
                )

    else:

        st.info(
            "The report image directory is not available."
        )
