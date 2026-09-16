import os
import time
from pathlib import Path
from collections import defaultdict

import faiss
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from surprise import Dataset, Reader, SVD
from surprise.model_selection import KFold


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coursera Search & Recommendation Dashboard",
    page_icon="🎓",
    layout="wide",
)

DATASET_CANDIDATES = (
    "coursera_mini_master.csv",
    "coursera_mini_master(1).csv",
)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SENTIMENT_WEIGHT = 0.15  # Heuristic calibration parameter; report it as such.
MIN_PROFILE_INTERACTIONS = 5
PROFILE_COUNT = 5
SVD_FACTORS = 50
SVD_LR = 0.005
SVD_REG = 0.02
RANDOM_STATE = 42
CV_FOLDS = 5
DEFAULT_TOP_N = 5
MIN_RETRIEVAL_POOL = 100
RETRIEVAL_MULTIPLIER = 20

BENCHMARK_QUERIES = [
    "Data Science",
    "Python",
    "Machine Learning",
    "Business",
    "History",
    "Artificial Intelligence",
]


# ============================================================
# SMALL UTILITIES
# ============================================================

def resolve_dataset_path() -> Path:
    """Find the project CSV in the app directory."""
    for candidate in DATASET_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the Coursera dataset. Expected one of: "
        + ", ".join(DATASET_CANDIDATES)
    )


def clean_enrollment(value) -> float:
    """Convert values such as '57k' and '1.2m' to numeric counts."""
    if pd.isna(value):
        return 0.0

    text = str(value).strip().lower().replace(",", "")
    if not text:
        return 0.0

    try:
        if text.endswith("m"):
            return float(text[:-1]) * 1_000_000
        if text.endswith("k"):
            return float(text[:-1]) * 1_000
        return float(text)
    except ValueError:
        return 0.0


def minmax_scale(series: pd.Series, constant_value: float = 0.5) -> pd.Series:
    """
    Scale a numeric series to [0, 1].
    If all values are identical, return a neutral constant so that one signal
    does not arbitrarily dominate the other.
    """
    values = pd.to_numeric(series, errors="coerce")
    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)

    if np.isclose(maximum, minimum):
        return pd.Series(
            np.full(len(values), constant_value, dtype=float),
            index=values.index,
        )

    return (values - minimum) / (maximum - minimum)


def is_known_to_trainset(trainset, raw_uid, raw_iid) -> bool:
    """Check whether a raw user and item were both seen in a Surprise trainset."""
    try:
        trainset.to_inner_uid(raw_uid)
        trainset.to_inner_iid(raw_iid)
        return True
    except ValueError:
        return False


# ============================================================
# OFFLINE EVALUATION
# ============================================================

def evaluate_svd_out_of_fold(data_for_svd: Dataset) -> dict:
    """
    Evaluate SVD using 5-fold out-of-fold predictions.

    Important:
    - RMSE/MAE are collaborative-filtering metrics only.
    - NDCG@5 ranks each user's OOF interacted items and is also SVD-only.
    - These metrics are intentionally NOT labelled as hybrid-system metrics.
    """
    splitter = KFold(
        n_splits=CV_FOLDS,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    fold_rmse = []
    fold_mae = []
    all_predictions = []
    all_warm_flags = []

    for trainset, testset in splitter.split(data_for_svd):
        model = SVD(
            n_factors=SVD_FACTORS,
            lr_all=SVD_LR,
            reg_all=SVD_REG,
            random_state=RANDOM_STATE,
        )
        model.fit(trainset)
        predictions = model.test(testset)

        true_values = np.array([p.r_ui for p in predictions], dtype=float)
        estimated_values = np.array([p.est for p in predictions], dtype=float)
        errors = true_values - estimated_values

        fold_rmse.append(float(np.sqrt(np.mean(errors ** 2))))
        fold_mae.append(float(np.mean(np.abs(errors))))

        for prediction in predictions:
            all_predictions.append(prediction)
            all_warm_flags.append(
                is_known_to_trainset(
                    trainset,
                    prediction.uid,
                    prediction.iid,
                )
            )

    all_true = np.array([p.r_ui for p in all_predictions], dtype=float)
    all_est = np.array([p.est for p in all_predictions], dtype=float)
    all_errors = all_true - all_est

    warm_mask = np.array(all_warm_flags, dtype=bool)
    if warm_mask.any():
        warm_errors = all_errors[warm_mask]
        warm_rmse = float(np.sqrt(np.mean(warm_errors ** 2)))
        warm_mae = float(np.mean(np.abs(warm_errors)))
    else:
        warm_rmse = np.nan
        warm_mae = np.nan

    # OOF NDCG@5:
    # relevance is binary: adjusted rating >= 4.0 is "liked".
    # We only evaluate users who have at least two OOF interactions and both
    # relevant and non-relevant items, because otherwise NDCG is not informative.
    grouped_predictions = defaultdict(list)
    for prediction in all_predictions:
        grouped_predictions[prediction.uid].append(
            (float(prediction.r_ui), float(prediction.est))
        )

    ndcg_scores = []
    for user_ratings in grouped_predictions.values():
        if len(user_ratings) < 2:
            continue

        binary_true = [
            1 if true_rating >= 4.0 else 0
            for true_rating, _ in user_ratings
        ]
        if len(set(binary_true)) < 2:
            continue

        predicted_scores = [
            predicted_rating
            for _, predicted_rating in user_ratings
        ]

        ndcg_scores.append(
            float(
                ndcg_score(
                    [binary_true],
                    [predicted_scores],
                    k=min(5, len(binary_true)),
                )
            )
        )

    return {
        "SVD 5-Fold CV RMSE": round(float(np.mean(fold_rmse)), 4),
        "SVD RMSE Std. Dev.": round(float(np.std(fold_rmse)), 4),
        "SVD 5-Fold CV MAE": round(float(np.mean(fold_mae)), 4),
        "SVD MAE Std. Dev.": round(float(np.std(fold_mae)), 4),
        "SVD OOF NDCG@5": (
            round(float(np.mean(ndcg_scores)), 4)
            if ndcg_scores
            else np.nan
        ),
        "NDCG Users Evaluated": int(len(ndcg_scores)),
        "Warm-Start Prediction Coverage (%)": round(
            float(warm_mask.mean() * 100), 2
        ),
        "Warm-Start RMSE": (
            round(warm_rmse, 4) if not np.isnan(warm_rmse) else np.nan
        ),
        "Warm-Start MAE": (
            round(warm_mae, 4) if not np.isnan(warm_mae) else np.nan
        ),
        "OOF Predictions": int(len(all_predictions)),
    }


def build_sentiment_diagnostics(master_df: pd.DataFrame) -> dict:
    """
    Compare VADER polarity with star-rating polarity as a proxy diagnostic.
    This is not presented as ground-truth sentiment accuracy.
    """
    subset = master_df[
        master_df["reviews"].str.strip().ne("")
        & (
            (master_df["rating"] <= 2)
            | (master_df["rating"] >= 4)
        )
    ].copy()

    if subset.empty:
        return {
            "Extreme-Rating Reviews": 0,
            "VADER Non-Neutral Coverage (%)": np.nan,
            "VADER/Star Polarity Agreement (%)": np.nan,
        }

    subset["star_positive"] = subset["rating"] >= 4
    subset["vader_class"] = np.where(
        subset["sentiment_score"] >= 0.05,
        1,
        np.where(subset["sentiment_score"] <= -0.05, 0, -1),
    )

    non_neutral = subset[subset["vader_class"] != -1].copy()

    coverage = len(non_neutral) / len(subset) * 100

    if non_neutral.empty:
        agreement = np.nan
    else:
        agreement = (
            (
                non_neutral["vader_class"].astype(bool)
                == non_neutral["star_positive"]
            ).mean()
            * 100
        )

    return {
        "Extreme-Rating Reviews": int(len(subset)),
        "VADER Non-Neutral Coverage (%)": round(float(coverage), 2),
        "VADER/Star Polarity Agreement (%)": (
            round(float(agreement), 2)
            if not np.isnan(agreement)
            else np.nan
        ),
    }


# ============================================================
# BACKEND LOADING / TRAINING
# ============================================================

@st.cache_resource(
    show_spinner="Loading data, evaluating SVD, and building the SBERT + FAISS index..."
)
def load_and_train_system():
    dataset_path = resolve_dataset_path()
    master_df = pd.read_csv(dataset_path)

    required_columns = {
        "reviews",
        "reviewers",
        "rating",
        "course_id",
        "name",
        "course_url",
        "course_organization",
        "course_Certificate_type",
        "course_rating",
        "course_difficulty",
        "course_students_enrolled",
    }
    missing_columns = required_columns - set(master_df.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # --------------------------------------------------------
    # 1. Basic cleaning
    # --------------------------------------------------------
    master_df = master_df.copy()
    master_df["reviews"] = (
        master_df["reviews"]
        .fillna("")
        .astype(str)
    )
    master_df["reviewers"] = (
        master_df["reviewers"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    master_df["course_id"] = (
        master_df["course_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    master_df["rating"] = pd.to_numeric(
        master_df["rating"],
        errors="coerce",
    )
    master_df["course_rating"] = pd.to_numeric(
        master_df["course_rating"],
        errors="coerce",
    )
    master_df["name"] = master_df["name"].fillna("").astype(str)

    master_df = master_df[
        master_df["course_id"].ne("")
        & master_df["rating"].between(1, 5, inclusive="both")
    ].copy()

    # --------------------------------------------------------
    # 2. Review-level VADER sentiment and adjusted rating
    # --------------------------------------------------------
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sentiment_analyser = SentimentIntensityAnalyzer()

    master_df["sentiment_score"] = master_df["reviews"].apply(
        lambda text: sentiment_analyser.polarity_scores(text)["compound"]
    )

    master_df["adjusted_rating"] = (
        master_df["rating"]
        + SENTIMENT_WEIGHT * master_df["sentiment_score"]
    ).clip(1.0, 5.0)

    master_df["has_text_review"] = master_df["reviews"].str.strip().ne("")

    # --------------------------------------------------------
    # 3. Pseudonymised demonstration learner profiles
    # --------------------------------------------------------
    valid_profile_rows = master_df[
        master_df["reviewers"].ne("")
        & ~master_df["reviewers"].str.contains(
            r"deleted|anonymous|unknown|guest|null|none",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    reviewer_stats = (
        valid_profile_rows
        .groupby("reviewers")
        .agg(
            interaction_count=("course_id", "size"),
            unique_course_count=("course_id", "nunique"),
            rating_std=("rating", "std"),
        )
        .fillna({"rating_std": 0.0})
    )

    eligible_profiles = reviewer_stats[
        (reviewer_stats["unique_course_count"] >= MIN_PROFILE_INTERACTIONS)
        & (reviewer_stats["rating_std"] > 0)
    ].copy()

    eligible_profiles = eligible_profiles.sort_values(
        ["interaction_count", "rating_std", "unique_course_count"],
        ascending=[False, False, False],
    )

    selected_real_reviewers = (
        eligible_profiles
        .head(PROFILE_COUNT)
        .index
        .tolist()
    )

    professional_ids = [
        f"Learner_{index + 1:03d}"
        for index in range(len(selected_real_reviewers))
    ]

    persona_mapping = dict(
        zip(selected_real_reviewers, professional_ids)
    )

    # Keep original reviewer text untouched and create a separate model ID.
    master_df["reviewer_model_id"] = master_df["reviewers"].replace(
        persona_mapping
    )

    # Missing reviewer labels are made row-unique so they do not become
    # a fake shared collaborative-filtering user.
    missing_reviewer_mask = master_df["reviewer_model_id"].eq("")
    if missing_reviewer_mask.any():
        master_df.loc[
            missing_reviewer_mask,
            "reviewer_model_id",
        ] = [
            f"MissingReviewer_{idx}"
            for idx in master_df.index[missing_reviewer_mask]
        ]

    # --------------------------------------------------------
    # 4. Collaborative SVD: honest OOF evaluation + final model
    # --------------------------------------------------------
    collaborative_df = master_df[
        ["reviewer_model_id", "course_id", "adjusted_rating"]
    ].dropna()

    reader = Reader(rating_scale=(1.0, 5.0))
    data_for_svd = Dataset.load_from_df(
        collaborative_df,
        reader,
    )

    svd_metrics = evaluate_svd_out_of_fold(data_for_svd)

    full_trainset = data_for_svd.build_full_trainset()
    final_svd_model = SVD(
        n_factors=SVD_FACTORS,
        lr_all=SVD_LR,
        reg_all=SVD_REG,
        random_state=RANDOM_STATE,
    )
    final_svd_model.fit(full_trainset)

    # --------------------------------------------------------
    # 5. Course-level aggregation
    # --------------------------------------------------------
    # This fixes the previous issue where one arbitrary review row supplied
    # the course-level sentiment and adjusted rating displayed in the UI.
    unique_courses = (
        master_df
        .groupby("course_id", as_index=False)
        .agg(
            name=("name", "first"),
            course_url=("course_url", "first"),
            course_organization=("course_organization", "first"),
            course_Certificate_type=(
                "course_Certificate_type",
                "first",
            ),
            course_rating=("course_rating", "first"),
            course_difficulty=("course_difficulty", "first"),
            course_students_enrolled=(
                "course_students_enrolled",
                "first",
            ),
            avg_review_rating=("rating", "mean"),
            mean_sentiment=("sentiment_score", "mean"),
            mean_adjusted_rating=("adjusted_rating", "mean"),
            review_count=("has_text_review", "sum"),
        )
        .reset_index(drop=True)
    )

    unique_courses["enrolled_numeric"] = (
        unique_courses["course_students_enrolled"]
        .apply(clean_enrollment)
    )

    # Use available course metadata to give SBERT more context than title-only
    # retrieval. This is still bounded by what the uploaded dataset contains.
    unique_courses["semantic_text"] = (
        unique_courses["name"].fillna("")
        + ". Offered by "
        + unique_courses["course_organization"].fillna("")
        + ". Difficulty: "
        + unique_courses["course_difficulty"].fillna("")
        + ". Credential: "
        + unique_courses["course_Certificate_type"].fillna("")
    )

    # --------------------------------------------------------
    # 6. SBERT + FAISS semantic retrieval
    # --------------------------------------------------------
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    course_embeddings = sbert_model.encode(
        unique_courses["semantic_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    course_embeddings = np.ascontiguousarray(
        course_embeddings,
        dtype=np.float32,
    )

    embedding_dimension = course_embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    faiss_index.add(course_embeddings)

    sentiment_metrics = build_sentiment_diagnostics(master_df)

    return {
        "master_df": master_df,
        "unique_courses": unique_courses,
        "svd_model": final_svd_model,
        "sbert_model": sbert_model,
        "faiss_index": faiss_index,
        "svd_metrics": svd_metrics,
        "sentiment_metrics": sentiment_metrics,
        "professional_ids": professional_ids,
        "dataset_path": str(dataset_path),
        "embedding_dimension": int(embedding_dimension),
    }


# ============================================================
# RECOMMENDATION PIPELINE
# ============================================================

def retrieve_semantic_candidates(
    query: str,
    unique_courses: pd.DataFrame,
    sbert_model: SentenceTransformer,
    faiss_index,
    top_n: int,
    selected_difficulty: str,
    completed_course_ids: set,
) -> pd.DataFrame:
    """
    Retrieve a bounded semantic candidate set with SBERT + FAISS.

    If filtering removes too many items, automatically expand the search to the
    full index and then reapply the same filters.
    """
    query_vector = sbert_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32,
    )

    total_courses = len(unique_courses)
    initial_pool_size = min(
        max(
            top_n * RETRIEVAL_MULTIPLIER,
            MIN_RETRIEVAL_POOL,
        ),
        total_courses,
    )

    def make_candidate_frame(k: int) -> pd.DataFrame:
        similarities, indices = faiss_index.search(
            query_vector,
            int(k),
        )

        valid_pairs = [
            (int(index), float(similarity))
            for index, similarity in zip(
                indices[0],
                similarities[0],
            )
            if index >= 0
        ]

        candidate_indices = [
            pair[0]
            for pair in valid_pairs
        ]
        candidate_scores = [
            pair[1]
            for pair in valid_pairs
        ]

        candidates = unique_courses.iloc[
            candidate_indices
        ].copy()

        candidates["content_match_score"] = candidate_scores

        if selected_difficulty != "Any":
            candidates = candidates[
                candidates["course_difficulty"]
                == selected_difficulty
            ].copy()

        if completed_course_ids:
            candidates = candidates[
                ~candidates["course_id"].isin(
                    completed_course_ids
                )
            ].copy()

        return candidates

    candidates = make_candidate_frame(initial_pool_size)

    # Fallback: expand to the whole index only if filtering/history leaves too
    # few candidates for the requested output size.
    if len(candidates) < top_n and initial_pool_size < total_courses:
        candidates = make_candidate_frame(total_courses)

    return candidates


def recommend_courses(
    query: str,
    selected_user: str,
    selected_difficulty: str,
    personalization_weight: float,
    top_n: int,
    backend: dict,
):
    """Run the complete recommendation pipeline and measure end-to-end latency."""
    start_time = time.perf_counter()

    master_df = backend["master_df"]
    unique_courses = backend["unique_courses"]
    svd_model = backend["svd_model"]
    sbert_model = backend["sbert_model"]
    faiss_index = backend["faiss_index"]

    anonymous = selected_user == "Anonymous / Cold Start Learner"

    if anonymous:
        completed_course_ids = set()
        effective_weight = 0.0
    else:
        completed_course_ids = set(
            master_df.loc[
                master_df["reviewer_model_id"] == selected_user,
                "course_id",
            ].tolist()
        )
        effective_weight = float(personalization_weight)

    candidates = retrieve_semantic_candidates(
        query=query,
        unique_courses=unique_courses,
        sbert_model=sbert_model,
        faiss_index=faiss_index,
        top_n=top_n,
        selected_difficulty=selected_difficulty,
        completed_course_ids=completed_course_ids,
    )

    if candidates.empty:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000
        return candidates, latency_ms, {
            "candidate_count": 0,
            "effective_weight": effective_weight,
            "completed_courses_removed": len(completed_course_ids),
        }

    # Semantic relevance is always available.
    candidates["semantic_norm"] = minmax_scale(
        candidates["content_match_score"]
    )

    # Only known demonstration learners receive a collaborative-personalisation
    # score. Anonymous mode remains genuinely content-based.
    if anonymous:
        candidates["predicted_score"] = np.nan
        candidates["svd_norm"] = 0.0
        candidates["hybrid_rank_metric"] = candidates[
            "semantic_norm"
        ]
    else:
        candidates["predicted_score"] = candidates[
            "course_id"
        ].apply(
            lambda course_id: svd_model.predict(
                selected_user,
                course_id,
            ).est
        )

        candidates["svd_norm"] = minmax_scale(
            candidates["predicted_score"]
        )

        candidates["hybrid_rank_metric"] = (
            candidates["semantic_norm"]
            * (1.0 - effective_weight)
            + candidates["svd_norm"]
            * effective_weight
        )

    candidates = candidates.sort_values(
        [
            "hybrid_rank_metric",
            "content_match_score",
            "mean_adjusted_rating",
        ],
        ascending=[False, False, False],
    )

    final_recommendations = candidates.head(top_n).copy()

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    diagnostics = {
        "candidate_count": int(len(candidates)),
        "effective_weight": effective_weight,
        "completed_courses_removed": int(
            len(completed_course_ids)
        ),
    }

    return final_recommendations, latency_ms, diagnostics


def run_latency_benchmark(
    backend: dict,
    selected_user: str,
    selected_difficulty: str,
    personalization_weight: float,
    top_n: int,
    repeats: int = 3,
) -> pd.DataFrame:
    """Benchmark the complete warm-start recommendation pipeline."""
    rows = []

    for query in BENCHMARK_QUERIES:
        for repetition in range(1, repeats + 1):
            _, latency_ms, _ = recommend_courses(
                query=query,
                selected_user=selected_user,
                selected_difficulty=selected_difficulty,
                personalization_weight=personalization_weight,
                top_n=top_n,
                backend=backend,
            )

            rows.append(
                {
                    "Query": query,
                    "Repetition": repetition,
                    "Latency (ms)": latency_ms,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# INITIALISE BACKEND
# ============================================================

try:
    backend = load_and_train_system()
except Exception as exc:
    st.error("The recommendation system could not be initialised.")
    st.exception(exc)
    st.stop()

master_df = backend["master_df"]
unique_courses = backend["unique_courses"]
professional_ids = backend["professional_ids"]


# ============================================================
# FRONTEND
# ============================================================

st.title("🎓 Smart Coursera Discovery & Analytics Platform")

st.markdown(
    "This application combines **SBERT semantic retrieval**, "
    "**FAISS vector indexing**, **SVD collaborative filtering**, and "
    "**VADER-derived review signals**. Collaborative personalisation is "
    "used only for pseudonymised learners with historical interactions."
)

st.sidebar.header("🔍 Course Discovery Controls")

user_options = [
    "Anonymous / Cold Start Learner"
] + professional_ids

selected_user = st.sidebar.selectbox(
    "Demonstration learner profile:",
    user_options,
    help=(
        "Learner profiles are pseudonymised historical users selected only "
        "when they have enough interactions and rating variation for a "
        "meaningful collaborative-filtering demonstration."
    ),
)

# ------------------------------------------------------------
# Profile panel
# ------------------------------------------------------------

if selected_user != "Anonymous / Cold Start Learner":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧑‍🎓 Active Learner Profile")
    st.sidebar.markdown(
        f"**Pseudonymised ID:** `{selected_user}`"
    )

    user_history = (
        master_df[
            master_df["reviewer_model_id"]
            == selected_user
        ][
            ["name", "rating"]
        ]
        .drop_duplicates()
        .sort_values(
            "rating",
            ascending=False,
        )
    )

    st.sidebar.markdown(
        f"**Historical courses ({len(user_history)}):**"
    )

    for _, row in user_history.head(5).iterrows():
        st.sidebar.caption(
            f"• {row['name']} — {row['rating']:.0f}/5"
        )

    if len(user_history) > 5:
        st.sidebar.caption("• …")

    st.sidebar.caption(
        "Previously completed courses are excluded from new recommendations."
    )
    st.sidebar.markdown("---")
else:
    st.sidebar.info(
        "Cold-start mode uses semantic retrieval only. "
        "Collaborative personalisation is disabled because no learner history exists."
    )

search_query = st.sidebar.text_input(
    "What topic do you want to learn today?",
    value="Data Science",
    help=(
        "Examples: Python, Machine Learning, History, Business, "
        "Artificial Intelligence"
    ),
)

selected_difficulty = st.sidebar.selectbox(
    "Pedagogical difficulty:",
    [
        "Any",
        "Beginner",
        "Mixed",
        "Intermediate",
        "Advanced",
    ],
)

if selected_user == "Anonymous / Cold Start Learner":
    personalization_weight = st.sidebar.slider(
        "Collaborative personalisation weight:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        disabled=True,
        help=(
            "Disabled in cold-start mode because there is no historical "
            "learner profile."
        ),
    )
else:
    personalization_weight = st.sidebar.slider(
        "Collaborative personalisation weight:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help=(
            "0.0 = semantic relevance only; "
            "1.0 = collaborative re-ranking only within the semantically "
            "retrieved candidate pool."
        ),
    )

top_n_slider = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=15,
    value=DEFAULT_TOP_N,
)

tab1, tab2 = st.tabs(
    [
        "🎯 Live Search & Recommendations",
        "📊 Evaluation & System Diagnostics",
    ]
)


# ============================================================
# TAB 1 — LIVE RECOMMENDATIONS
# ============================================================

with tab1:
    if not search_query.strip():
        st.warning(
            "Enter a learning topic in the sidebar to generate recommendations."
        )
    else:
        st.subheader(
            f"Recommendations for: “{search_query.strip()}”"
        )

        final_recommendations, query_latency_ms, diagnostics = (
            recommend_courses(
                query=search_query.strip(),
                selected_user=selected_user,
                selected_difficulty=selected_difficulty,
                personalization_weight=personalization_weight,
                top_n=top_n_slider,
                backend=backend,
            )
        )

        if final_recommendations.empty:
            st.error(
                "No eligible courses remained after applying the selected "
                "difficulty and learner-history filters."
            )
        else:
            top_match = final_recommendations.iloc[0]

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric(
                    "Top course",
                    str(top_match["name"])[:35],
                )

            with kpi2:
                st.metric(
                    "Institution",
                    str(
                        top_match["course_organization"]
                    )[:35],
                )

            with kpi3:
                st.metric(
                    "Difficulty",
                    top_match["course_difficulty"],
                )

            with kpi4:
                st.metric(
                    "Mean sentiment-adjusted rating",
                    f"{top_match['mean_adjusted_rating']:.2f} / 5",
                )

            st.caption(
                f"End-to-end warm query latency: {query_latency_ms:.2f} ms · "
                f"Candidate courses after filtering: "
                f"{diagnostics['candidate_count']} · "
                f"Effective collaborative weight: "
                f"{diagnostics['effective_weight']:.1f}"
            )

            st.markdown("### 🎯 Ranked recommendations")

            display_columns = [
                "name",
                "course_organization",
                "course_difficulty",
                "course_Certificate_type",
                "course_students_enrolled",
                "content_match_score",
                "mean_adjusted_rating",
                "predicted_score",
                "hybrid_rank_metric",
            ]

            display_df = final_recommendations[
                display_columns
            ].copy()

            display_df["content_match_score"] = (
                display_df["content_match_score"]
                .map(lambda value: f"{value:.3f}")
            )
            display_df["mean_adjusted_rating"] = (
                display_df["mean_adjusted_rating"]
                .map(lambda value: f"{value:.2f} ★")
            )

            if selected_user == "Anonymous / Cold Start Learner":
                display_df["predicted_score"] = "N/A"
            else:
                display_df["predicted_score"] = (
                    display_df["predicted_score"]
                    .map(lambda value: f"{value:.2f} ★")
                )

            display_df["hybrid_rank_metric"] = (
                display_df["hybrid_rank_metric"]
                .map(lambda value: f"{value:.3f}")
            )

            display_df.columns = [
                "Course",
                "Institution",
                "Difficulty",
                "Credential",
                "Enrolment",
                "SBERT Similarity",
                "Mean Adjusted Rating",
                "SVD User Prediction",
                "Final Ranking Score",
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown(
                "### 📊 Side-by-side course comparison"
            )

            comparison_selection = st.multiselect(
                "Choose up to 3 recommended courses:",
                options=final_recommendations[
                    "name"
                ].tolist(),
                default=final_recommendations[
                    "name"
                ].head(2).tolist(),
                max_selections=3,
            )

            if comparison_selection:
                comparison_data = final_recommendations[
                    final_recommendations["name"].isin(
                        comparison_selection
                    )
                ]

                comparison_table = pd.DataFrame(
                    {
                        "Metric": [
                            "Institution",
                            "Difficulty",
                            "Platform course rating",
                            "Mean review rating",
                            "Mean sentiment-adjusted rating",
                            "Mean VADER sentiment",
                            "Number of text reviews",
                            "Platform enrolment",
                            "SBERT similarity",
                            "SVD user prediction",
                            "Final ranking score",
                        ]
                    }
                )

                for _, row in comparison_data.iterrows():
                    short_name = (
                        row["name"]
                        if len(str(row["name"])) <= 38
                        else f"{str(row['name'])[:35]}..."
                    )

                    svd_display = (
                        "N/A"
                        if pd.isna(row["predicted_score"])
                        else f"{row['predicted_score']:.3f}"
                    )

                    comparison_table[short_name] = [
                        row["course_organization"],
                        row["course_difficulty"],
                        (
                            f"{float(row['course_rating']):.2f} / 5"
                            if pd.notna(row["course_rating"])
                            else "N/A"
                        ),
                        f"{row['avg_review_rating']:.3f} / 5",
                        f"{row['mean_adjusted_rating']:.3f} / 5",
                        f"{row['mean_sentiment']:.4f}",
                        int(row["review_count"]),
                        row["course_students_enrolled"],
                        f"{row['content_match_score']:.4f}",
                        svd_display,
                        f"{row['hybrid_rank_metric']:.4f}",
                    ]

                st.table(comparison_table)


# ============================================================
# TAB 2 — EVALUATION / DIAGNOSTICS
# ============================================================

with tab2:
    st.subheader(
        "Evaluation & System Diagnostics"
    )

    st.info(
        "The collaborative-filtering metrics below evaluate the SVD component "
        "using out-of-fold predictions. They are deliberately not presented as "
        "overall hybrid-recommender accuracy. A valid hybrid NDCG/Precision/"
        "Recall score requires an external or manually labelled query-relevance "
        "test set, which is not present in the uploaded Coursera dataset."
    )

    st.markdown(
        "### 🧮 Collaborative-filtering evaluation"
    )

    svd_metrics_df = pd.DataFrame(
        {
            "Metric": list(
                backend["svd_metrics"].keys()
            ),
            "Value": list(
                backend["svd_metrics"].values()
            ),
        }
    )

    st.dataframe(
        svd_metrics_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "RMSE and MAE use 5-fold cross-validation. OOF NDCG@5 uses only "
        "users whose held-out interactions contain both liked and non-liked "
        "courses, and the number of evaluated users is shown explicitly."
    )

    st.markdown(
        "### 🗣️ Sentiment diagnostic"
    )

    sentiment_df = pd.DataFrame(
        {
            "Metric": list(
                backend["sentiment_metrics"].keys()
            ),
            "Value": list(
                backend["sentiment_metrics"].values()
            ),
        }
    )

    st.dataframe(
        sentiment_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Star ratings are used only as a proxy diagnostic for VADER polarity, "
        "not as a claim of ground-truth sentiment accuracy."
    )

    st.markdown(
        "### ⚙️ Current architecture"
    )

    architecture_df = pd.DataFrame(
        {
            "Component": [
                "Semantic encoder",
                "Embedding dimension",
                "Vector index",
                "Collaborative model",
                "Sentiment model",
                "Sentiment adjustment weight",
                "Course-level sentiment display",
                "Cold-start behaviour",
                "Seen-item filtering",
            ],
            "Implementation": [
                SBERT_MODEL_NAME,
                backend["embedding_dimension"],
                "FAISS IndexFlatIP over L2-normalised embeddings",
                (
                    f"Surprise SVD "
                    f"({SVD_FACTORS} latent factors)"
                ),
                "NLTK VADER",
                SENTIMENT_WEIGHT,
                "Mean across all available text-review rows per course",
                "Semantic retrieval only",
                "Previously completed courses are removed for known learners",
            ],
        }
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### ⏱️ End-to-end latency benchmark"
    )

    benchmark_user_options = [
        "Anonymous / Cold Start Learner"
    ] + professional_ids

    benchmark_user = st.selectbox(
        "Benchmark profile:",
        benchmark_user_options,
        index=(
            1
            if len(benchmark_user_options) > 1
            else 0
        ),
    )

    benchmark_weight = (
        0.0
        if benchmark_user
        == "Anonymous / Cold Start Learner"
        else 0.5
    )

    if st.button(
        "Run reproducible 6-query latency benchmark"
    ):
        with st.spinner(
            "Benchmarking the complete warm recommendation pipeline..."
        ):
            benchmark_df = run_latency_benchmark(
                backend=backend,
                selected_user=benchmark_user,
                selected_difficulty="Any",
                personalization_weight=benchmark_weight,
                top_n=DEFAULT_TOP_N,
                repeats=3,
            )

        st.session_state[
            "latency_benchmark_df"
        ] = benchmark_df

    if "latency_benchmark_df" in st.session_state:
        benchmark_df = st.session_state[
            "latency_benchmark_df"
        ]

        latency_values = benchmark_df[
            "Latency (ms)"
        ].to_numpy()

        summary1, summary2, summary3 = st.columns(3)

        with summary1:
            st.metric(
                "Median latency",
                f"{np.median(latency_values):.2f} ms",
            )

        with summary2:
            st.metric(
                "Mean latency",
                f"{np.mean(latency_values):.2f} ms",
            )

        with summary3:
            st.metric(
                "95th percentile",
                f"{np.percentile(latency_values, 95):.2f} ms",
            )

        query_summary = (
            benchmark_df
            .groupby("Query")["Latency (ms)"]
            .agg(["mean", "median", "max"])
            .reset_index()
        )

        query_summary.columns = [
            "Query",
            "Mean (ms)",
            "Median (ms)",
            "Max (ms)",
        ]

        st.dataframe(
            query_summary,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### 📚 Dataset / course statistics"
    )

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric(
            "Review interactions",
            f"{len(master_df):,}",
        )

    with stat2:
        st.metric(
            "Unique courses",
            f"{len(unique_courses):,}",
        )

    with stat3:
        st.metric(
            "Unique reviewers",
            f"{master_df['reviewer_model_id'].nunique():,}",
        )

    with stat4:
        st.metric(
            "Demonstration profiles",
            len(professional_ids),
        )

    difficulty_counts = (
        unique_courses["course_difficulty"]
        .value_counts()
        .rename_axis("Difficulty")
        .reset_index(name="Courses")
    )

    st.markdown("#### Course difficulty distribution")
    st.bar_chart(
        difficulty_counts.set_index(
            "Difficulty"
        )
    )

    top_institutions = (
        unique_courses["course_organization"]
        .value_counts()
        .head(10)
        .rename_axis("Institution")
        .reset_index(name="Courses")
    )

    st.markdown("#### Top institutions by course count")
    st.bar_chart(
        top_institutions.set_index(
            "Institution"
        )
    )

    image_folder = Path("report_images_extended")
    if image_folder.exists():
        st.markdown(
            "### 🖼️ Existing report EDA figures"
        )
        st.caption(
            "Only dataset-oriented figures are surfaced here. "
            "Previous model-evaluation figures should be regenerated because "
            "the recommendation pipeline and evaluation methodology have changed."
        )

        eda_figures = [
            (
                "Fig_4_01_Difficulty.png",
                "Corpus difficulty breakdown",
            ),
            (
                "Fig_4_02_Institutions.png",
                "Institutional distribution",
            ),
            (
                "Fig_4_08_SentimentGap.png",
                "Review-level sentiment gap",
            ),
            (
                "Fig_4_10_RatingShift.png",
                "Review-level rating calibration shift",
            ),
        ]

        available_figures = [
            (filename, caption)
            for filename, caption in eda_figures
            if (image_folder / filename).exists()
        ]

        if available_figures:
            left, right = st.columns(2)
            for index, (filename, caption) in enumerate(
                available_figures
            ):
                target_column = (
                    left if index % 2 == 0 else right
                )
                with target_column:
                    st.image(
                        str(image_folder / filename),
                        caption=caption,
                        use_container_width=True,
                    )
