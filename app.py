import streamlit as st
import pandas as pd
import pickle

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ==================================================
# LOAD ARTIFACTS
# ==================================================

@st.cache_resource
def load_artifacts():

    svd = pickle.load(
        open(
            "artifacts/svd_model.pkl",
            "rb"
        )
    )

    tfidf = pickle.load(
        open(
            "artifacts/tfidf.pkl",
            "rb"
        )
    )

    similarity = pickle.load(
        open(
            "artifacts/similarity.pkl",
            "rb"
        )
    )

    movie_indices = pickle.load(
        open(
            "artifacts/movie_indices.pkl",
            "rb"
        )
    )

    movie_id_to_title = pickle.load(
        open(
            "artifacts/movie_id_to_title.pkl",
            "rb"
        )
    )

    movies = pd.read_csv(
        "artifacts/movies_processed.csv"
    )

    ratings = pd.read_csv(
        "artifacts/ratings_processed.csv"
    )

    popular_movies = pd.read_csv(
        "artifacts/popular_movies.csv"
    )

    return (
        svd,
        tfidf,
        similarity,
        movie_indices,
        movie_id_to_title,
        movies,
        ratings,
        popular_movies
    )


(
    svd,
    tfidf,
    similarity,
    movie_indices,
    movie_id_to_title,
    movies,
    ratings,
    popular_movies
) = load_artifacts()

# ==================================================
# RECOMMENDATION FUNCTIONS
# ==================================================

def cold_start_recommend(top_n=10):

    return popular_movies[
        ["title", "avg_rating"]
    ].head(top_n)


def content_recommend_scores(
        movie_title,
        top_n=50
):

    idx = movie_indices[movie_title]

    scores = list(
        enumerate(
            similarity[idx]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[
        1:top_n+1
    ]

    recommendations = []

    for movie_idx, score in scores:

        recommendations.append(
            {
                "movieId":
                movies.iloc[
                    movie_idx
                ]["movieId"],

                "title":
                movies.iloc[
                    movie_idx
                ]["title"],

                "content_score":
                score
            }
        )

    return pd.DataFrame(
        recommendations
    )


def hybrid_recommend(
        user_id,
        movie_title,
        top_n=10
):

    # -------------------------
    # Cold Start Handling
    # -------------------------

    if user_id not in ratings[
        "userId"
    ].unique():

        return cold_start_recommend(
            top_n
        )

    # -------------------------
    # Content Candidates
    # -------------------------

    content_df = content_recommend_scores(
        movie_title,
        top_n=100
    )

    watched_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].tolist()

    content_df = content_df[
        ~content_df[
            "movieId"
        ].isin(
            watched_movies
        )
    ]

    hybrid_scores = []

    for _, row in content_df.iterrows():

        movie_id = row["movieId"]

        content_score = row[
            "content_score"
        ]

        collab_score = svd.predict(
            user_id,
            movie_id
        ).est

        hybrid_score = (
            0.4 * content_score
            +
            0.6 * (
                collab_score / 5
            )
        )

        hybrid_scores.append(
            {
                "title":
                row["title"],

                "content_score":
                round(
                    content_score,
                    4
                ),

                "collaborative_score":
                round(
                    collab_score,
                    4
                ),

                "hybrid_score":
                round(
                    hybrid_score,
                    4
                )
            }
        )

    results = pd.DataFrame(
        hybrid_scores
    )

    results = results.sort_values(
        "hybrid_score",
        ascending=False
    )

    return results.head(
        top_n
    )

# ==================================================
# UI
# ==================================================

st.title(
    "🎬 Hybrid Movie Recommendation System"
)

st.markdown(
    """
    Personalized movie recommendations using:

    - Content-Based Filtering
    - Collaborative Filtering (SVD)
    - Hybrid Recommendation Strategy
    """
)

# ==================================================
# MODEL METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "RMSE",
    "0.8807"
)

col2.metric(
    "Precision@10",
    "0.5660"
)

col3.metric(
    "Recall@10",
    "0.2794"
)

st.divider()

# ==================================================
# USER INPUTS
# ==================================================

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    value=1
)

movie_title = st.selectbox(
    "Select a Movie",
    sorted(
        movies["title"].unique()
    )
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# ==================================================
# BUTTON
# ==================================================

if st.button(
    "Get Recommendations"
):

    recommendations = hybrid_recommend(
        user_id=user_id,
        movie_title=movie_title,
        top_n=top_n
    )

    st.success(
        f"Top {top_n} Recommendations"
    )

    st.dataframe(
        recommendations,
        use_container_width=True
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.markdown(
    "Built using Content-Based Filtering, SVD Collaborative Filtering and Hybrid Recommendation Techniques."
)