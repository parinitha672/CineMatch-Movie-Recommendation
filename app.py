
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------
# CINEMATIC DESIGN
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 0% 0%, #29204c 0%, transparent 35%),
        radial-gradient(ellipse at 100% 5%, #123c50 0%, transparent 30%),
        #0b1020;
    color: #f1f5ff;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(11,16,32,.85);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#171c32,#101628);
    border-right: 1px solid #303954;
}

[data-testid="stSidebar"] * {
    color: #edf0ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1,h2,h3,h4 {
    font-family: 'Outfit',sans-serif !important;
    color: #f5f3ff !important;
}

p,label {
    color: #d5dcef;
}

.hero {
    padding: 34px;
    border-radius: 24px;
    margin-bottom: 25px;
    background: linear-gradient(120deg,#302052,#20294d 55%,#124253);
    border: 1px solid #514774;
    box-shadow: 0 18px 45px #0003;
}

.eyebrow {
    color: #c4b5fd;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 800;
}

.hero h1 {
    font-size: clamp(32px,4vw,48px);
    margin: 12px 0;
}

.hero p {
    max-width: 720px;
    color: #d3daf0;
}

.metric-card {
    background: linear-gradient(145deg,#1d2741,#141c30);
    border: 1px solid #303c59;
    border-radius: 17px;
    padding: 19px;
    min-height: 120px;
    box-shadow: 0 8px 25px #0002;
}

.metric-icon {
    font-size: 23px;
}

.metric-label {
    color: #a8b4ce;
    font-size: 12px;
    margin-top: 8px;
}

.metric-value {
    color: #f5f3ff;
    font-family: 'Outfit',sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin-top: 4px;
}

.movie-card {
    background: linear-gradient(150deg,#202b47,#151d31);
    border: 1px solid #35415f;
    border-radius: 17px;
    padding: 18px;
    min-height: 190px;
    margin-bottom: 14px;
    transition: .2s ease;
}

.movie-card:hover {
    border-color: #a78bfa;
    transform: translateY(-2px);
}

.movie-rank {
    color: #c4b5fd;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.movie-title {
    color: #ffffff;
    font-family: 'Outfit',sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin: 12px 0;
}

.movie-meta {
    color: #b5c0d9;
    font-size: 12px;
    margin-top: 8px;
}

.score {
    display: inline-block;
    background: #493477;
    color: #e9ddff;
    border: 1px solid #7656a8;
    padding: 5px 9px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}

.genre-tag {
    display: inline-block;
    background: #183c4b;
    color: #a5f3fc;
    padding: 4px 8px;
    margin: 3px 3px 0 0;
    border-radius: 7px;
    font-size: 10px;
}

div[data-baseweb="tab-list"] {
    gap: 7px;
    background: #12192b;
    padding: 8px;
    border: 1px solid #293650;
    border-radius: 14px;
}

button[data-baseweb="tab"] {
    border-radius: 9px;
    padding: 10px 14px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(110deg,#7c3aed,#a855f7);
    color: white;
}

.stButton > button {
    background: linear-gradient(110deg,#7c3aed,#a855f7);
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
}

.stButton > button:hover {
    border: 1px solid #c4b5fd;
    color: white;
}

.stTextInput input,.stNumberInput input,
.stSelectbox [data-baseweb="select"] {
    background: #121a2c;
    color: #f1f5ff;
    border-radius: 9px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #303c59;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #8e9ab5;
    font-size: 12px;
    padding: 30px 0 10px;
}

hr {
    border-color: #293650;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
@st.cache_data

def load_data():
    ratings_path = DATA_DIR / "u.data"
    movies_path = DATA_DIR / "u.item"

    # Check that both dataset files exist
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file missing: {ratings_path}")

    if not movies_path.exists():
        raise FileNotFoundError(f"Movies file missing: {movies_path}")

    # Load ratings
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # Load movie information
    movie_columns = [
        "movie_id", "title", "release_date",
        "video_release_date", "IMDb_URL"
    ] + [f"genre_{i}" for i in range(19)]

    movies = pd.read_csv(
        movies_path,
        sep="|",
        names=movie_columns,
        encoding="latin-1",
        usecols=range(24)
    )

    # Clean ratings
    for col in ["user_id", "movie_id", "rating"]:
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")

    ratings = ratings.dropna(
        subset=["user_id", "movie_id", "rating"]
    )

    ratings = ratings[ratings["rating"].between(1, 5)].copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)

    # Clean movie information
    movies = movies.dropna(subset=["title"]).copy()

    movies["movie_id"] = pd.to_numeric(
        movies["movie_id"], errors="coerce"
    )

    movies = movies.dropna(subset=["movie_id"])
    movies["movie_id"] = movies["movie_id"].astype(int)

    return ratings, movies



ratings_path = DATA_DIR / "u.data"
movies_path = DATA_DIR / "u.item"

st.write("Ratings file exists:", ratings_path.exists())
st.write("Movies file exists:", movies_path.exists())

if ratings_path.exists():
    st.write("Ratings file size:", ratings_path.stat().st_size, "bytes")

if movies_path.exists():
    st.write("Movies file size:", movies_path.stat().st_size, "bytes")

try:
    ratings, movies = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {type(e).__name__}: {e}")
    st.stop()

if ratings is None or movies is None:
    st.error("The dataset files could not be found.")
    st.stop()


GENRES = [
    "Unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

GENRE_COLS = [f"genre_{i}" for i in range(19)]


def get_genres(row):
    return [
        GENRES[i]
        for i, col in enumerate(GENRE_COLS)
        if col in row.index and row[col] == 1 and i != 0
    ]


def movie_card(title, subtitle, score, rank=None):
    rank_text = f"#{rank} · FEATURED PICK" if rank else "CINEMATCH PICK"
    safe_title = str(title).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_subtitle = str(subtitle).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-rank">{rank_text}</div>
            <div class="movie-title">🎬 {safe_title}</div>
            <div class="movie-meta">{safe_subtitle}</div>
            <div class="score">{score}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# RECOMMENDATION ENGINES
# ---------------------------------------------------------
@st.cache_data
def build_popularity(ratings_df, movies_df):
    stats = ratings_df.groupby("movie_id")["rating"].agg(
        average_rating="mean",
        rating_count="count"
    ).reset_index()

    global_mean = ratings_df["rating"].mean()
    minimum_votes = 50

    stats["weighted_score"] = (
        stats["rating_count"] / (stats["rating_count"] + minimum_votes)
        * stats["average_rating"]
        + minimum_votes / (stats["rating_count"] + minimum_votes)
        * global_mean
    )

    return stats.merge(
        movies_df[["movie_id", "title"]],
        on="movie_id", how="left"
    ).sort_values("weighted_score", ascending=False)


@st.cache_data
def build_genre_similarity(movies_df):
    matrix = movies_df[GENRE_COLS].fillna(0).to_numpy()
    return cosine_similarity(matrix)


@st.cache_data
def build_user_model(ratings_df):
    matrix = ratings_df.pivot_table(
        index="user_id", columns="movie_id",
        values="rating", aggfunc="mean"
    ).fillna(0)

    similarity = cosine_similarity(matrix)
    similarity_df = pd.DataFrame(
        similarity, index=matrix.index, columns=matrix.index
    )
    return matrix, similarity_df


popular_movies = build_popularity(ratings, movies)
genre_similarity = build_genre_similarity(movies)
user_matrix, user_similarity = build_user_model(ratings)

# Position-based lookup avoids errors if DataFrame indexes are non-contiguous.
movie_positions = {
    int(movie_id): pos
    for pos, movie_id in enumerate(movies["movie_id"].tolist())
}
title_to_movie_id = (
    movies.drop_duplicates("title")
    .set_index("title")["movie_id"].to_dict()
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("MOVIE INTELLIGENCE")
    st.divider()

    st.markdown("### 🎛️ Explore")
    st.write("Discover movies based on ratings, genres, and user preferences.")
    st.divider()

    st.markdown("### 📚 MovieLens 100K")
    st.write(f"🎞️ **{movies['movie_id'].nunique():,}** movies")
    st.write(f"👥 **{ratings['user_id'].nunique():,}** users")
    st.write(f"⭐ **{len(ratings):,}** ratings")

    st.divider()
    st.markdown("### 🧠 Recommendation methods")
    st.caption("• Weighted popularity ranking")
    st.caption("• Genre-based similarity")
    st.caption("• User collaborative filtering")
    st.caption("• MAE / RMSE evaluation")
    st.divider()
    st.caption("Python · Pandas · Scikit-learn · Streamlit")


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ YOUR PERSONAL MOVIE DISCOVERY SPACE</div>
    <h1>Find your next<br>favourite movie.</h1>
    <p>Explore popular classics, discover similar movies, and uncover
    personalised picks using patterns from MovieLens ratings.</p>
</div>
""", unsafe_allow_html=True)

metrics = [
    ("🎞️", "MOVIE LIBRARY", f"{movies['movie_id'].nunique():,}"),
    ("👥", "ACTIVE RATERS", f"{ratings['user_id'].nunique():,}"),
    ("⭐", "TOTAL RATINGS", f"{len(ratings):,}"),
    ("💜", "AVERAGE RATING", f"{ratings['rating'].mean():.2f} / 5")
]

cols = st.columns(4)
for col, (icon, label, value) in zip(cols, metrics):
    with col:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            </div>""",
            unsafe_allow_html=True
        )

st.write("")
st.markdown("## ✨ Explore CineMatch")
st.caption("Choose a section below to begin your movie discovery.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Popular Movies",
    "🎭 Similar Movies",
    "✨ For You",
    "📊 Model Evaluation"
])


# ---------------------------------------------------------
# TAB 1: POPULAR MOVIES
# ---------------------------------------------------------
with tab1:
    st.markdown("### 🏆 Popular with MovieLens users")
    st.caption(
        "The weighted score balances average ratings and vote counts, "
        "so movies with very few ratings are not automatically ranked first."
    )

    number = st.slider(
        "Movies to display", 5, 30, 10, step=5, key="popular_count"
    )
    display_popular = popular_movies.head(number)

    card_cols = st.columns(2)
    for i, (_, row) in enumerate(display_popular.iterrows()):
        with card_cols[i % 2]:
            movie_row = movies[movies["movie_id"] == row["movie_id"]]
            genre_text = ", ".join(get_genres(movie_row.iloc[0])) if not movie_row.empty else "Genres unavailable"

            movie_card(
                row["title"],
                f"⭐ {row['average_rating']:.2f}/5  ·  {int(row['rating_count']):,} ratings · {genre_text}",
                f"Weighted score: {row['weighted_score']:.2f}",
                rank=i + 1
            )

    with st.expander("📋 View detailed ranking table"):
        table = display_popular[
            ["title", "average_rating", "rating_count", "weighted_score"]
        ].rename(columns={
            "title": "Movie",
            "average_rating": "Average rating",
            "rating_count": "Number of ratings",
            "weighted_score": "Weighted score"
        }).copy()

        table["Average rating"] = table["Average rating"].round(2)
        table["Weighted score"] = table["Weighted score"].round(3)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### 📈 Weighted-score comparison")
    st.bar_chart(
        display_popular.head(8).set_index("title")[["weighted_score"]],
        horizontal=True
    )


# ---------------------------------------------------------
# TAB 2: SIMILAR MOVIES
# ---------------------------------------------------------
with tab2:
    st.markdown("### 🎭 Find movies with a similar vibe")
    st.caption(
        "Genre similarity compares each movie's genre profile using cosine similarity. "
        "It does not use external posters or movie databases."
    )

    movie_titles = sorted(title_to_movie_id.keys())
    default_title = (
        "Star Wars (1977)"
        if "Star Wars (1977)" in movie_titles
        else movie_titles[0]
    )

    selected_movie = st.selectbox(
        "Choose a movie you like",
        movie_titles,
        index=movie_titles.index(default_title),
        key="similar_movie"
    )

    count_similar = st.slider(
        "Recommendations to show", 5, 15, 10, key="similar_count"
    )

    if st.button("✨ Discover similar movies", key="find_similar"):
        selected_id = int(title_to_movie_id[selected_movie])
        movie_position = movie_positions[selected_id]

        scores = genre_similarity[movie_position]
        ordered_positions = np.argsort(scores)[::-1]

        recommendations = []
        for pos in ordered_positions:
            row = movies.iloc[int(pos)]
            if int(row["movie_id"]) == selected_id:
                continue
            score = float(scores[pos])
            if score <= 0:
                continue

            recommendations.append((row, score))
            if len(recommendations) >= count_similar:
                break

        st.markdown(f"#### 🎬 Since you liked {selected_movie}")

        if recommendations:
            card_cols = st.columns(2)
            for i, (row, score) in enumerate(recommendations):
                genres = ", ".join(get_genres(row)) or "Genre not specified"
                with card_cols[i % 2]:
                    movie_card(
                        row["title"],
                        f"Genres: {genres}",
                        f"Genre similarity: {score:.0%}",
                        rank=i + 1
                    )
        else:
            st.info("No similar movies were found.")


# ---------------------------------------------------------
# TAB 3: COLLABORATIVE FILTERING
# ---------------------------------------------------------
with tab3:
    st.markdown("### ✨ Recommendations tailored to a user")
    st.caption(
        "Choose a MovieLens user ID. The system finds users with similar "
        "rating patterns and estimates scores for movies this user has not rated."
    )

    valid_users = user_matrix.index.tolist()
    user_id = st.selectbox(
        "Choose a MovieLens user ID",
        valid_users,
        index=0,
        key="recommend_user"
    )

    n_recommendations = st.slider(
        "Recommendations to show", 5, 15, 10, key="user_count"
    )
    neighbor_count = st.slider(
        "Similar users to consider", 5, 30, 20, key="neighbor_count"
    )

    if st.button("🚀 Generate recommendations", key="generate_user"):
        similarities = user_similarity.loc[user_id].drop(
            labels=[user_id], errors="ignore"
        ).sort_values(ascending=False)

        neighbors = similarities.head(neighbor_count)
        rated_by_user = user_matrix.loc[user_id]
        unseen = rated_by_user[rated_by_user == 0].index

        rows = []
        for movie_id in unseen:
            neighbor_ratings = user_matrix.loc[neighbors.index, movie_id]
            mask = neighbor_ratings > 0

            if not mask.any():
                continue

            weights = neighbors[mask]
            values = neighbor_ratings[mask]
            denominator = weights.abs().sum()

            if denominator == 0:
                continue

            predicted = float((values * weights).sum() / denominator)
            rows.append((int(movie_id), predicted))

        recommendations = sorted(
            rows, key=lambda item: item[1], reverse=True
        )[:n_recommendations]

        if recommendations:
            result = pd.DataFrame(
                recommendations,
                columns=["movie_id", "Predicted rating"]
            ).merge(
                movies[["movie_id", "title"]].drop_duplicates("movie_id"),
                on="movie_id", how="left"
            )

            result["Predicted rating"] = result["Predicted rating"].clip(1, 5)
            result["Predicted rating"] = result["Predicted rating"].round(2)

            st.success(f"Recommendations generated for user {user_id}.")

            card_cols = st.columns(2)
            for i, (_, row) in enumerate(result.iterrows()):
                with card_cols[i % 2]:
                    movie_card(
                        row["title"],
                        "Suggested from similar users' rating patterns",
                        f"Estimated rating: {row['Predicted rating']:.2f}/5",
                        rank=i + 1
                    )

            with st.expander("📋 View recommendation table"):
                st.dataframe(
                    result[["title", "Predicted rating"]].rename(
                        columns={"title": "Recommended movie"}
                    ),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No recommendations could be generated for this user.")


# ---------------------------------------------------------
# TAB 4: MODEL EVALUATION
# ---------------------------------------------------------
with tab4:
    st.markdown("### 📊 Model performance")
    st.caption(
        "Review the saved evaluation results. Lower MAE and RMSE generally "
        "indicate smaller rating-prediction errors."
    )

    evaluation_path = OUTPUT_DIR / "model_evaluation.csv"

    if evaluation_path.exists():
        evaluation = pd.read_csv(evaluation_path)

        st.markdown("#### Evaluation results")
        st.dataframe(
            evaluation, use_container_width=True, hide_index=True
        )

        metric_lookup = {
            str(col).strip().lower(): col
            for col in evaluation.columns
        }

        mae_col = next(
            (metric_lookup[k] for k in metric_lookup
             if k == "mae" or "mean_absolute_error" in k),
            None
        )
        rmse_col = next(
            (metric_lookup[k] for k in metric_lookup
             if k == "rmse" or "root_mean_squared_error" in k),
            None
        )

        available_metrics = []
        if mae_col:
            available_metrics.append(
                ("MAE", pd.to_numeric(
                    evaluation[mae_col], errors="coerce"
                ).mean())
            )
        if rmse_col:
            available_metrics.append(
                ("RMSE", pd.to_numeric(
                    evaluation[rmse_col], errors="coerce"
                ).mean())
            )

        if available_metrics:
            metric_cols = st.columns(len(available_metrics))
            for col, (label, value) in zip(metric_cols, available_metrics):
                with col:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value:.4f}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

        st.info(
            "Use the model names and metric definitions in the table "
            "when comparing results. These metrics measure prediction error, "
            "not the quality of every recommendation."
        )
    else:
        st.warning(
            "model_evaluation.csv was not found in outputs/. "
            "Run your model evaluation cells first."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    🎬 CineMatch · Movie Recommendation System<br>
    Built with the MovieLens 100K dataset · Python · Pandas ·
    Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
