import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from html import escape
import re
import io
import zipfile


st.set_page_config(
    page_title="MovieDate",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CONFIG
# =========================
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# لو عندك TMDB API key حطيها هنا
TMDB_API_KEY = "d5f9a81215e04dfff4585e9c6267d7dd"   # مثال: "xxxxxxxxxxxxxxxx"

# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    data_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    response = requests.get(data_url, timeout=30)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    movies = pd.read_csv(zip_file.open("ml-latest-small/movies.csv"))
    ratings = pd.read_csv(zip_file.open("ml-latest-small/ratings.csv"))
    links = pd.read_csv(zip_file.open("ml-latest-small/links.csv"))

    return movies, ratings, links


movies, ratings, links = load_data()

movies_small = movies.head(5000).copy()
movies_small["genres"] = movies_small["genres"].fillna("")

movies_small["year"] = movies_small["title"].str.extract(r"\((\d{4})\)")
movies_small["year"] = movies_small["year"].fillna("N/A")

movie_avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movie_avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)

movies_small = movies_small.merge(movie_avg_ratings, on="movieId", how="left")
movies_small["avg_rating"] = movies_small["avg_rating"].fillna(0).round(2)

movies_small = movies_small.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

# =========================
# CONTENT-BASED MODEL
# =========================
cv = CountVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None)
genre_matrix = cv.fit_transform(movies_small["genres"])
similarity = cosine_similarity(genre_matrix)

movie_list = sorted(movies_small["title"].dropna().unique())

def normalize_movie_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def filter_movie_list(query, limit=120):
    normalized_query = normalize_movie_text(query)
    if not normalized_query:
        return movie_list

    query_words = normalized_query.split()
    matches = []
    for title in movie_list:
        normalized_title = normalize_movie_text(title)
        if all(word in normalized_title for word in query_words):
            starts_with_score = 0 if normalized_title.startswith(normalized_query) else 1
            matches.append((starts_with_score, normalized_title.index(query_words[0]), title))

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return [title for _, _, title in matches[:limit]]

def searchable_movie_selectbox(title, key, default_index=0):
    search_value = st.text_input(
        f"Search {title.lower()}",
        key=f"{key}_search",
        placeholder="Type a movie name...",
        label_visibility="collapsed"
    )
    options = filter_movie_list(search_value)

    if not options:
        st.caption("No matches found. Try another word from the title.")
        options = movie_list

    return st.selectbox(
        title,
        options,
        index=min(default_index, len(options) - 1),
        label_visibility="collapsed",
        key=key
    )

def get_tmdb_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    if not TMDB_API_KEY:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return TMDB_IMAGE_BASE + poster_path
    except:
        return None
    return None

def get_movie_details(title):
    row = movies_small[movies_small["title"] == title].iloc[0]
    return {
        "title": row["title"],
        "genres": row["genres"].replace("|", " • "),
        "year": row["year"],
        "rating": row["avg_rating"],
        "tmdbId": row["tmdbId"]
    }

def render_movie_card(item, badge=None):
    badge_html = f'<div class="rank-badge">{badge}</div>' if badge is not None else ""
    genres = str(item["genres"]).replace("â€¢", "•")
    poster_url = get_tmdb_poster(item["tmdbId"])
    poster_html = (
        f'<div class="poster-wrap"><img src="{poster_url}"></div>'
        if poster_url
        else '<div class="poster-placeholder">🎞️</div>'
    )
    st.markdown(f"""
    <div class="movie-card">
        {badge_html}
        {poster_html}
        <div class="movie-title">{escape(str(item["title"]))}</div>
        <div class="movie-meta"><b>Genres:</b> {escape(genres)}</div>
        <div class="movie-meta"><b>Year:</b> {escape(str(item["year"]))}</div>
        <div class="movie-rating">★ {escape(str(item["rating"]))}</div>
    </div>
    """, unsafe_allow_html=True)

def recommend_two_movies(movie1, movie2, n=6):
    idx1 = movies_small[movies_small["title"] == movie1].index[0]
    idx2 = movies_small[movies_small["title"] == movie2].index[0]

    combined_similarity = (similarity[idx1] + similarity[idx2]) / 2
    scores = list(enumerate(combined_similarity))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in sorted_scores:
        row = movies_small.iloc[i[0]]
        title = row["title"]
        if title != movie1 and title != movie2:
            recommended.append({
                "title": row["title"],
                "genres": row["genres"].replace("|", " • "),
                "year": row["year"],
                "rating": row["avg_rating"],
                "tmdbId": row["tmdbId"]
            })
        if len(recommended) == n:
            break
    return recommended

# =========================
# SESSION STATE FOR UI MODE
# =========================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# =========================
# THEME COLORS
# =========================
if st.session_state.dark_mode:
    BG = "#1A1716"
    CARD = "#2D2926"
    PANEL = "#24211F"
    TEXT = "#FDF5E6"
    MUTED = "#D5C8BE"
    BORDER = "#4E4540"
    PRIMARY = "#5D0E1D"
    SECONDARY = "#FFB347"
    INPUT_BG = "#342F2C"
else:
    BG = "#E6DDD7"
    CARD = "#F3ECE7"
    PANEL = "#FDF5E6"
    TEXT = "#2D2926"
    MUTED = "#7B6D67"
    BORDER = "#D8C9C0"
    PRIMARY = "#5D0E1D"
    SECONDARY = "#FFB347"
    INPUT_BG = "#F5ECE6"

# =========================
# CSS
# =========================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=Noto+Serif:wght@600;700;800&display=swap');

.stApp {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Be Vietnam Pro', sans-serif;
}}

.stApp * {{
    font-family: 'Be Vietnam Pro', sans-serif;
    letter-spacing: 0;
}}

.block-container {{
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}}

[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
    padding-top: 1rem;
}}

.sidebar-brand {{
    color: {PRIMARY};
    font-family: 'Noto Serif', serif;
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 1rem;
}}

.sidebar-box {{
    padding: 0;
    margin-bottom: 1rem;
}}

.sidebar-divider {{
    height: 1px;
    background: {BORDER};
    margin: 1.7rem 0;
    opacity: 0.9;
}}

.sidebar-title {{
    color: {TEXT};
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}}

.hero {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}}

.hero-title {{
    color: {PRIMARY};
    font-family: 'Noto Serif', serif;
    font-size: 3.25rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.45rem;
}}

.hero-subtitle {{
    color: {MUTED};
    font-size: 1.08rem;
}}

.panel {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1.15rem;
    height: 100%;
}}

.section-title {{
    color: {TEXT};
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.85rem;
}}

.panel .section-title::first-letter {{
    font-size: 0;
    color: transparent;
}}

.search-box {{
    background: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: {MUTED};
    margin-top: 0.35rem;
}}

.results-wrap {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 1.2rem;
    margin-top: 1rem;
}}

.results-wrap:empty {{
    display: none;
}}

.search-box {{
    display: none;
}}

.movie-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 0.8rem;
    box-shadow: 0 4px 14px rgba(45,41,38,0.05);
    height: 100%;
}}

.rank-badge {{
    display: inline-block;
    background: {PRIMARY};
    color: white;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 700;
    padding: 0.3rem 0.68rem;
    margin-bottom: 0.75rem;
}}

.poster-wrap img {{
    width: 100%;
    border-radius: 10px;
    object-fit: cover;
}}

.poster-placeholder {{
    width: 100%;
    height: 310px;
    border-radius: 10px;
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY} 58%, {SECONDARY} 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 3rem;
    font-weight: 700;
}}

.movie-title {{
    color: {TEXT};
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.35;
    margin-top: 0.8rem;
    margin-bottom: 0.45rem;
}}

.movie-meta {{
    color: {MUTED};
    font-size: 0.96rem;
    margin-bottom: 0.35rem;
}}

.movie-rating {{
    color: #9B6900;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0.35rem;
}}

.footer {{
    text-align: center;
    color: {MUTED};
    padding-top: 1.6rem;
    font-size: 0.95rem;
}}

.stSelectbox [data-baseweb="select"] > div,
.stSlider [data-baseweb="slider"],
.stTextInput input {{
    background-color: {INPUT_BG};
    border-color: {BORDER};
}}

.stTextInput input {{
    border-radius: 8px;
    min-height: 50px;
}}

.stButton > button {{
    background: {PRIMARY};
    color: #FFFFFF;
    border: 1px solid {PRIMARY};
    border-radius: 8px;
    font-weight: 700;
}}

.stButton > button:hover {{
    background: #4A0715;
    border-color: #4A0715;
    color: #FFFFFF;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🎬 MovieDate</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Number of recommendations</div>', unsafe_allow_html=True)
    num_results = st.slider("Number of recommendations", 3, 12, 6, label_visibility="collapsed")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">UI Mode</div>', unsafe_allow_html=True)
    dark_toggle = st.toggle("Enable Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_toggle
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    This app recommends movies based on genre similarity.

    Choose two movies you like and discover similar ones that match their overall vibe.
    """)

# =========================
# HEADER
# =========================
top_left, top_right = st.columns([4, 1.4])

with top_left:
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">Movie Recommender</div>
        <div class="hero-subtitle">
            Choose two movies and get recommendations based on genre similarity.
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="search-box">🔍 Search movies...</div>', unsafe_allow_html=True)

# =========================
# CONTROLS
# =========================
left, middle, right = st.columns([2.2, 2.2, 1.2])

with left:
    st.markdown('<div class="panel"><div class="section-title">1️⃣ Choose first movie</div>', unsafe_allow_html=True)
    movie1 = searchable_movie_selectbox("Choose first movie", "movie1")
    st.markdown('</div>', unsafe_allow_html=True)

with middle:
    st.markdown('<div class="panel"><div class="section-title">2️⃣ Choose second movie</div>', unsafe_allow_html=True)
    movie2 = searchable_movie_selectbox("Choose second movie", "movie2", default_index=1)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    run = st.button("✨ Get Recommendations", use_container_width=True)

st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Selected Movies</div>', unsafe_allow_html=True)
selected_cols = st.columns(2)
for selected_col, selected_movie, selected_label in zip(selected_cols, [movie1, movie2], ["Primary", "Secondary"]):
    with selected_col:
        render_movie_card(get_movie_details(selected_movie), selected_label)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RESULTS
# =========================
if run:
    results = recommend_two_movies(movie1, movie2, num_results)

    st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🍿 Recommended Movies</div>', unsafe_allow_html=True)

    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, item in enumerate(results[i:i+cols_per_row]):
            with cols[j]:
                render_movie_card(item, i+j+1)
                continue
                st.markdown(f'<div class="movie-card"><div class="rank-badge">{i+j+1}</div>', unsafe_allow_html=True)

                poster_url = get_tmdb_poster(item["tmdbId"])
                if poster_url:
                    st.markdown(f'<div class="poster-wrap"><img src="{poster_url}"></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="poster-placeholder">🎞️</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="movie-title">{item["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-meta"><b>Genres:</b> {item["genres"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-meta"><b>Year:</b> {item["year"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-rating">⭐ {item["rating"]}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="results-wrap">
        <div class="section-title">🍿 Recommended Movies</div>
        <div style="color:#7B6D67; font-size:1rem;">
            Pick two movies, then click <b>Get Recommendations</b> to see your movie matches.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    MovieDate • Content-Based Filtering • Built with Streamlit
</div>
""", unsafe_allow_html=True)
