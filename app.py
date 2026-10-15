import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# 1. Configure the application
st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)


# 2. Load the saved objects
@st.cache_resource
def load_models():
    kmeans_model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    recommendation_data = joblib.load(
        "models/recommendation_data.pkl"
    )

    return kmeans_model, scaler, recommendation_data


kmeans_model, scaler, recommendation_data = load_models()

df = recommendation_data["data"]
audio_features = recommendation_data["features"]
X_scaled = recommendation_data["scaled_features"]


# 3. Application title
st.title("🎵 Spotify Music Recommendation System")

st.write(
    "Discover songs with similar musical characteristics "
    "using Unsupervised Machine Learning."
)

st.divider()


# 4. Search section
st.subheader("🔍 Find Your Song")

song_name = st.text_input(
    "Enter a song name",
    placeholder="Example: Blinding Lights"
)

number_of_recommendations = st.selectbox(
    "Number of recommendations",
    [5, 10, 20],
    index=1
)


# 5. Recommendation function
def recommend_songs(song_name, number_of_recommendations):

    matches = df[
        df["track_name"].str.contains(
            song_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        return None, None

    # Select the first matching song
    song_index = matches.index[0]

    # Identify its cluster
    song_cluster = df.loc[song_index, "cluster"]

    # Find all songs in the same cluster
    cluster_indices = df.index[
        df["cluster"] == song_cluster
    ]

    # Get the feature vector for the selected song
    song_position = df.index.get_loc(song_index)

    # Get the feature vectors for candidate songs
    candidate_positions = [
        df.index.get_loc(i) for i in cluster_indices
    ]

    # Calculate cosine similarity
    similarities = cosine_similarity(
        X_scaled[[song_position]],
        X_scaled[candidate_positions]
    )[0]

    # Store candidates and similarity scores
    results = pd.DataFrame({
        "index": cluster_indices,
        "similarity": similarities
    })

    # Exclude the selected song
    results = results[
        results["index"] != song_index
    ]

    # Sort by similarity
    results = results.sort_values(
        "similarity",
        ascending=False
    )

    # Select the top recommendations
    results = results.head(
        number_of_recommendations
    )

    # Display track information
    recommendations = df.loc[
        results["index"],
        ["track_name", "artists", "track_genre"]
    ].copy()

    recommendations["Similarity"] = (
        results["similarity"].to_numpy() * 100
    ).round(2)

    return df.loc[song_index], recommendations.reset_index(
        drop=True
    )


# 6. Generate recommendations
if st.button("Recommend Songs", type="primary"):

    if not song_name.strip():
        st.warning("Please enter a song name.")

    else:
        selected_song, recommendations = recommend_songs(
            song_name,
            number_of_recommendations
        )

        if selected_song is None:
            st.error(
                "Song not found. Try another song name."
            )

        elif recommendations.empty:
            st.warning(
                "No other songs were found in this cluster."
            )

        else:
            st.success("Recommendations generated!")

            # Display selected song
            st.subheader("🎧 Selected Song")

            col1, col2, col3 = st.columns(3)

            col1.write("**Song**")
            col1.write(selected_song["track_name"])

            col2.write("**Artist**")
            col2.write(selected_song["artists"])

            col3.write("**Genre**")
            col3.write(selected_song["track_genre"])

            st.divider()

            # Display recommendations
            st.subheader("🎶 Recommended Songs")

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Similarity scores are calculated from standardized "
                "audio features. They are not probabilities that "
                "you will like a song."
            )

st.divider()

st.caption(
    "Built using Python, Streamlit, K-Means clustering, "
    "and cosine similarity."
)