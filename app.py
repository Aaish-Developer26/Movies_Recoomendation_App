import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    movies_data = pd.read_csv('data/movies.csv')
    return movies_data

# Main Function for Movie Recommendation
def recommend_movies(movie_name, movies_data, similarity):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return ["No close matches found. Please try again with a different movie title."]
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for movie in sorted_similar_movies[:30]:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)

    return recommended_movies

# Streamlit App UI
def main():
    st.title("Movie Recommendation System 🎥")
    st.markdown("Get personalized movie recommendations based on your favorite movie!")

    # Load data
    movies_data = load_data()

    # Preprocess data
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = (
        movies_data['genres'] + " " +
        movies_data['keywords'] + " " +
        movies_data['tagline'] + " " +
        movies_data['cast'] + " " +
        movies_data['director']
    )

    # Convert text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined_features)

    # Compute similarity matrix
    similarity = cosine_similarity(feature_vector)

    # Input from user
    movie_name = st.text_input("Enter the name of your favorite movie:", "")

    if st.button("Get Recommendations"):
        if movie_name.strip():
            recommended_movies = recommend_movies(movie_name, movies_data, similarity)
            if len(recommended_movies) > 1:
                st.success("Movies recommended for you:")
                for i, movie in enumerate(recommended_movies, start=1):
                    st.write(f"{i}. {movie}")
            else:
                st.warning(recommended_movies[0])
        else:
            st.error("Please enter a movie name!")

if __name__ == "__main__":
    main()
