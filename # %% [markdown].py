# %% [markdown]
# # Movie Recommendation System

# %%
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %%
movies_data = pd.read_csv('data\movies.csv')
print(movies_data.head())

# %%
movies_data.shape

# %%
# Selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)

# %%
# replacing the null values with null string or empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# %%
# Combining All the 5 features together 
combined_features = movies_data ['genres']+''+movies_data['keywords']+''+movies_data['tagline']+''+movies_data['cast']+''+movies_data['director']
print(combined_features)

# %%
# Converting the text data to feature vectors
vectorizer = TfidfVectorizer() # creating an instance of this TfidfVectorizer


# %%
feature_vector = vectorizer.fit_transform(combined_features) # creating another variable as feature_vector to store all the numerical values.
print(feature_vector)


# %% [markdown]
# # Cosine Similarity

# %%
# Getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vector)
print(similarity)
print(similarity.shape)

# %%
# Getting the movie name from user
import ipywidgets as widgets
from IPython.display import display

movie_name_widget = widgets.Text(
    description="Movie Name:",
    placeholder="Enter your favorite movie name",
)
display(movie_name_widget)

def get_movie_name(change):
    print(f"Your favorite movie is: {change.new}")

movie_name_widget.observe(get_movie_name, names='value')
print(movie_name_widget)


# %%
# Creating a list with all movie names 
movies_list = movies_data['title'].tolist()
print(movies_list)

# %%
# Finding the close match
current_movie_name = movie_name_widget.value
close_match = difflib.get_close_matches(current_movie_name, movies_list)
print(close_match)

# %%
closest_match = close_match[0]
print(closest_match)

# %%
# Finding the index of the movie with title
movie_index = movies_data[movies_data.title == closest_match]['index'].values[0]
print(movie_index)

# %%
# Getting a list of similar movies
similarity_score = list(enumerate(similarity[movie_index]))
print(similarity_score)

# %%
len(similarity_score)

# %%
# Sorting the movies based on their similarity score (Higher similarity score to the lower similarity score)
sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse = True)
print(sorted_similar_movies)

# %%
# Printing the name of similar movies based on index
print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i<30):
     print(i, '.',title_from_index)
    i+=1

# %% [markdown]
# # All code in one cell

# %%
movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


