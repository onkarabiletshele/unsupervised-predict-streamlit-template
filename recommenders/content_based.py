"""
    Content-based filtering for item recommendation.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.
    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.
    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!
    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.
    ---------------------------------------------------------------------
    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.
"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
ratings = pd.read_csv('resources/data/ratings.csv')
movies = pd.read_csv('resources/data/movies.csv')
imdb_df = pd.read_csv('resources/data/imdb_data.csv')

movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.
    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.
    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.
    """
    #merge imdb with movies using the inner join
    imdb = imdb_df[['movieId', 'title_cast', 'director', 'plot_keywords']]
    new_movie = imdb.merge(movies[['movieId', 'genres', 'title']], on='movieId', how='inner')

    #Data Preprocessing
    #convert data types to string 
    new_movie['title_cast'] = new_movie.title_cast.astype(str)
    new_movie['plot_keywords'] = new_movie.plot_keywords.astype(str)
    new_movie['genres'] = new_movie.genres.astype(str)
    new_movie['director'] = new_movie.director.astype(str)

    #clean directors,title_cast,plot_keywords and genres columns and remove empty,white spaces and "|" 
    new_movie['director'] = new_movie['director'].apply(lambda x: "".join(x.lower() for x in x.split()))
    new_movie['title_cast'] = new_movie['title_cast'].apply(lambda x: "".join(x.lower() for x in x.split()))
    new_movie['title_cast'] = new_movie['title_cast'].map(lambda x: x.split('|'))
    new_movie['plot_keywords'] = new_movie['plot_keywords'].map(lambda x: x.split('|'))
    new_movie['plot_keywords'] = new_movie['plot_keywords'].apply(lambda x: " ".join(x))
    new_movie['genres'] = new_movie['genres'].map(lambda x: x.lower().split('|'))
    new_movie['genres'] = new_movie['genres'].apply(lambda x: " ".join(x))

    #convert title cast back to string and remove commas
    new_movie['title_cast'] = new_movie['title_cast'].apply(lambda x: ','.join(map(str, x)))
    new_movie['title_cast'] = new_movie['title_cast'].replace(',', ' ', regex=True)

    #create a new subset table to only return required columns
    new_features = new_movie[['title_cast', 'director', 'plot_keywords', 'genres']]

    #then combine the features columns to forma new single string
    new_movie['combined_features'] = new_features['title_cast'] + ' ' + new_features['director'] + ' ' + new_features[
        'plot_keywords'] + ' ' + new_features['genres']
    movie_subset = new_movie[:subset_size]

    return movie_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.
    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """
    # Initializing the empty list of recommended movies
    processed_df = data_preprocessing(12000)
    # Instantiating and generating the count matrix
    cv = CountVectorizer()
    cv_model = cv.fit_transform(processed_df['combined_features'])
    indices = pd.DataFrame(processed_df.index)
    cosine_sim = cosine_similarity(cv_model, cv_model)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending=False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending=False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending=False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending=False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes, [idx_1, idx_2, idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies