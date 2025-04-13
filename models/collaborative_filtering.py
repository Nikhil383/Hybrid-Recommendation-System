"""
Collaborative filtering recommendation model.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    """
    User-based collaborative filtering recommendation system.
    """
    def __init__(self, n_neighbors=10, metric='cosine'):
        """
        Initialize the model.
        
        Args:
            n_neighbors (int): Number of neighbors to use
            metric (str): Distance metric to use
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None
    
    def fit(self, ratings_df):
        """
        Fit the model to the ratings data.
        
        Args:
            ratings_df (DataFrame): Ratings dataframe with columns 'user_id', 'movie_id', 'rating'
        """
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()
        
        # Fit the nearest neighbors model
        self.model.fit(self.user_item_matrix.values)
        
        return self
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
        
        Returns:
            float: Predicted rating
        """
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return 0
        
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        
        # Get the user's vector
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=self.n_neighbors+1)
        
        # Exclude the user itself
        indices = indices.flatten()[1:]
        distances = distances.flatten()[1:]
        
        # Calculate similarity weights
        weights = 1 - distances
        
        # Get the ratings of the neighbors for the movie
        neighbor_ratings = []
        for idx in indices:
            neighbor_id = self.user_ids[idx]
            neighbor_rating = self.user_item_matrix.loc[neighbor_id, movie_id]
            if neighbor_rating > 0:  # Only consider if the neighbor has rated the movie
                neighbor_ratings.append(neighbor_rating)
        
        if not neighbor_ratings:
            # If no neighbors have rated the movie, return the average rating of the user
            user_ratings = self.user_item_matrix.iloc[user_idx]
            user_ratings = user_ratings[user_ratings > 0]
            if len(user_ratings) > 0:
                return user_ratings.mean()
            else:
                return 0
        
        # Calculate the weighted average rating
        return np.mean(neighbor_ratings)
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Recommend movies for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude movies the user has already rated
        
        Returns:
            DataFrame: Recommended movies with predicted ratings
        """
        if user_id not in self.user_ids:
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])
        
        user_idx = self.user_ids.index(user_id)
        
        # Get the user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Identify movies the user hasn't rated
        if exclude_rated:
            unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        else:
            unrated_movies = self.movie_ids
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        top_n = predictions[:n_recommendations]
        return pd.DataFrame(top_n, columns=['movie_id', 'predicted_rating'])


class ItemBasedCF:
    """
    Item-based collaborative filtering recommendation system.
    """
    def __init__(self):
        """
        Initialize the model.
        """
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None
    
    def fit(self, ratings_df):
        """
        Fit the model to the ratings data.
        
        Args:
            ratings_df (DataFrame): Ratings dataframe with columns 'user_id', 'movie_id', 'rating'
        """
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()
        
        # Calculate item-item similarity matrix
        item_features = self.user_item_matrix.T.values
        self.item_similarity_matrix = cosine_similarity(item_features)
        
        return self
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
        
        Returns:
            float: Predicted rating
        """
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return 0
        
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        
        # Get the user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        # Get the similarity scores for the movie
        movie_similarities = self.item_similarity_matrix[movie_idx]
        
        # Calculate the weighted average rating
        weighted_sum = 0
        similarity_sum = 0
        
        for i, rating in enumerate(user_ratings):
            if rating > 0:  # Only consider rated items
                weighted_sum += rating * movie_similarities[i]
                similarity_sum += abs(movie_similarities[i])
        
        if similarity_sum == 0:
            return 0
        
        return weighted_sum / similarity_sum
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Recommend movies for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude movies the user has already rated
        
        Returns:
            DataFrame: Recommended movies with predicted ratings
        """
        if user_id not in self.user_ids:
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])
        
        user_idx = self.user_ids.index(user_id)
        
        # Get the user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Identify movies the user hasn't rated
        if exclude_rated:
            unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        else:
            unrated_movies = self.movie_ids
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        top_n = predictions[:n_recommendations]
        return pd.DataFrame(top_n, columns=['movie_id', 'predicted_rating'])
