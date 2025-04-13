"""
Hybrid recommendation model combining collaborative filtering and content-based filtering.
"""
import numpy as np
import pandas as pd
from models.collaborative_filtering import UserBasedCF, ItemBasedCF
from models.content_based_filtering import ContentBasedFiltering

class HybridRecommender:
    """
    Hybrid recommendation system combining collaborative filtering and content-based filtering.
    """
    def __init__(self, cf_weight=0.7, cb_weight=0.3, user_cf_weight=0.5, item_cf_weight=0.5):
        """
        Initialize the hybrid model.
        
        Args:
            cf_weight (float): Weight for collaborative filtering (0-1)
            cb_weight (float): Weight for content-based filtering (0-1)
            user_cf_weight (float): Weight for user-based CF within the CF component (0-1)
            item_cf_weight (float): Weight for item-based CF within the CF component (0-1)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.user_cf_weight = user_cf_weight
        self.item_cf_weight = item_cf_weight
        
        # Initialize component models
        self.user_cf_model = UserBasedCF()
        self.item_cf_model = ItemBasedCF()
        self.cb_model = ContentBasedFiltering()
    
    def fit(self, ratings_df, movies_df):
        """
        Fit the hybrid model to the data.
        
        Args:
            ratings_df (DataFrame): Ratings dataframe
            movies_df (DataFrame): Movies dataframe
        """
        # Fit collaborative filtering models
        self.user_cf_model.fit(ratings_df)
        self.item_cf_model.fit(ratings_df)
        
        # Fit content-based filtering model
        self.cb_model.fit(movies_df, ratings_df)
        
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
        # Get predictions from component models
        user_cf_pred = self.user_cf_model.predict(user_id, movie_id)
        item_cf_pred = self.item_cf_model.predict(user_id, movie_id)
        cb_pred = self.cb_model.predict(user_id, movie_id)
        
        # Combine collaborative filtering predictions
        cf_pred = (self.user_cf_weight * user_cf_pred + 
                   self.item_cf_weight * item_cf_pred)
        
        # Combine CF and CB predictions
        hybrid_pred = (self.cf_weight * cf_pred + 
                       self.cb_weight * cb_pred)
        
        return hybrid_pred
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True, ratings_df=None):
        """
        Recommend movies for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude movies the user has already rated
            ratings_df (DataFrame, optional): Ratings dataframe to identify rated movies
        
        Returns:
            DataFrame: Recommended movies with predicted ratings
        """
        # Identify movies the user has already rated
        rated_movies = []
        if exclude_rated and ratings_df is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            rated_movies = user_ratings['movie_id'].tolist()
        
        # Get all possible movie IDs
        all_movie_ids = set(self.user_cf_model.movie_ids + 
                           self.cb_model.movie_ids)
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in all_movie_ids:
            if movie_id not in rated_movies:
                predicted_rating = self.predict(user_id, movie_id)
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        top_n = predictions[:n_recommendations]
        return pd.DataFrame(top_n, columns=['movie_id', 'predicted_rating'])
