"""
Content-based filtering recommendation model.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBasedFiltering:
    """
    Content-based filtering recommendation system.
    """
    def __init__(self):
        """
        Initialize the model.
        """
        self.movie_features = None
        self.movie_ids = None
        self.movie_similarity_matrix = None
        self.user_profiles = {}
    
    def fit(self, movies_df, ratings_df=None):
        """
        Fit the model to the movie data.
        
        Args:
            movies_df (DataFrame): Movies dataframe with genre features
            ratings_df (DataFrame, optional): Ratings dataframe to build user profiles
        """
        # Extract movie features (genres)
        genre_columns = [col for col in movies_df.columns if col not in 
                         ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']]
        
        self.movie_features = movies_df[['movie_id'] + genre_columns].copy()
        self.movie_ids = self.movie_features['movie_id'].tolist()
        
        # Calculate movie-movie similarity matrix based on genres
        feature_matrix = self.movie_features.drop('movie_id', axis=1).values
        self.movie_similarity_matrix = cosine_similarity(feature_matrix)
        
        # Build user profiles if ratings are provided
        if ratings_df is not None:
            self._build_user_profiles(ratings_df)
        
        return self
    
    def _build_user_profiles(self, ratings_df):
        """
        Build user profiles based on their ratings and movie features.
        
        Args:
            ratings_df (DataFrame): Ratings dataframe
        """
        # Get unique user IDs
        user_ids = ratings_df['user_id'].unique()
        
        # Create a mapping from movie_id to index in movie_features
        movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Build user profiles
        for user_id in user_ids:
            # Get the user's ratings
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            if len(user_ratings) == 0:
                continue
            
            # Initialize user profile
            user_profile = np.zeros(self.movie_features.shape[1] - 1)  # Exclude movie_id column
            
            # Update user profile based on rated movies
            for _, row in user_ratings.iterrows():
                movie_id = row['movie_id']
                rating = row['rating']
                
                if movie_id in movie_id_to_idx:
                    movie_idx = movie_id_to_idx[movie_id]
                    movie_features = self.movie_features.iloc[movie_idx, 1:].values
                    user_profile += movie_features * (rating / 5.0)  # Normalize rating to [0, 1]
            
            # Normalize user profile
            if np.sum(user_profile) > 0:
                user_profile = user_profile / np.sum(user_profile)
            
            self.user_profiles[user_id] = user_profile
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """
        Get similar movies based on content features.
        
        Args:
            movie_id (int): Movie ID
            n_similar (int): Number of similar movies to return
        
        Returns:
            DataFrame: Similar movies with similarity scores
        """
        if movie_id not in self.movie_ids:
            return pd.DataFrame(columns=['movie_id', 'similarity'])
        
        movie_idx = self.movie_ids.index(movie_id)
        
        # Get similarity scores
        similarities = self.movie_similarity_matrix[movie_idx]
        
        # Create a list of (movie_id, similarity) tuples
        movie_similarities = [(self.movie_ids[i], similarities[i]) 
                             for i in range(len(self.movie_ids)) if i != movie_idx]
        
        # Sort by similarity
        movie_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N similar movies
        top_n = movie_similarities[:n_similar]
        return pd.DataFrame(top_n, columns=['movie_id', 'similarity'])
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
        
        Returns:
            float: Predicted rating
        """
        if movie_id not in self.movie_ids:
            return 0
        
        if user_id not in self.user_profiles:
            return 0
        
        movie_idx = self.movie_ids.index(movie_id)
        movie_features = self.movie_features.iloc[movie_idx, 1:].values
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity between user profile and movie features
        similarity = np.dot(user_profile, movie_features)
        
        # Convert similarity to rating scale (0-5)
        predicted_rating = similarity * 5.0
        
        return min(5.0, max(0.0, predicted_rating))
    
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
        if user_id not in self.user_profiles:
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])
        
        # Identify movies the user has already rated
        rated_movies = []
        if exclude_rated and ratings_df is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            rated_movies = user_ratings['movie_id'].tolist()
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in self.movie_ids:
            if movie_id not in rated_movies:
                predicted_rating = self.predict(user_id, movie_id)
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        top_n = predictions[:n_recommendations]
        return pd.DataFrame(top_n, columns=['movie_id', 'predicted_rating'])
