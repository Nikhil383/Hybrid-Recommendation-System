"""
Script to download the MovieLens dataset for the recommendation system.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile
import urllib.request

# URLs for the MovieLens datasets
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_movielens(size="100k", data_dir="./data"):
    """
    Download the MovieLens dataset.
    
    Args:
        size (str): Size of the dataset ('100k' or '1m')
        data_dir (str): Directory to save the data
    
    Returns:
        str: Path to the extracted data directory
    """
    if size == "100k":
        url = MOVIELENS_100K_URL
        dataset_dir = os.path.join(data_dir, "ml-100k")
        zip_file = os.path.join(data_dir, "ml-100k.zip")
    elif size == "1m":
        url = MOVIELENS_1M_URL
        dataset_dir = os.path.join(data_dir, "ml-1m")
        zip_file = os.path.join(data_dir, "ml-1m.zip")
    else:
        raise ValueError("Size must be '100k' or '1m'")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(dataset_dir):
        print(f"Downloading MovieLens {size} dataset...")
        urllib.request.urlretrieve(url, zip_file)
        
        # Extract the dataset
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove the zip file
        os.remove(zip_file)
        print(f"Downloaded and extracted to {dataset_dir}")
    else:
        print(f"Dataset already exists at {dataset_dir}")
    
    return dataset_dir

def load_movielens_100k(data_dir="./data/ml-100k"):
    """
    Load the MovieLens 100K dataset.
    
    Args:
        data_dir (str): Path to the dataset directory
    
    Returns:
        tuple: (ratings_df, users_df, movies_df)
    """
    # Load ratings
    ratings_df = pd.read_csv(os.path.join(data_dir, 'u.data'), 
                             sep='\t', 
                             names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Load user information
    users_df = pd.read_csv(os.path.join(data_dir, 'u.user'), 
                           sep='|', 
                           names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    
    # Load movie information
    movies_df = pd.read_csv(os.path.join(data_dir, 'u.item'), 
                            sep='|', 
                            encoding='latin-1',
                            names=['movie_id', 'title', 'release_date', 'video_release_date',
                                   'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                   'Thriller', 'War', 'Western'])
    
    return ratings_df, users_df, movies_df

def split_data(ratings_df, test_size=0.2, random_state=42):
    """
    Split the ratings data into training and testing sets.
    
    Args:
        ratings_df (DataFrame): Ratings dataframe
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def preprocess_data(ratings_df, users_df, movies_df):
    """
    Preprocess the data for the recommendation system.
    
    Args:
        ratings_df (DataFrame): Ratings dataframe
        users_df (DataFrame): Users dataframe
        movies_df (DataFrame): Movies dataframe
    
    Returns:
        tuple: (processed_ratings, processed_users, processed_movies)
    """
    # Convert timestamp to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    # Create a user-item matrix
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Extract movie genres as a feature matrix
    genre_columns = movies_df.columns[5:].tolist()
    movie_features = movies_df[['movie_id'] + genre_columns].copy()
    
    return ratings_df, users_df, movies_df, user_item_matrix, movie_features

if __name__ == "__main__":
    # Download the dataset
    dataset_dir = download_movielens(size="100k")
    
    # Load the dataset
    ratings_df, users_df, movies_df = load_movielens_100k(dataset_dir)
    
    # Preprocess the data
    ratings_df, users_df, movies_df, user_item_matrix, movie_features = preprocess_data(
        ratings_df, users_df, movies_df
    )
    
    # Split the data
    train_df, test_df = split_data(ratings_df)
    
    # Save the processed data
    processed_dir = os.path.join("data", "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    users_df.to_csv(os.path.join(processed_dir, "users.csv"), index=False)
    movies_df.to_csv(os.path.join(processed_dir, "movies.csv"), index=False)
    
    print(f"Data processing complete. Files saved to {processed_dir}")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
