"""
Evaluation metrics for recommendation systems.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
    
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
    
    Returns:
        float: MAE score
    """
    return mean_absolute_error(y_true, y_pred)

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate precision@k.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of relevant (true positive) item IDs
        k (int): Number of recommendations to consider
    
    Returns:
        float: Precision@k score
    """
    # Consider only the top-k recommendations
    recommended_items = recommended_items[:k]
    
    # Count the number of relevant items in the recommendations
    num_relevant = len(set(recommended_items) & set(relevant_items))
    
    # Calculate precision
    return num_relevant / min(k, len(recommended_items)) if len(recommended_items) > 0 else 0

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate recall@k.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of relevant (true positive) item IDs
        k (int): Number of recommendations to consider
    
    Returns:
        float: Recall@k score
    """
    # Consider only the top-k recommendations
    recommended_items = recommended_items[:k]
    
    # Count the number of relevant items in the recommendations
    num_relevant = len(set(recommended_items) & set(relevant_items))
    
    # Calculate recall
    return num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0

def evaluate_model(model, test_df, movies_df=None, k=10):
    """
    Evaluate a recommendation model.
    
    Args:
        model: Recommendation model with predict() and recommend() methods
        test_df (DataFrame): Test data with user-item-rating triplets
        movies_df (DataFrame, optional): Movies dataframe for additional information
        k (int): Number of recommendations to consider for precision and recall
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate RMSE and MAE
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        true_rating = row['rating']
        
        pred_rating = model.predict(user_id, movie_id)
        
        y_true.append(true_rating)
        y_pred.append(pred_rating)
    
    rmse_score = rmse(y_true, y_pred)
    mae_score = mae(y_true, y_pred)
    
    # Calculate precision and recall for each user
    precision_scores = []
    recall_scores = []
    
    # Get unique users in the test set
    unique_users = test_df['user_id'].unique()
    
    for user_id in unique_users:
        # Get the user's relevant items (items with high ratings)
        user_ratings = test_df[test_df['user_id'] == user_id]
        relevant_items = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
        
        if len(relevant_items) == 0:
            continue
        
        # Get recommendations for the user
        recommendations = model.recommend(user_id, n_recommendations=k)
        recommended_items = recommendations['movie_id'].tolist()
        
        # Calculate precision and recall
        precision = precision_at_k(recommended_items, relevant_items, k)
        recall = recall_at_k(recommended_items, relevant_items, k)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Calculate average precision and recall
    avg_precision = np.mean(precision_scores) if len(precision_scores) > 0 else 0
    avg_recall = np.mean(recall_scores) if len(recall_scores) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        'rmse': rmse_score,
        'mae': mae_score,
        'precision@k': avg_precision,
        'recall@k': avg_recall,
        'f1@k': f1_score
    }
