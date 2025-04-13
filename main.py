"""
Main script to run the recommendation system.
"""
import os
import pandas as pd
import argparse
from models.collaborative_filtering import UserBasedCF, ItemBasedCF
from models.content_based_filtering import ContentBasedFiltering
from models.hybrid_model import HybridRecommender
from utils.evaluation import evaluate_model
from data.download_data import download_movielens, load_movielens_100k, preprocess_data, split_data

def main():
    """Main function to run the recommendation system."""
    parser = argparse.ArgumentParser(description='Run the recommendation system')
    parser.add_argument('--download', action='store_true', help='Download the MovieLens dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the models')
    parser.add_argument('--recommend', action='store_true', help='Generate recommendations for a user')
    parser.add_argument('--user_id', type=int, help='User ID for recommendations')
    parser.add_argument('--model', type=str, choices=['user_cf', 'item_cf', 'content', 'hybrid'], 
                        default='hybrid', help='Model to use for recommendations')
    parser.add_argument('--n_recommendations', type=int, default=10, 
                        help='Number of recommendations to generate')
    
    args = parser.parse_args()
    
    # Download the dataset if requested
    if args.download:
        print("Downloading MovieLens dataset...")
        dataset_dir = download_movielens(size="100k")
        print(f"Dataset downloaded to {dataset_dir}")
    
    # Check if processed data exists
    processed_dir = os.path.join("data", "processed")
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        print("Processed data not found. Processing the data...")
        
        # Download the dataset if it doesn't exist
        dataset_dir = download_movielens(size="100k")
        
        # Load the dataset
        ratings_df, users_df, movies_df = load_movielens_100k(dataset_dir)
        
        # Preprocess the data
        ratings_df, users_df, movies_df, user_item_matrix, movie_features = preprocess_data(
            ratings_df, users_df, movies_df
        )
        
        # Split the data
        train_df, test_df = split_data(ratings_df)
        
        # Create the processed directory if it doesn't exist
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Save the processed data
        train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
        users_df.to_csv(os.path.join(processed_dir, "users.csv"), index=False)
        movies_df.to_csv(os.path.join(processed_dir, "movies.csv"), index=False)
        
        print("Data processing complete.")
    else:
        print("Loading processed data...")
        train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
        users_df = pd.read_csv(os.path.join(processed_dir, "users.csv"))
        movies_df = pd.read_csv(os.path.join(processed_dir, "movies.csv"))
    
    # Initialize models
    print("Initializing models...")
    user_cf = UserBasedCF()
    item_cf = ItemBasedCF()
    content_based = ContentBasedFiltering()
    hybrid = HybridRecommender()
    
    # Fit models
    print("Fitting models...")
    user_cf.fit(train_df)
    item_cf.fit(train_df)
    content_based.fit(movies_df, train_df)
    hybrid.fit(train_df, movies_df)
    
    # Evaluate models if requested
    if args.evaluate:
        print("\nEvaluating models...")
        
        print("\nUser-based Collaborative Filtering:")
        user_cf_metrics = evaluate_model(user_cf, test_df)
        for metric, value in user_cf_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nItem-based Collaborative Filtering:")
        item_cf_metrics = evaluate_model(item_cf, test_df)
        for metric, value in item_cf_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nContent-based Filtering:")
        content_metrics = evaluate_model(content_based, test_df)
        for metric, value in content_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nHybrid Model:")
        hybrid_metrics = evaluate_model(hybrid, test_df)
        for metric, value in hybrid_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Generate recommendations if requested
    if args.recommend:
        if args.user_id is None:
            print("Please provide a user ID with --user_id")
            return
        
        user_id = args.user_id
        n_recommendations = args.n_recommendations
        
        # Select the appropriate model
        if args.model == 'user_cf':
            model = user_cf
            model_name = "User-based Collaborative Filtering"
        elif args.model == 'item_cf':
            model = item_cf
            model_name = "Item-based Collaborative Filtering"
        elif args.model == 'content':
            model = content_based
            model_name = "Content-based Filtering"
        else:  # hybrid
            model = hybrid
            model_name = "Hybrid Model"
        
        print(f"\nGenerating {n_recommendations} recommendations for user {user_id} using {model_name}...")
        
        # Get recommendations
        recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
        
        if recommendations.empty:
            print(f"No recommendations found for user {user_id}")
            return
        
        # Merge with movie information
        recommendations = recommendations.merge(
            movies_df[['movie_id', 'title']], on='movie_id', how='left'
        )
        
        # Print recommendations
        print("\nRecommended Movies:")
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"{i}. {row['title']} (Predicted Rating: {row['predicted_rating']:.2f})")
    
    # If no specific action is requested, print usage
    if not (args.download or args.evaluate or args.recommend):
        parser.print_help()

if __name__ == "__main__":
    main()
