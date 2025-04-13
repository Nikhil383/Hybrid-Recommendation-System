"""
Flask web application for the recommendation system.
"""
import os
import sys
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import HybridRecommender
from models.collaborative_filtering import UserBasedCF, ItemBasedCF
from models.content_based_filtering import ContentBasedFiltering

app = Flask(__name__)

# Load data
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
movies_df = pd.read_csv(os.path.join(data_dir, 'movies.csv'))

# Initialize models
user_cf_model = UserBasedCF()
item_cf_model = ItemBasedCF()
content_model = ContentBasedFiltering()
hybrid_model = HybridRecommender()

# Fit models
user_cf_model.fit(train_df)
item_cf_model.fit(train_df)
content_model.fit(movies_df, train_df)
hybrid_model.fit(train_df, movies_df)

@app.route('/')
def index():
    """Render the home page."""
    # Get a list of users for the dropdown
    user_list = users_df['user_id'].tolist()
    user_list.sort()
    
    # Get a list of movies for the dropdown
    movie_list = movies_df[['movie_id', 'title']].values.tolist()
    movie_list.sort(key=lambda x: x[1])
    
    return render_template('index.html', users=user_list, movies=movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations for a user."""
    data = request.json
    user_id = int(data.get('user_id'))
    model_type = data.get('model_type', 'hybrid')
    n_recommendations = int(data.get('n_recommendations', 10))
    
    # Select the appropriate model
    if model_type == 'user_cf':
        model = user_cf_model
    elif model_type == 'item_cf':
        model = item_cf_model
    elif model_type == 'content':
        model = content_model
    else:  # hybrid
        model = hybrid_model
    
    # Get recommendations
    recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
    
    # Merge with movie information
    if not recommendations.empty:
        recommendations = recommendations.merge(
            movies_df[['movie_id', 'title']], on='movie_id', how='left'
        )
    
    # Convert to list of dictionaries
    result = recommendations.to_dict('records')
    
    return jsonify(result)

@app.route('/movie_details/<int:movie_id>')
def movie_details(movie_id):
    """Get details for a movie."""
    movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
    
    # Extract genre information
    genre_columns = [col for col in movies_df.columns if col not in 
                    ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']]
    
    genres = []
    for genre in genre_columns:
        if movie[genre] == 1:
            genres.append(genre)
    
    # Get similar movies
    similar_movies = content_model.get_similar_movies(movie_id, n_similar=5)
    similar_movies = similar_movies.merge(
        movies_df[['movie_id', 'title']], on='movie_id', how='left'
    )
    
    result = {
        'movie_id': movie_id,
        'title': movie['title'],
        'release_date': movie['release_date'],
        'genres': genres,
        'similar_movies': similar_movies.to_dict('records')
    }
    
    return jsonify(result)

@app.route('/user_details/<int:user_id>')
def user_details(user_id):
    """Get details for a user."""
    user = users_df[users_df['user_id'] == user_id].iloc[0]
    
    # Get the user's ratings
    user_ratings = train_df[train_df['user_id'] == user_id]
    user_ratings = user_ratings.merge(
        movies_df[['movie_id', 'title']], on='movie_id', how='left'
    )
    user_ratings = user_ratings.sort_values('rating', ascending=False)
    
    result = {
        'user_id': user_id,
        'age': user['age'],
        'gender': user['gender'],
        'occupation': user['occupation'],
        'ratings': user_ratings[['movie_id', 'title', 'rating']].to_dict('records')
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
