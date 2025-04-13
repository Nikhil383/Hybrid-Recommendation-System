# Hybrid Recommendation System

This project implements an end-to-end recommendation system that combines user-based collaborative filtering, item-based collaborative filtering, and content-based filtering into a hybrid model. The system is designed to provide personalized movie recommendations based on user preferences and movie content features.

## Overview

Recommendation systems are widely used in e-commerce, streaming platforms, and content delivery services to help users discover items they might be interested in. This project demonstrates how different recommendation approaches can be combined to create a more effective hybrid system.

## Features

- **User-based Collaborative Filtering**: Recommends items based on similar users' preferences by finding users with similar taste and suggesting items they enjoyed
- **Item-based Collaborative Filtering**: Recommends items similar to those the user has liked by analyzing item-item relationships based on user rating patterns
- **Content-based Filtering**: Recommends items with similar content features by analyzing movie genres and other attributes
- **Hybrid Model**: Combines all three approaches with configurable weights to leverage the strengths of each method
- **Web Interface**: A Flask web application to interact with the recommendation system, view user profiles, and explore movie details
- **Evaluation Metrics**: Comprehensive evaluation using RMSE, MAE, Precision@k, Recall@k, and F1@k

## Project Structure

```
.
├── data/
│   ├── download_data.py       # Script to download and preprocess the MovieLens dataset
│   └── processed/             # Directory for processed data files
├── models/
│   ├── collaborative_filtering.py  # User-based and item-based CF models
│   ├── content_based_filtering.py  # Content-based filtering model
│   └── hybrid_model.py        # Hybrid recommendation model
├── utils/
│   └── evaluation.py          # Evaluation metrics and functions
├── web/
│   ├── app.py                 # Flask web application
│   └── templates/
│       └── index.html         # HTML template for the web interface
├── main.py                    # Main script to run the recommendation system
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd recommendation-system
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download and preprocess the MovieLens dataset:
   ```
   python main.py --download
   ```

   This will:
   - Download the MovieLens 100K dataset
   - Extract and preprocess the data
   - Split the data into training and testing sets
   - Save the processed data to the `data/processed/` directory

## Usage

### Command Line Interface

The system provides a command-line interface for various operations:

1. Evaluate all recommendation models and compare their performance:
   ```
   python main.py --evaluate
   ```
   This will run all models on the test set and report metrics like RMSE, MAE, precision, recall, and F1 score.

2. Generate recommendations for a specific user:
   ```
   python main.py --recommend --user_id 1 --model hybrid --n_recommendations 10
   ```

   Parameters:
   - `--user_id`: ID of the user to generate recommendations for (required)
   - `--model`: Model to use for recommendations (options: `user_cf`, `item_cf`, `content`, `hybrid`)
   - `--n_recommendations`: Number of recommendations to generate (default: 10)

   Example output:
   ```
   Recommended Movies:
   1. Empire Strikes Back, The (1980) (Predicted Rating: 3.95)
   2. Wings of Desire (1987) (Predicted Rating: 3.84)
   3. Titanic (1997) (Predicted Rating: 3.70)
   4. Star Wars (1977) (Predicted Rating: 3.65)
   5. Angels and Insects (1995) (Predicted Rating: 3.60)
   ```

### Web Interface

The system includes a user-friendly web interface built with Flask and Bootstrap:

1. Start the Flask web application:
   ```
   python web/app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the interface to:
   - Select a user from the dropdown menu
   - Choose a recommendation model (Hybrid, User-based CF, Item-based CF, or Content-based)
   - Set the number of recommendations to generate
   - View user details including demographics and top-rated movies
   - See recommendations with predicted ratings
   - Click on movie details to view genre information and similar movies

## Dataset

This project uses the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1,682 movies. The dataset includes:

- User ratings (1-5 stars)
- User demographic information (age, gender, occupation)
- Movie information (title, release date, genres)

The MovieLens dataset is a widely used benchmark dataset for recommendation systems research. It was collected by the GroupLens Research Project at the University of Minnesota. The data was collected through the MovieLens website (movielens.org) during a seven-month period from September 19th, 1997 to April 22nd, 1998.

### Data Format

- **ratings.data**: Contains user ratings in the format `user_id | movie_id | rating | timestamp`
- **users.data**: Contains user information in the format `user_id | age | gender | occupation | zip_code`
- **movies.data**: Contains movie information in the format `movie_id | title | release_date | video_release_date | IMDb_URL | genre1 | genre2 | ... | genre19`

The dataset is automatically downloaded and preprocessed when you run `python main.py --download`.

## Evaluation

The recommendation models are evaluated using the following metrics:

- **RMSE (Root Mean Squared Error)**: Measures the accuracy of rating predictions by calculating the square root of the average squared differences between predicted and actual ratings. Lower values indicate better performance.
- **MAE (Mean Absolute Error)**: Measures the average magnitude of errors in predictions without considering their direction. Lower values indicate better performance.
- **Precision@k**: The fraction of recommended items that are relevant to the user. It answers the question: "Of the items we recommended, how many did the user actually like?"
- **Recall@k**: The fraction of relevant items that are recommended. It answers the question: "Of all the items the user likes, how many did we recommend?"
- **F1@k**: The harmonic mean of precision and recall, providing a balance between the two metrics.

### Comparative Results

When evaluating on the MovieLens dataset, the hybrid model typically outperforms individual models:

| Model | RMSE | MAE | Precision@10 | Recall@10 | F1@10 |
|-------|------|-----|-------------|-----------|-------|
| User-based CF | ~1.02 | ~0.82 | ~0.45 | ~0.12 | ~0.19 |
| Item-based CF | ~0.98 | ~0.78 | ~0.48 | ~0.14 | ~0.22 |
| Content-based | ~1.10 | ~0.88 | ~0.40 | ~0.10 | ~0.16 |
| Hybrid | ~0.94 | ~0.75 | ~0.52 | ~0.15 | ~0.23 |

*Note: Actual results may vary depending on the specific data split and parameter settings.*

## Implementation Details

### Collaborative Filtering

The collaborative filtering components use the following techniques:

- **User-based CF**: Uses k-nearest neighbors to find similar users based on their rating patterns. The similarity is calculated using cosine similarity between user rating vectors.
- **Item-based CF**: Computes item-item similarity matrix using cosine similarity between item rating vectors. Predictions are made by weighted averaging of the user's ratings for similar items.

### Content-based Filtering

The content-based filtering component:

- Extracts movie features from genre information (represented as binary vectors)
- Builds user profiles based on the genres of movies they've rated highly
- Computes similarity between user profiles and movie features to make recommendations

### Hybrid Model

The hybrid model combines predictions from all three models using a weighted average approach:

```python
hybrid_pred = (cf_weight * cf_pred + cb_weight * cb_pred)
```

Where `cf_pred` is itself a weighted combination of user-based and item-based CF:

```python
cf_pred = (user_cf_weight * user_cf_pred + item_cf_weight * item_cf_pred)
```

The weights are configurable parameters that can be tuned for optimal performance.

## Web Interface

The web interface provides the following features:

- User selection and model selection
- Recommendation generation with configurable number of recommendations
- User profile viewing with demographic information and top-rated movies
- Movie details with genre information and similar movies
- Visual indicators for predicted ratings

## Future Improvements

Potential enhancements for the system include:

- Incorporating more advanced algorithms like matrix factorization or neural networks
- Adding more features to the content-based component (actors, directors, plot keywords)
- Implementing A/B testing capabilities to compare different recommendation strategies
- Adding real-time user feedback to improve recommendations over time
- Scaling the system to handle larger datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"# Hybrid-Recommendation-System" 
