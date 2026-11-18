# Movie Recommendation System
https://cinematch-movie-recommendation-najcygdvcm4wygadrrsac4.streamlit.app/
## Project Overview

The Movie Recommendation System recommends movies to users based on
movie ratings, genre similarity, and collaborative filtering.

The project uses the GroupLens MovieLens 100K dataset and provides
an interactive dashboard built with Streamlit.

## Features

1. **Popular Movies:** Ranks movies using average ratings, rating counts,
   and a weighted scoring method.
2. **Similar Movies:** Recommends movies based on genre similarity
   using cosine similarity.
3. **Personalized Recommendations:** Uses collaborative filtering
   based on similarities between users' rating patterns.
4. **Model Evaluation:** Displays evaluation results using MAE and RMSE
   and compares the recommendation model with a movie-average baseline.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Google Colab

## Dataset

**MovieLens 100K Dataset — GroupLens Research**

The dataset contains 100,000 movie ratings from 943 users
across 1,682 movies.

Dataset source:
https://grouplens.org/datasets/movielens/100k/

## Project Structure

Movie_Recommendation_System/
├── app.py
├── README.md
├── data/
│   └── ml-100k/
├── outputs/
│   ├── popular_movie_recommendations.csv
│   ├── content_based_recommendations.csv
│   ├── collaborative_filtering_recommendations.csv
│   └── model_evaluation.csv

## How to Run the Application

1. Install Python on your computer.
2. Install the required libraries:

   pip install pandas numpy scikit-learn streamlit

3. Ensure the MovieLens dataset is available at data/ml-100k/.
4. Ensure the model evaluation file is available in outputs/.
5. Open a terminal in the project directory.
6. Run:

   streamlit run app.py

7. Open the local URL displayed in the terminal.

## Recommendation Methods

### Popularity-Based Recommendation
Uses movie ratings and rating counts to rank popular movies.

### Content-Based Recommendation
Uses movie genre information and cosine similarity to identify
movies with similar genres.

### Collaborative Filtering
Uses similarities between users' movie-rating patterns to recommend
movies that a user may enjoy.

## Evaluation

The project evaluates rating predictions using Mean Absolute Error
(MAE) and Root Mean Squared Error (RMSE). It also compares the model
with a movie-average baseline.

Refer to outputs/model_evaluation.csv for the saved evaluation results.

## Project Status

The Streamlit dashboard has been tested, and all four tabs are working.

## Author

Parinitha
