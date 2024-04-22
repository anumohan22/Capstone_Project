import streamlit as st # type: ignore
import pickle
import requests
import joblib

# Load Movie Recommender System data and model
movies = pickle.load(open('/Users/anusreemohanan/Downloads/Data Science/movie_list.pkl','rb'))
similarity = pickle.load(open('/Users/anusreemohanan/Downloads/Data Science/similarity.pkl','rb'))

# Load Sentiment Analysis model and vectorizer
model = joblib.load('/Users/anusreemohanan/Downloads/Data Science/logistic_regression_model.pkl')
vectorizer = joblib.load('/Users/anusreemohanan/Downloads/Data Science/tfidf_vectorizer.pkl')


api_key = '7052602f118b896558544cd00fa597a8'
# Function to get movie posters
def get_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path', None)
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        return None  # Return None if no poster available
    except Exception as e:
        st.error(f'Error fetching poster: {e}')
        return None

# Function to Find Recommended Movies
def recommendation(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = []
        recommended_movie_posters = []
        for i in distances[1:6]:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movie_posters.append(get_poster(movie_id))
            recommended_movie_names.append(movies.iloc[i[0]].title)
        return recommended_movie_names, recommended_movie_posters
    except Exception as e:
        st.error(f'Error in recommendation: {e}')
        return [], []

# Function to preprocess text for sentiment analysis
def preprocess_text(text):
    return vectorizer.transform([text])

# Setup sidebar for navigation
st.sidebar.title('Navigation')
app_mode = st.sidebar.radio("Choose the application:", ["Movie Recommender", "Sentiment Analysis"])

if app_mode == "Movie Recommender":
    st.header('Movie Recommendation System')
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Type or Select a Movie from the Dropdown", movie_list)

    if st.button('Show Recommendations'):
        recommended_movie_names, recommended_movie_posters = recommendation(selected_movie)
        if recommended_movie_names:
            cols = st.columns(5)
            for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
                with col:
                    st.text(name)
                    if poster:  
                        st.image(poster)
                    else:
                        st.write("No poster available")

elif app_mode == "Sentiment Analysis":
    st.header("Sentiment Analysis for Reviews")
    review_text = st.text_area("Enter Review:", "Type Review...")

    if st.button("Analyze"):
        try:
            transformed_review = preprocess_text(review_text)
            prediction = model.predict(transformed_review)
            prediction_label = 'Positive' if prediction[0] == 1 else 'Negative'
            st.write(f"The predicted sentiment is: **{prediction_label}**")
        except Exception as e:
            st.error(f'Error in sentiment analysis: {e}')
