import pickle
import pandas as pd 
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


st.set_page_config(page_title="WebSeries Recommendation Engine", page_icon="📺", layout="wide")

# Load data
@st.cache_data
def load_data():
    series = pickle.load(open('final_df.pkl', 'rb'))
    return pd.DataFrame(series)

final_df = load_data()
train_df = final_df.drop(columns=['poster_path', 'number_of_seasons', 'number_of_episodes', 'popularity', 'homepage', 'type', 'tagline'], axis=1)

# Prepare model
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(train_df)

IMAGE_PATH = 'http://image.tmdb.org/t/p/w500'

def format_title(title):
    """Format the title with proper capitalization."""
    return ' '.join(word.capitalize() for word in title.split())

def find_closest_match(user_input, series_list):
    """Find the closest match to the user input."""
    closest_match, score = process.extractOne(user_input, series_list)
    return closest_match if score >= 80 else None

def get_poster_path(path):
    """Get the full poster path."""
    if path.startswith('https://') or path.startswith('http://'):
        return path
    return IMAGE_PATH + path

def recommendation(user_input):
    """Generate recommendations based on user input."""
    formatted_input = format_title(user_input)
    closest_match = find_closest_match(formatted_input, train_df.index)
    
    if closest_match is None:
        st.error(f"No close match found for '{formatted_input}'. Please try another series name.")
        return []
    
    distances, indices = model.kneighbors(train_df.loc[closest_match,:].values.reshape(1,-1), n_neighbors=6)
    
    recommendations = []
    for i in indices[0]:
        series = final_df.iloc[i]
        recommendations.append({
            'name': series.name,
            'poster_path': get_poster_path(series['poster_path']),
            'seasons': series['number_of_seasons'],
            'episodes': series['number_of_episodes'],
            'popularity': series['popularity'],
            'homepage': series['homepage'],
            'type': series['type'],
            'tagline': series['tagline']
        })
    
    return recommendations

# Streamlit UI
st.title('WebSeries Recommendation Engine')
st.write("Discover your next favorite series!")

user_input = st.text_input("Enter the name of a series you like:", help="Type the name of a series and press Enter")

if user_input:
    recommendations = recommendation(user_input)
    
    if recommendations:
        st.success(f"Showing recommendations based on '{format_title(user_input)}'")
        
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                series = recommendations[i+1]  # Skip the first one as it's the input series
                st.image(series['poster_path'], use_column_width=True)
                st.subheader(series['name'])
                st.write(f"Seasons: {series['seasons']}")
                st.write(f"Episodes: {series['episodes']}")
                st.write(f"Popularity: {series['popularity']:.2f}")
                
                if series['homepage'] and series['homepage'] != '':
                    st.markdown(f"[Visit Homepage]({series['homepage']})")
                
                if series['type'] and series['type'] != '':
                    st.write(f"Type: {series['type']}")
                
                if series['tagline'] and series['tagline'] != '':
                    st.write(f"Tagline: {series['tagline']}")

st.sidebar.title("About")
st.sidebar.info(
    "This app uses a machine learning model to recommend web series "
    "based on your input. Enter the name of a series you like, and "
    "we'll suggest similar shows you might enjoy!"
)

st.sidebar.title("How to use")
st.sidebar.markdown(
    """
    1. Enter the name of a web series you enjoy in the text box.
    2. Press Enter or click outside the text box.
    3. View the recommended series along with their details.
    4. If no results are found, try entering a different series name.
    """
)