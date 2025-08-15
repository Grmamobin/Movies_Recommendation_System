import pickle
import gradio as gr
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler


API_KEY = "f5165b88f19a84ebf4d50c4788ff8fb3"
BASE_URL = "https://image.tmdb.org/t/p/w500"

# Task2 (Popularity)
popularity = pickle.load(open('./src/models/popularityRecommender.pkl', 'rb'))
weighted_score = pickle.load(open('./src/models/weight_average.pkl', 'rb'))

# Task3 (Conetent Based)
sig = pickle.load(open('./src/models/sig.pkl', 'rb'))
df = pickle.load(open('./src/models/df.pkl', 'rb'))

# Task4 (Collabrative Filtering)
model_knn = pickle.load(open('./src/models/knn_model.pkl', 'rb'))
cosine_df = pickle.load(open('./src/models/cosine_df.pkl', 'rb'))

# Task5 (Hybrid Model)
hybrid = pickle.load(open('./src/models/hybrid.pkl', 'rb'))
hybrid_df = pickle.load(open('./src/models/hybrid_df.pkl', 'rb'))
ratings_small = pd.read_csv('./data/ratings_small.csv')
ratings_matrix = ratings_small.pivot(index='userId', columns='movieId', values='rating').fillna(0)


def popularity_recommender(movie_title, top_k=5):
    if movie_title:
        filtered = popularity[popularity['title'].str.contains(movie_title, case=False, na=False)]
    else:
        filtered = popularity

    filtered = filtered.sort_values('score', ascending=False).head(top_k)

    titles = filtered['title'].tolist()
    overviews = filtered['overview'].tolist() if 'overview' in filtered.columns else [""]*len(filtered)

    poster_urls = []
    for movie_id in filtered['id']:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        response = requests.get(url)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            poster_urls.append(BASE_URL + poster_path)
        else:
            poster_urls.append("") 

    return titles, overviews, poster_urls


def content_recommender(movie_title, top_k=5):
    if movie_title not in df['title'].values:
        return ["Movie Not Found"]*top_k, [""]*top_k, ["https://via.placeholder.com/500x750?text=No+Image"]*top_k, [""]*top_k

    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(sig[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    movie_indices = [i[0] for i in sim_scores]

    titles = list(df['title'].iloc[movie_indices])
    overviews = list(df['overview'].iloc[movie_indices]) if 'overview' in df.columns else [""]*len(movie_indices)

    poster_urls = []
    explanations = []
    base = df.iloc[idx]

    for i in movie_indices:
        rec = df.iloc[i]

        movie_id = rec['id'] if 'id' in rec else None
        if movie_id:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            response = requests.get(url)
            data = response.json()
            poster_path = data.get('poster_path')
            poster_urls.append(BASE_URL + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image")
        else:
            poster_urls.append("https://via.placeholder.com/500x750?text=No+Image")

        shared_genres = set(base['genres'].split()) & set(rec['genres'].split())
        shared_cast = set(base['cast'].split()) & set(rec['cast'].split())

        parts = []
        if shared_genres:
            parts.append(f"shared genres: {', '.join(shared_genres)}")
        if shared_cast:
            parts.append(f"{len(shared_cast)} actors in common: {', '.join(shared_cast)}")

        explanations.append(f"Why Recommended? {', and '.join(parts)}." if parts else
                            "These movies are recommended based on similarity but have no direct genre or cast overlap.")

    return titles, overviews, poster_urls, explanations



def collaborative_recommender(movie_title, top_k=5):
    if movie_title not in cosine_df.index:
        return ["Movie Not Found"]*top_k, [""]*top_k, ["https://via.placeholder.com/500x750?text=No+Image"]*top_k, [""]*top_k

    distances, indices = model_knn.kneighbors(
        cosine_df.loc[movie_title].values.reshape(1, -1),
        n_neighbors=top_k + 1
    )

    movie_indices = indices.flatten()[1:] 
    titles, overviews, poster_urls, explanations = [], [], [], []

    base_row = df[df['title'] == movie_title].iloc[0]

    for idx in movie_indices:
        title = cosine_df.index[idx]
        row = df[df['title'] == title]

        titles.append(title)
        overviews.append(row['overview'].values[0] if not row.empty and 'overview' in row.columns else "")

        movie_id = row['id'].values[0] if not row.empty and 'id' in row.columns else None
        if movie_id:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            response = requests.get(url)
            data = response.json()
            poster_path = data.get('poster_path')
            poster_urls.append(BASE_URL + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image")
        else:
            poster_urls.append("https://via.placeholder.com/500x750?text=No+Image")

        rec_row = row.iloc[0] if not row.empty else None
        explanation_parts = []
        if rec_row is not None:
            shared_genres = set(base_row['genres'].split()) & set(rec_row['genres'].split())
            shared_cast = set(base_row['cast'].split()) & set(rec_row['cast'].split())
            if shared_genres:
                explanation_parts.append(f"shared genres: {', '.join(shared_genres)}")
            if shared_cast:
                explanation_parts.append(f"{len(shared_cast)} actors in common: {', '.join(shared_cast)}")

        if explanation_parts:
            explanations.append(f"Why Recommended? {', and '.join(explanation_parts)}.")
        else:
            explanations.append("Recommended based on user behavior similarity.")

    return titles, overviews, poster_urls, explanations



def hybrid_recommender(user_input, top_k=5):
    try:
        user_id = int(user_input)
    except:
        return [f"Invalid User ID"] * top_k, [""] * top_k, [""] * top_k

    if user_id not in ratings_matrix.index:
        return [f"User not found"] * top_k, [""] * top_k, [""] * top_k

    cb_sim = hybrid['cb_sim']
    movie_id_to_idx = hybrid['movie_id_to_idx']

    user_ratings = ratings_matrix.loc[user_id]
    user_rated_items = user_ratings.to_numpy().nonzero()[0]

    sCB = cb_sim[user_rated_items, :].mean(axis=0) if len(user_rated_items) > 0 else np.zeros(cb_sim.shape[0])

    user_mean = user_ratings[user_ratings > 0].mean() if np.any(user_ratings > 0) else 0
    sCF = user_ratings.copy()
    sCF[sCF == 0] = user_mean
    sCF_full = np.zeros(len(hybrid_df))
    for mid, score in sCF.items():
        if mid in movie_id_to_idx:
            idx = movie_id_to_idx[mid]
            sCF_full[idx] = score

    scaler = MinMaxScaler()
    sCF_norm = scaler.fit_transform(sCF_full.reshape(-1,1)).flatten()
    sCB_norm = scaler.fit_transform(sCB.reshape(-1,1)).flatten()

    alpha = 0.7 
    s_hyb = alpha * sCF_norm + (1 - alpha) * sCB_norm

    top_indices = s_hyb.argsort()[::-1][:top_k]

    recommended_titles = []
    recommended_overviews = []
    poster_urls = []

    for i in top_indices:
        recommended_titles.append(hybrid_df.loc[i, 'title'])
        recommended_overviews.append(hybrid_df.loc[i, 'overview'] if 'overview' in hybrid_df.columns else "")

        # fetch poster from TMDb API
        movie_id = hybrid_df.loc[i, 'id'] if 'id' in hybrid_df.columns else None
        if movie_id:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            response = requests.get(url)
            data = response.json()
            poster_path = data.get('poster_path')
            poster_urls.append(BASE_URL + poster_path if poster_path else "")
        else:
            poster_urls.append("")
    explanations = []
    return recommended_titles, recommended_overviews, poster_urls , explanations


def generate_explanation(base_title, recommended_title , ):
    base = df[df['title'] == base_title].iloc[0]
    rec = df[df['title'] == recommended_title].iloc[0]

    shared_genres = set(base['genres'].split()) & set(rec['genres'].split())
    shared_cast = set(base['cast'].split()) & set(rec['cast'].split())

    explanation_parts = []
    if shared_genres:
        explanation_parts.append(f"shared genres: {', '.join(shared_genres)}")
    if shared_cast:
        explanation_parts.append(f"{len(shared_cast)} actors in common: {', '.join(shared_cast)}")

    if explanation_parts:
        return f"Why Recommended? {', and '.join(explanation_parts)}."
    else:
        return "These movies are recommended based on similarity but have no direct genre or cast overlap."


def recommend_movies(user_input, model_type):
    if model_type == "Popularity":
        titles, overviews, poster_urls = popularity_recommender(user_input)
        explanations = ["Recommended based on popularity scores."]
    elif model_type == "Content-Based":
        titles, overviews, poster_urls, explanations = content_recommender(user_input)
    elif model_type == "Collaborative Filtering":
        titles, overviews, poster_urls , explanations= collaborative_recommender(user_input)
    elif model_type == "Hybrid":
        titles, overviews, poster_urls , explanations = hybrid_recommender(user_input)
    else:
        titles, overviews, poster_urls, explanations = (["No Recommendation"], [""], [""], [""])

    output_text = "\n".join(titles)
    output_explanation = "\n\n".join(explanations)
    output_overview = "\n\n".join(overviews)

    return output_text, output_explanation, output_overview, poster_urls 



with gr.Blocks() as demo:
    
    gr.Markdown("# Movie Recommender System 🎬 ")
    
    with gr.Row():
        textbox = gr.Textbox(
            lines=1,
            placeholder="Type here...",
            label="Movie Title Or UserID(Hybrid Model)",
            show_label=True
        )
        model_dropdown = gr.Dropdown(
            ["Popularity", "Content-Based", "Collaborative Filtering", "Hybrid"],
            value="Popularity",
            label="Select Recommender"
        )
    button = gr.Button("Recommend", variant="primary")
    
    output_text = gr.Textbox(label="Movie Titles", lines=5)
    output_explanation = gr.Textbox(label="Why Recommended?", lines=10)
    output_overview = gr.Textbox(label="Movie Description", lines=5)
    output_poster = output_poster = gr.Gallery(label="Movie Posters",show_label=True,elem_id=None,columns=5)
    
    button.click(
        fn=recommend_movies,
        inputs=[textbox, model_dropdown],
        outputs=[output_text, output_explanation, output_overview, output_poster]
    )

demo.launch(share=True)
