import pandas as pd
import re
import requests
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

# Chargement des données
# Ce fichier contient les informations sur les films (titre, genres, etc.)
movies = pd.read_csv("/Users/ousmanediallo/PycharmProjects/Ai_projet/ai/movies.csv")
# Ce fichier contient les notes attribuées par les utilisateurs aux films
ratings = pd.read_csv("/Users/ousmanediallo/PycharmProjects/Ai_projet/ai/ratings.csv")

# Prétraitement des genres des films
# On transforme les genres en une liste
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# On applique une binarisation des genres pour les transformer en une matrice exploitable
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies['movieId'])

# Calcul de la similarité entre films en se basant sur les genres
content_similarity = cosine_similarity(genre_df)
content_similarity_df = pd.DataFrame(content_similarity, index=movies['movieId'], columns=movies['movieId'])

# Création d'une matrice utilisateur-film
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Calcul de la similarité entre utilisateurs
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)


# Fonction pour nettoyer les titres de films
def clean_movie_title(title):
    """Nettoyer les titres en supprimant ', The' et l'année de sortie."""
    title = re.sub(r'\s*\(\d{4}\)$', '', title)  # Supprimer l'année
    match = re.match(r'^(.*), (The|A|An)$', title)
    return f"{match.group(2)} {match.group(1)}" if match else title


# Fonction pour récupérer l'affiche d'un film
def get_movie_poster(title):
    """Requêter l'API OMDb pour récupérer l'affiche du film."""
    api_key = "1d34d942"
    cleaned_title = clean_movie_title(title)
    url = f"http://www.omdbapi.com/?t={cleaned_title}&apikey={api_key}"
    response = requests.get(url).json()
    return response["Poster"] if "Poster" in response and response[
        "Poster"] != "N/A" else "https://via.placeholder.com/150"


# Système de recommandation hybride
def HybridRecommender(user_id, top_n=4):
    """Recommande des films en combinant filtrage collaboratif et basé sur le contenu."""
    similar_users = user_similarity_df[user_id].drop(user_id)
    similar_users = similar_users[similar_users > 0]
    if similar_users.empty:
        return []

    movie_scores = {}
    for similar_user, similarity in similar_users.items():
        rated_movies = user_movie_matrix.loc[similar_user][user_movie_matrix.loc[similar_user] > 0]
        for movie, rating in rated_movies.items():
            movie_scores[movie] = movie_scores.get(movie, 0) + rating * similarity

    movie_scores = pd.Series(movie_scores).sort_values(ascending=False)
    user_watched_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    recommended_movie_ids = [movie for movie in movie_scores.index if movie not in user_watched_movies][:top_n]

    content_scores = {}
    for movie_id in recommended_movie_ids:
        similar_movies = content_similarity_df[movie_id].sort_values(ascending=False)
        similar_movies = similar_movies[similar_movies.index != movie_id].head(top_n)
        for similar_movie, similarity in similar_movies.items():
            if similar_movie not in recommended_movie_ids:
                content_scores[similar_movie] = content_scores.get(similar_movie, 0) + similarity

    hybrid_scores = pd.Series({**movie_scores.to_dict(), **content_scores}).sort_values(ascending=False)
    recommended_movies = movies[movies['movieId'].isin(hybrid_scores.index[:top_n])][['title', 'movieId']]

    return [(row['title'], get_movie_poster(row['title'])) for _, row in recommended_movies.iterrows()]


# Interface utilisateur avec Streamlit
st.title("🎬 Movie Recommendation System")
user_id = st.number_input("Entrez votre ID utilisateur :", min_value=1, step=1)
if st.button("Recommander"):
    with st.spinner("Génération des recommandations..."):
        recommendations = HybridRecommender(user_id)
    if recommendations:
        st.subheader("📌 Films recommandés :")
        for title, poster_url in recommendations:
            st.image(poster_url, caption=title, width=200)
    else:
        st.warning("Aucune recommandation disponible pour cet utilisateur .")

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


# Fonction pour évaluer la performance
def evaluate_performance(user_id, recommended_movies, top_n=4):
    """Évalue la performance du système de recommandation en utilisant les métriques de précision, rappel et F1-score."""

    # Charger les évaluations réelles de l'utilisateur (ensemble de test)
    test_movies = ratings[ratings['userId'] == user_id]
    test_movies = test_movies[
        test_movies['rating'] >= 4.0]  # Considérer les films que l'utilisateur a aimés (note >= 4)

    # Films recommandés par le modèle (avec ID du film)
    recommended_movie_ids = [movies[movies['title'] == title]['movieId'].iloc[0] for title, _ in recommended_movies]

    # Films réels que l'utilisateur a appréciés
    actual_movie_ids = test_movies['movieId'].tolist()

    # Calcul de la précision, du rappel et du F1-score
    # Transformer les listes en vecteurs binaires pour les calculs
    y_true = [1 if movie_id in actual_movie_ids else 0 for movie_id in recommended_movie_ids]
    y_pred = [1] * len(recommended_movie_ids)  # Tous les films recommandés sont considérés comme "prédits positifs"

    # Calcul des métriques
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


# Exemple d'utilisation : évaluer pour l'ID utilisateur 1
user_id = 1
recommended_movies = HybridRecommender(user_id)

# Évaluation de la performance
precision, recall, f1 = evaluate_performance(user_id, recommended_movies)

# Affichage des résultats
st.subheader("📊 Performance du modèle")
st.markdown(f"**Précision** : {precision:.2f}")
st.markdown(f"**Rappel** : {recall:.2f}")
st.markdown(f"**F1-score** : {f1:.2f}")