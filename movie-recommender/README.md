# Next Reel – Movie Recommendation System

A small full-stack machine learning project.

- **ML model:** content-based filtering with TF-IDF + cosine similarity (scikit-learn)
- **Backend:** Flask REST API
- **Frontend:** plain HTML/CSS/JavaScript, served by Flask
- **Data:** 65 movies in `data/movies.csv` (title, year, genres, keywords, overview, rating)

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

To test the model alone: `python recommender.py`

## How the model works

1. Each movie gets two TF-IDF vectors: one from its **genres**, one from its **keywords + overview**.
2. The two are stacked with weights (genres 60%, text 40%), so a shared genre counts for more than one shared word.
3. Your picks are averaged into a single **taste profile** vector.
4. Cosine similarity between your profile and every movie gives a score.
5. The score is nudged up slightly by the movie's rating (`score = similarity × (1 + 0.25 × rating)`), so among similar films the better-rated one wins.
6. Movies you already picked are removed. The "match %" shown is relative to the best result.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/search?q=incep` | Title search / autocomplete |
| GET | `/api/browse?genre=Comedy&limit=12` | Top-rated movies |
| GET | `/api/genres` | List of genres |
| POST | `/api/recommend` | Body: `{"liked":[6,7], "n":9, "genre":"Sci-Fi"}` |

## Ideas to extend

- Replace `movies.csv` with the MovieLens or TMDB 5000 dataset.
- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`).
- Add collaborative filtering once you have user ratings.
- Add real posters from the TMDB API.
