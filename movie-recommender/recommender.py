"""Content-based movie recommender (TF-IDF + cosine similarity)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "movies.csv"

# A movie's fingerprint has two blocks: genres and free text (keywords + overview).
# Genres get more say so one shared word ("mind", "los angeles") can't outvote them.
GENRE_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
# Final score = similarity * (1 + RATING_BOOST * normalised rating).
# Rating only re-orders similar movies; it can't rescue an unrelated one.
RATING_BOOST = 0.25


def _slug(text: str) -> str:
    return text.lower().replace("-", "")


class Recommender:
    def __init__(self, path: Path = DATA_PATH):
        self.df = pd.read_csv(path)
        self.df["genre_list"] = self.df["genres"].str.split("|")

        genre_text = self.df["genre_list"].apply(lambda g: " ".join(_slug(x) for x in g))
        text = self.df["keywords"] + " " + self.df["keywords"] + " " + self.df["overview"]

        self.genre_vec = TfidfVectorizer(token_pattern=r"[a-z]+")
        self.text_vec = TfidfVectorizer(stop_words="english")
        genre_m = self.genre_vec.fit_transform(genre_text)
        text_m = self.text_vec.fit_transform(text)

        # Scaling each (row-normalised) block by sqrt(weight) makes the cosine of
        # the stacked vectors a weighted blend of genre and text similarity.
        self.matrix = hstack(
            [genre_m * np.sqrt(GENRE_WEIGHT), text_m * np.sqrt(TEXT_WEIGHT)]
        ).tocsr()

        r = self.df["rating"]
        self.rating_norm = ((r - r.min()) / (r.max() - r.min())).to_numpy()
        self.id_to_idx = {int(m): i for i, m in enumerate(self.df["id"])}

    # ---------- helpers ----------
    def _movie(self, idx: int, **extra) -> dict:
        row = self.df.iloc[idx]
        return {
            "id": int(row["id"]),
            "title": row["title"],
            "year": int(row["year"]),
            "genres": row["genre_list"],
            "overview": row["overview"],
            "rating": float(row["rating"]),
            **extra,
        }

    def genres(self) -> list[str]:
        return sorted({g for gl in self.df["genre_list"] for g in gl})

    def search(self, query: str, limit: int = 8) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        titles = self.df["title"].str.lower()
        starts = [i for i, t in enumerate(titles) if t.startswith(q)]
        contains = [i for i, t in enumerate(titles) if q in t and i not in starts]
        return [self._movie(i) for i in (starts + contains)[:limit]]

    def browse(self, genre: str | None = None, limit: int = 12) -> list[dict]:
        df = self.df
        if genre:
            df = df[df["genre_list"].apply(lambda g: genre in g)]
        top = df.sort_values("rating", ascending=False).head(limit)
        return [self._movie(self.id_to_idx[int(i)]) for i in top["id"]]

    # ---------- the model ----------
    def recommend(self, liked_ids: list[int], n: int = 8, genre: str | None = None) -> list[dict]:
        liked_idx = [self.id_to_idx[i] for i in liked_ids if i in self.id_to_idx]
        if not liked_idx:
            return []

        # User profile = average fingerprint of everything they liked.
        profile = np.asarray(self.matrix[liked_idx].mean(axis=0))
        sim = cosine_similarity(profile, self.matrix).ravel()
        score = sim * (1 + RATING_BOOST * self.rating_norm)

        score[liked_idx] = -1  # never recommend what they already picked
        if genre:
            mask = self.df["genre_list"].apply(lambda g: genre in g).to_numpy()
            score[~mask] = -1

        liked_genres = {g for i in liked_idx for g in self.df.iloc[i]["genre_list"]}
        liked_kw = {k for i in liked_idx for k in self.df.iloc[i]["keywords"].split()}

        results = []
        top = score.max()
        for i in np.argsort(score)[::-1][:n]:
            if score[i] <= 0:
                break
            row = self.df.iloc[i]
            results.append(
                self._movie(
                    int(i),
                    match=round(float(score[i]) / top * 100),  # relative to the best pick
                    shared_genres=[g for g in row["genre_list"] if g in liked_genres],
                    shared_keywords=[k for k in row["keywords"].split() if k in liked_kw][:3],
                )
            )
        return results


if __name__ == "__main__":  # quick CLI demo: python recommender.py
    rec = Recommender()
    pick = rec.search("inception")[0]
    print(f"Because you liked {pick['title']}:")
    for m in rec.recommend([pick["id"]], n=5):
        print(f"  {m['title']} ({m['year']}) - {m['match']}% match")
