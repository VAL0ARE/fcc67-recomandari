import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import pickle, json, os

# 1. Citim datele din folderul local (NU mai descarcă nimic)
base = os.path.join(os.path.dirname(__file__), "ml-100k")
ratings = pd.read_csv(
    os.path.join(base, "u.data"),
    sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
)
print(f"Date incarcate: {len(ratings)} ratings, {ratings.user_id.nunique()} useri")

# 2. Surprise citeste direct din DataFrame
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)

# 3. Split train/test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 4. Antrenare SVD cu 16 factori latenti
model = SVD(n_factors=16, lr_all=0.005, reg_all=0.1, random_state=42)
model.fit(trainset)

# 5. Evaluare
preds = model.test(testset)
rmse = accuracy.rmse(preds)
mae = accuracy.mae(preds)

# 6. Salvare
with open("metrics.json", "w") as f:
    json.dump({"rmse": rmse, "mae": mae, "n_factors": 16,
               "n_ratings": len(ratings)}, f)

pickle.dump(model, open("model.pkl", "wb"))
print(f"Model salvat. RMSE={rmse:.4f}, MAE={mae:.4f}")