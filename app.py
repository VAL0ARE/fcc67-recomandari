from flask import Flask, jsonify, render_template_string
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml-100k")
ratings = pd.read_csv(
    os.path.join(BASE, "u.data"),
    sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
)
titles_df = pd.read_csv(
    os.path.join(BASE, "u.item"), sep="|", encoding="latin-1",
    header=None, usecols=[0, 1], names=["movie_id", "title"]
)
TITLES = dict(zip(titles_df.movie_id, titles_df.title))

users = sorted(ratings.user_id.unique())
movies = sorted(ratings.movie_id.unique())
u2i = {u: i for i, u in enumerate(users)}
m2i = {m: i for i, m in enumerate(movies)}

R = np.zeros((len(users), len(movies)))
for _, row in ratings.iterrows():
    R[u2i[row.user_id], m2i[row.movie_id]] = row.rating

R_mean = R[R > 0].mean()
R_centered = R.copy()
R_centered[R > 0] -= R_mean

k = 16
U, sigma, Vt = svds(csr_matrix(R_centered), k=k)
R_pred = np.dot(np.dot(U, np.diag(sigma)), Vt) + R_mean

user_seen = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

HTML = """
<!doctype html><html><head><meta charset="utf-8">
<title>Recomandari filme</title>
<style>
body{font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;padding:0 20px;background:#fafafa}
h1{color:#333}
input,button{padding:10px;font-size:16px;margin:4px;border:1px solid #ccc;border-radius:4px}
button{background:#0066cc;color:white;cursor:pointer;border:none}
button:hover{background:#0055aa}
ul{list-style:none;padding:0}
li{margin:8px 0;padding:12px;background:white;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.score{color:#666;font-size:14px}
.err{color:#c00;background:#fee;padding:12px;border-radius:6px}
</style></head>
<body>
<h1>Recomandari filme</h1>
<p>Model: SVD matrix factorization, 16 factori latenti, MovieLens 100K</p>
<input id="uid" type="number" value="42" min="1" max="943">
<button onclick="go()">Recomanda</button>
<ul id="out"></ul>
<script>
async function go(){
  const uid = document.getElementById('uid').value;
  const out = document.getElementById('out');
  out.innerHTML = '<li>Se incarca...</li>';
  try {
    const r = await fetch('/recommend/'+uid);
    const data = await r.json();
    out.innerHTML = '';
    if(data.error){ out.innerHTML = '<li class="err">Eroare: '+data.error+'</li>'; return; }
    data.forEach((x,i) => out.innerHTML +=
      '<li><b>'+(i+1)+'. '+x.title+'</b><br><span class="score">scor prezis: '+x.score.toFixed(2)+'/5</span></li>');
  } catch(e) { out.innerHTML = '<li class="err">Eroare retea</li>'; }
}
</script></body></html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    try:
        if user_id < 1 or user_id > 943:
            return jsonify({"error": f"user_id {user_id} invalid (trebuie 1-943)"}), 400
        if user_id not in u2i:
            return jsonify({"error": f"user_id {user_id} nu exista in dataset"}), 404

        seen = user_seen.get(user_id, set())
        row = u2i[user_id]
        scored = []
        for mid in movies:
            if mid in seen:
                continue
            col = m2i[mid]
            scored.append((mid, float(R_pred[row, col])))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [{"movie_id": int(m),
                   "title": TITLES.get(m, f"Film {m}"),
                   "score": s}
                  for m, s in scored[:10]]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)