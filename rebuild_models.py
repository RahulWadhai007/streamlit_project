# rebuild_models.py
# Run this ONCE to rebuild sentences.pkl + embeddings.pkl

import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

CSV_PATH = "arxiv_data_210930-054931.csv"
SENTENCES_PKL = "models/sentences.pkl"
EMBEDDINGS_PKL = "models/embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"   # your model name

print("🔍 Loading CSV file...")
df = pd.read_csv(CSV_PATH)

if "titles" not in df.columns or "abstracts" not in df.columns:
    raise ValueError("CSV missing required 'titles' and 'abstracts' columns.")

titles = df["titles"].astype(str).tolist()
abstracts = df["abstracts"].astype(str).tolist()

# Combine title + abstract for better embeddings
combined_texts = [(t + " " + a).strip() for t, a in zip(titles, abstracts)]

print(f"📌 Loaded {len(titles)} titles.")

print("🧠 Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME)

print("⚙️ Generating embeddings... (This may take time)")
embeddings = model.encode(combined_texts, show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings)

print("💾 Saving sentences.pkl and embeddings.pkl...")
pickle.dump(titles, open(SENTENCES_PKL, "wb"))
pickle.dump(embeddings, open(EMBEDDINGS_PKL, "wb"))

print("🎉 Rebuilding complete!")
print(f"✔ sentences.pkl saved ({len(titles)} titles)")
print(f"✔ embeddings.pkl saved ({embeddings.shape})")
