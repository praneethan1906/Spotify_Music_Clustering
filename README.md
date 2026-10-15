Here is project structure

Spotify_Music_Clustering/
│
├── data/
│   └── dataset.csv
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── recommendation_data.pkl
├── recommendation.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

I have downloaded the dataset from kaggle
then I used this music dataset to create clusters using K means clustering
then I used cosine similarity to find the similarity between the songs in cluster
finaly I used streamlit to deploy my model