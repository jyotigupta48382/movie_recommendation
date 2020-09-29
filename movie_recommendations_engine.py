
# # Movie Recommdations


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
    return df[df.index==index]['title'].values[0]
def get_index_from_title(title):
    return df[df.title==title]['index'].values[0]


df=pd.read_csv('Desktop/movie_recommender/movie_dataset.csv')
df.head()
features=['keywords','cast','genres','director']

for feature in features:
    df[feature]=df[feature].fillna('')

def combine_features(row):
    return row['keywords']+ ' '+row['cast']+' '+row['genres']+' ' + row['director']

df['combined_features']=df.apply(combine_features,axis=1)
print('combined features',df['combined_features'].head())

cv=CountVectorizer()
count_matrix=cv.fit_transform(df['combined_features'])
cosine_sim=cosine_similarity(count_matrix)

user_like='John Carter'
movie_index=get_index_from_title(user_like)

similar_movie=list(enumerate(cosine_sim[movie_index]))
print(similar_movie)
sorted_similar_movies=sorted(similar_movie,key=lambda x:x[1], reverse=True)

l=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    l=l+1

    if l>50:
        break



