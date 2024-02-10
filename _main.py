import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')

movies_df = movies_df.merge(credits_df, on="title")

movies_df = movies_df[["movie_id", 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def convert(obj):
    res = []
    for i in ast.literal_eval(obj):
        res.append(i['name'])
    return res

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)

def convert3(obj):
    res = []
    couter = 0
    for i in ast.literal_eval(obj):
        if couter != 3:
            res.append(i['name'])
            couter += 1
    return res

movies_df['cast'] = movies_df['cast'].apply(convert3)

def fetch_director(obj):
    res = []

    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            res.append(i['name'])
            break
            
    return res

movies_df['crew'] = movies_df['crew'].apply(fetch_director)

movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['overview'] = movies_df['overview'].apply(lambda x : x.split())

movies_df['genres'] = movies_df['genres'].apply(lambda x : [i.replace(' ', '') for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [i.replace(' ', '') for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x : [i.replace(' ', '') for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x : [i.replace(' ', '') for i in x])

movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

new_df = movies_df[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x : ' '.join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    res = []

    for i in text.split():
        res.append(ps.stem(i))
    return ' '.join(res)

new_df['tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x : x[1])[1:6]

    print()

    for i in movie_list:
        print(new_df.iloc[i[0]].title)


recommend('Independence Day')
