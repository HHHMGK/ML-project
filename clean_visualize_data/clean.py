import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

users = pd.read_csv('dataset/users.dat', sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
ratings = pd.read_csv('dataset/ratings.dat', engine='python',
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
movies_train = pd.read_csv('dataset/movies_train.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
movies_test = pd.read_csv('dataset/movies_test.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
movies_train['genre'] = movies_train.genre.str.split('|')
movies_test['genre'] = movies_test.genre.str.split('|')

users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')
ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')
print('Data loaded')

# article words (Eng, Fr, Ger, Sp)
words = ['a', 'an', 'the', 'el', 'il', 'le', 'la', 'les' ,'l\'', 'der', 'das']
def cleanTitle(title):
    loc = title.find('(')
    if loc != -1:
        title = title[:loc]
    title = title.strip()
    title = title.lower()
    if title.find(',') != -1:
        last_word = title.split(', ')[-1]
        if last_word in words:
            loc = title.rfind(',')
            title = last_word + ' ' + title[:loc]
    return title

movies_test['title'] = movies_test['title'].apply(cleanTitle)
movies_train['title'] = movies_train['title'].apply(cleanTitle)
print('Title cleaned')

movies_train, movies_val = train_test_split(movies_train, test_size=0.2, random_state=42)

# add img_path column
movies_train['img_path'] = movies_train.index.map(lambda x: 'dataset/images/' + str(x) + '.jpg')
movies_val['img_path'] = movies_val.index.map(lambda x: 'dataset/images/' + str(x) + '.jpg')
movies_test['img_path'] = movies_test.index.map(lambda x: 'dataset/images/' + str(x) + '.jpg')
print('Add img_path column done')

# clean 2 minor data userRating and users
ratings.drop(columns=['timestamp'], inplace=True)
# ratings.reset_index(drop=True, inplace=True)
users.drop(columns=['zip','occupation'], inplace=True)

# write to csv 
movies_train.to_csv('dataset/movies_train.csv')
movies_val.to_csv('dataset/movies_val.csv')
movies_test.to_csv('dataset/movies_test.csv')

ratings.to_csv('dataset/ratings.csv', index=False)
users.to_csv('dataset/users.csv')
print('Title cleaned and saved to csv')

# drop every row with no poster
list_img = os.listdir('dataset/images/')
list_img = [int(i.split('.')[0]) for i in list_img]
movies_train = movies_train[movies_train.index.isin(list_img)]
movies_val = movies_val[movies_val.index.isin(list_img)]
movies_test = movies_test[movies_test.index.isin(list_img)]
# movies_train = movies_train[movies_train.img_path.apply(lambda x: os.path.isfile(x))]
# movies_val = movies_val[movies_val.img_path.apply(lambda x: os.path.isfile(x))]
# movies_test = movies_test[movies_test.img_path.apply(lambda x: os.path.isfile(x))]
print('Drop rows with no poster done')
# write to csv
movies_train.to_csv('dataset/movies_train_dropped.csv')
movies_val.to_csv('dataset/movies_val_dropped.csv')
movies_test.to_csv('dataset/movies_test_dropped.csv')