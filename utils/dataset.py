import pandas as pd
import os, re
from nltk.tokenize import wordpunct_tokenize
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

    
def getDataLoader(dataset_dir, use_dropped_data=False, batch_size=32, img_size='256', NUM_WORKERS = os.cpu_count()):
    posterTransformer = getImgTransformer(size='256')
    movies_train, movies_test, movies_val, ratings, genres_list = load_data(dataset_dir, use_dropped_data)
    maxUserID = len(ratings.index)
    train = theDataset(movies_train, ratings, posterTransformer, genres_list, maxUserID)
    val = theDataset(movies_val, ratings, posterTransformer, genres_list, maxUserID)
    test = theDataset(movies_test, ratings, posterTransformer, genres_list, maxUserID)
    train.merge_vocab(val)
    train.merge_vocab(test)
    val.merge_vocab(train)
    test.merge_vocab(train)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    sizes = (train.vocab_size, maxUserID)
    return train_dataloader, val_dataloader, test_dataloader, sizes

def load_data(dataset_dir, use_dropped_data=False):
    print('Loading data... from ', dataset_dir)
    with open(dataset_dir + 'genres.txt', 'r') as f:
        genres_list = [g.replace('\n','') for g in f.readlines()]
    is_drop = ''
    if use_dropped_data:
        is_drop = '_dropped'
    movies_train = pd.read_csv(dataset_dir + 'movies_train' + is_drop + '.csv')
    movies_test = pd.read_csv(dataset_dir + 'movies_test' + is_drop + '.csv')
    movies_val = pd.read_csv(dataset_dir + 'movies_val' + is_drop + '.csv')
    ratings = pd.read_csv(dataset_dir + 'ratings.csv')
    return movies_train, movies_test, movies_val, ratings, genres_list

class theDataset(Dataset):
    def __init__(self, df, allRatingdf, posterTransformer, genres_list=None, maxUserID=6040):
        self.df = df

        # title process
        vocab = create_vocab(df, column='title')
        self.vocab_size = len(vocab)
        self.title2int = {word: i for i, word in enumerate(vocab)}

        # image process
        self.transformer = posterTransformer

        # rating process 
        self.user_ratings = {}
        for movie_id in df['movieid']:
            rating_for_current_movie = np.zeros(maxUserID)
            rated_users = allRatingdf.loc[allRatingdf['movieid'] == movie_id].userid.tolist()
            rated_v = allRatingdf['rating'].values
            for user in rated_users:
                rating_for_current_movie[user - 1] = int(rated_v[user])
            self.user_ratings[movie_id] = rating_for_current_movie
            
        # genres process
        if genres_list == None:
            genres_list = ['Crime', 'Thriller', 'Fantasy', 'Horror', 'Sci-Fi', 'Comedy', 'Documentary', 'Adventure', 'Film-Noir', 'Animation', 'Romance', 'Drama', 'Western', 'Musical', 'Action', 'Mystery', 'War', "Children's"]
        self.genre2int = {genre: i for i, genre in enumerate(genres_list)} 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        oneItem = self.df.iloc[idx]
        title = oneItem['title']
        title_vec = onehot_vectorize(title, self.title2int)
        img_path = oneItem['img_path']
        img = convert_img(img_path, self.transformer)
        movie_id = oneItem['movieid']
        rating = self.user_ratings[movie_id]
        genres = oneItem['genre']
        genres_vec = multihot_genres(genres, self.genre2int)
        return title_vec, img, rating, genres_vec
    
    def merge_vocab(self, other):
        self.title2int.update(other.title2int)
        self.vocab_size = len(self.title2int)


# Title processing function
TITLE_MAX_LEN = 15
pad_token = '<PAD>'
unk_token = '<UNK>'

def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    return tokens

def create_vocab(dataset, column='title'):
    df = dataset.copy()
    titles = df[column].tolist()
    vocab = set()
    for title in titles:
        tokens = tokenize(title)
        vocab.update(tokens)
    vocab = list(vocab)
    vocab.append(pad_token)
    vocab.append(unk_token)
    return vocab

def onehot_vectorize(title, title2int):
    tokens = tokenize(title)
    tokens = tokens[:TITLE_MAX_LEN]
    while len(tokens) < TITLE_MAX_LEN:
        tokens.append(pad_token)
    title_vec = np.zeros((TITLE_MAX_LEN,len(title2int)), dtype=np.float32)
    for i, token in enumerate(tokens):
        if token in title2int:
            title_vec[i][title2int[token]] = 1
        else:
            title_vec[i][title2int[unk_token]] = 1
    return title_vec
    
def multihot_genres(genres,  genres_dict):
    genres = genres.strip('][').replace("'", "").split(', ')
    multi_hot = np.zeros(len(genres_dict))
    for genre in genres:
        if genre in genres_dict:
                multi_hot[genres_dict[genre]] = 1
    return multi_hot


# Image processing function
IMAGE_SIZE={"16": (16, 16),
            "24": (24, 24),
            "32": (32, 32),
            "40": (40, 40),
            "64": (64, 64),
            "72": (72, 72),
            "128": (128, 128),
            "224": (224, 224),
            "256": (256, 256)}
def getImgTransformer(size = '256'):
    return transforms.Compose([
            transforms.Resize(size=IMAGE_SIZE[size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def convert_img(img_path, transformer):
    if not os.path.exists(img_path):
        last_slash = img_path.rfind('/')
        img_path = img_path[:last_slash] + '/0.jpg'
    img = Image.open(img_path)
    if len(img.getbands()) == 1: # check if the image have only one channel
        trans = transforms.Grayscale(num_output_channels=3)
        img = trans(img) # convert image to a three-channel image
    img = transformer(img)
    return img
