
# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    
    genre= []
    for a,b in enumerate(movies.genres):
        tokens=tokenize_string(movies.genres[a])
        #print(tokens)
        genre.append(tokens)
        #print(genre)
    movies['tokens'] = genre
    return movies
    pass

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns_:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    
    vocab = {}
    occurence = {}
    for tokens in movies['tokens']:
        for token in tokens:
            if token not in vocab.keys():
                vocab.setdefault(token,-1)
            if token not in occurence.keys():
                occurence.setdefault(token,1)
            else:
                occurence[token] += 1
     
    vocab_list = sorted(vocab.keys(), key= lambda x: x)
    for i,token in enumerate(vocab_list):
            vocab[token] = i
    #print('vocab=',sorted(vocab.items()))
    #print('occurence=',sorted(occurence.items()))
    
    matrix = []
    N = len(movies)
    for tokens in movies['tokens']:
        col = []
        row = []
        data = []
        max_k = Counter(tokens).most_common(1)[0][1]
        for token in tokens:
            row.append(0)
            col.append(vocab[token])
            
            tf = Counter(tokens)[token]
            
            tfidf = tf/max_k * math.log10(N/occurence[token])
            data.append(tfidf)
            
            #print('row = ',row)
            #print('col = ',col)
            #print('data = ',data)
            
        X = csr_matrix((data,(row,col)),shape=(1,len(vocab)),dtype = np.float64)
        matrix.append(X)
            
    movies['features'] = np.array(matrix)
    return(movies,vocab)


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
 
    csr1 = a.data * a.data
    X = math.sqrt(csr1.sum())
    csr2 = b.data * b.data
    Y =math.sqrt(csr2.sum())
   
    dot_AB = a.dot(b.transpose())
    numerator = dot_AB.sum()
    if (X!=0) | (Y!=0):
        cosine = numerator / X * Y
    else :
        cosine = 0
    #print(type (cosine))
    return(cosine)
    
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and iprint
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """


    predicted = list()
    for index, row in ratings_test.iterrows():
        test_user = row['userId']
        predict = row['movieId']
        a = movies[movies.movieId == predict]['features'].values[0]

        wtratingsum = 0.0
        cosinesum = 0.0
        ratings = list()
        for index2, train in ratings_train[ratings_train.userId == test_user].iterrows():
            b = movies[movies.movieId == train['movieId']]['features'].values[0]
            cosine = cosine_sim(a,b)
            if (cosine > 0):
                wtratingsum += train['rating'] * cosine
                cosinesum += cosine

        if (cosinesum == 0.0 and wtratingsum == 0.0):
            train = ratings_train[ratings_train.userId == test_user]
            allrating = train['rating']
            predicted.append(sum(allrating) / len(allrating))

        else:
            predicted.append(wtratingsum / cosinesum)

    predictedratings = np.array(predicted)
    return predictedratings



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions=make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
