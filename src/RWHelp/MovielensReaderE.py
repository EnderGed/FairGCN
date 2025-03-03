import pandas as pd
import numpy as np

from RWHelp import DataReaderE
from consts import MOVIELK_PATH, MOVIELM_PATH


class MovielensReaderE(DataReaderE):
    def __init__(self, name='movielK', in_path=None):
        if in_path is None:
            if name == 'movielK':
                in_path = MOVIELK_PATH
            elif name == 'movielM':
                in_path = MOVIELM_PATH
            else:
                raise Exception("Unknown Movie Lens name {}.".format(name))

        super().__init__(name, in_path, add_to_path=False)

    def read(self):
        in_path = self.in_path
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings = pd.read_csv(in_path + 'u.data', sep='\t', names=r_cols, encoding='latin-1')
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv(in_path + 'u.user', sep='|', names=u_cols, encoding='latin-1', parse_dates=True)
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action',
                  'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                  'film-noir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']
        movies = pd.read_csv(in_path + 'u.item', sep='|', names=m_cols, encoding='latin-1', parse_dates=True)

        # convert ids to start from 0
        # users['user_id'] -= 1
        # movies['movie_id'] -= 1
        ratings['user_id'] -= 1
        ratings['movie_id'] -= 1

        # reformat ratings
        ratings.drop('unix_timestamp', axis=1, inplace=True)
        ratings.columns = ['user', 'item', 'label']

        # reformat users
        users = users.loc[:, ['age', 'sex']]
        users.sex = [int(s == 'M') for s in users.sex]

        # reformat movies
        movies = movies.iloc[:, 5:]

        return ratings, {'user': users, 'item': movies}