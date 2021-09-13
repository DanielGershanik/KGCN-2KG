import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'movie': {
                'item2id_path': 'data/movie/item_index2entity_id.txt',
                'kg_path': 'data/movie/kg.txt',
                'rating_path': 'data/movie/ratings.csv',
                'rating_sep': ',',
                'threshold': 4.0
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            },
            'custom_music_v4': {
                'user_kg_path': 'data/custom_music_v4/user_kg_v4.txt',
                'item_kg_path': 'data/custom_music_v4/item_kg_v4.txt',
                'rating_path': 'data/custom_music_v4/bi_v4.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            },
            'book_v1': {
                'user_kg_path': 'data/book_v1/user_kg_v1.txt',
                'item_kg_path': 'data/book_v1/item_kg_v1.txt',
                'rating_path': 'data/book_v1/bi_v1.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            },
             'book_v2': {
                'user_kg_path': 'data/book_v2/user_kg_v2.txt',
                'item_kg_path': 'data/book_v2/item_kg_v2.txt',
                'rating_path': 'data/book_v2/bi_v2.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            },
              'book_v3': {
                'user_kg_path': 'data/book_v3/user_kg_v3.txt',
                'item_kg_path': 'data/book_v3/item_kg_v3.txt',
                'rating_path': 'data/book_v3/bi_v3.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            },
              'book_v4': {
                'user_kg_path': 'data/book_v4/user_kg_v4.txt',
                'item_kg_path': 'data/book_v4/item_kg_v4.txt',
                'rating_path': 'data/book_v4/bi_v4.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            },
              'book_v5': {
                'user_kg_path': 'data/book_v5/user_kg_v5.txt',
                'item_kg_path': 'data/book_v5/item_kg_v5.txt',
                'rating_path': 'data/book_v5/bi_v5.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            },
              'book_v6': {
                'user_kg_path': 'data/book_v6/user_kg_v6.txt',
                'item_kg_path': 'data/book_v6/item_kg_v6.txt',
                'rating_path': 'data/book_v6/bi_v6.txt',
                'rating_sep': '\t',
                'threshold': 0.0
            }
        }
        self.data = data
        
        # df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv(self.cfg[data]['item_kg_path'], sep='\t', header=None, names=['head','relation','tail'], skiprows=1)
        df_user_kg = pd.read_csv(self.cfg[data]['user_kg_path'], sep='\t', header=None, names=['head','relation','tail'], skiprows=1)
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating'], skiprows=1)

        # df_rating = df_rating[:8000]

        df_rating.reset_index(inplace=True, drop=True)
        
        # self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        self.df_user_kg = df_user_kg
        
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()
        
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        ############################## DEBUG ##############################

        # user_encoder = pd.concat([self.df_rating['userID'].astype(int), self.df_user_kg['head'].astype(int), self.df_user_kg['tail'].astype(int)]).sort_values(ascending=True)
        # entity_encoder = pd.concat([self.df_rating['itemID'].astype(int), self.df_kg['head'].astype(int), self.df_kg['tail'].astype(int)]).sort_values(ascending=True)
        # relation_encoder = pd.concat([self.df_kg['relation'], self.df_user_kg['relation']]).sort_values(ascending=True)

        # self.user_encoder.fit(user_encoder)
        # self.entity_encoder.fit(entity_encoder)
        # self.relation_encoder.fit(relation_encoder)

        ##############################       ##############################

        # self.user_encoder.fit(self.df_rating['userID'])
        self.user_encoder.fit(pd.concat([self.df_rating['userID'].astype(str), self.df_user_kg['head'].astype(str), self.df_user_kg['tail'].astype(str)]))
        self.entity_encoder.fit(pd.concat([self.df_rating['itemID'].astype(str), self.df_kg['head'].astype(str), self.df_kg['tail'].astype(str)]))
        self.relation_encoder.fit(pd.concat([self.df_kg['relation'], self.df_user_kg['relation']]))
        
        # encode df_user_kg
        self.df_user_kg['head'] = self.user_encoder.transform(self.df_user_kg['head'].astype(str)) #.astype(str)
        self.df_user_kg['tail'] = self.user_encoder.transform(self.df_user_kg['tail'].astype(str)) #.astype(str)
        self.df_user_kg['relation'] = self.relation_encoder.transform(self.df_user_kg['relation'])

        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'].astype(str)) #.astype(str)
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'].astype(str)) #.astype(str)
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

        self.df_dataset = pd.DataFrame()

        self.df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'].astype(str)) #.astype(str)
        self.df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'].astype(str)) #.astype(str)
        self.df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)


    def _build_dataset(self, mode):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update

        # df_dataset = self.df_dataset
        
        # negative sampling
        df_dataset = self.df_dataset[self.df_dataset['label']==1]
        # df_dataset requires columns to have new entity ID
        user_list = []
        item_list = []
        label_list = []
        
        if mode == 'item':
            full_item_set = set(range(len(self.entity_encoder.classes_)))
            for user, group in df_dataset.groupby(['userID']):
                item_set = set(group['itemID'])
                negative_set = full_item_set - item_set
                negative_sampled = random.sample(negative_set, int(len(item_set))) 
                user_list.extend([user] * len(negative_sampled))
                item_list.extend(negative_sampled)
                label_list.extend([0] * len(negative_sampled))

        # User
        if mode == 'user':
            full_user_set = set(range(len(self.user_encoder.classes_)))
            for item, group in df_dataset.groupby(['itemID']):
                user_set = set(group['userID'])
                negative_set = full_user_set - user_set
                negative_sampled = random.sample(negative_set, int(len(user_set)))
                item_list.extend([item] * len(negative_sampled))
                user_list.extend(negative_sampled)
                label_list.extend([0] * len(negative_sampled))

        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        df_dataset = pd.concat([df_dataset, negative])
        
        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset
        
        
    def _construct_kg(self, df_kg):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(df_kg)):
            head = df_kg.iloc[i]['head']
            relation = df_kg.iloc[i]['relation']
            tail = df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_dataset(self, mode):
        if mode == 'user':
            return self._build_dataset(mode='user')
        return self._build_dataset(mode='item')

    def load_kg(self, mode):
        if mode == 'user':
          return self._construct_kg(self.df_user_kg)
        return self._construct_kg(self.df_kg)
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
