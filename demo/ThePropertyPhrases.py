import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder


class ThePropertyPhrasesClassifier(object):
    '''
    Классификатор объектов недвижимости по отзывам
    '''
    def __init__(self):
        self.mlb = joblib.load("mlb_dump.pkl")
        self.pca = joblib.load("pca_dump.pkl")
        self.ovrc = joblib.load("ovrc_dump.pkl")
        self.phrases = pd.read_csv('phrases.csv', index_col='id')
        
        self.num_cols = ['latitude',
                         'longitude',
                         'accommodates',
                         'bathrooms',
                         'bedrooms',
                         'beds',
                         'square_feet',
                         'security_deposit',
                         'cleaning_fee',
                         'guests_included',
                         'extra_people',
                         'minimum_nights',
                         'price']
        
        self.cat_cols = ['experiences_offered',
                         'host_response_time',
                         'host_is_superhost',
                         'host_has_profile_pic',
                         'host_identity_verified',
                         'neighbourhood_cleansed',
                         'is_location_exact',
                         'property_type',
                         'room_type',
                         'bed_type',
                         'cancellation_policy',
                         'require_guest_phone_verification']
        
        self.tresholds = [0.3, 0.2, 0.1, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.2,
                          0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2,
                          0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


    def get_feature_names(self):
        return ['experiences_offered',
                 'host_response_time',
                 'host_is_superhost',
                 'host_has_profile_pic',
                 'host_identity_verified',
                 'neighbourhood_cleansed',
                 'latitude',
                 'longitude',
                 'is_location_exact',
                 'property_type',
                 'room_type',
                 'accommodates',
                 'bathrooms',
                 'bedrooms',
                 'beds',
                 'bed_type',
                 'amenities',
                 'square_feet',
                 'security_deposit',
                 'cleaning_fee',
                 'guests_included',
                 'extra_people',
                 'minimum_nights',
                 'cancellation_policy',
                 'require_guest_phone_verification',
                 'price']
        
        
    def preprocess(self, df):
            df_amenities = df['amenities'] \
                .apply(lambda x: str(x).strip('{}')) \
                .apply(lambda x: str(x).replace('to shower, toilet', 'to shower and toilet')) \
                .apply(lambda x: str(x).split(','))
            
            df_amenities = pd.DataFrame(self.mlb.transform(df_amenities),
                                        columns=self.mlb.classes_)
            
            df_pca = self.pca.transform(df_amenities)
            pca_cols = [f'pca{i:02}' for i in range(df_pca.shape[1])]
            i = 0
            
            df = df[self.num_cols + self.cat_cols].copy()
            for col in pca_cols:
                df[col] = df_pca[:,i]
                i += 1

            for col in self.cat_cols:
                le = LabelEncoder()
                le.fit(df[col].fillna(''))
                df[col] = le.transform(df[col].fillna(''))

            df.fillna(0, inplace=True)
            df['price'] = np.log1p(df['price'])

            return df

        
    def predict(self, df):
        pred = self.ovrc.predict_proba(df) 
        for n in range(pred.shape[1]):
            pred[:, n] = pred[:, n] > self.tresholds[n]
        return pred


class ThePropertyPhrasesGenerator(object):
    def __init__(self):
        self.topics_count = 32
        self.clf = ThePropertyPhrasesClassifier()
        self.phrases = pd.read_csv('phrases.csv', index_col='id')

        
    def get_feature_names(self):
        return self.clf.get_feature_names()
    
    
    def generate_key_phrases(self, data, n_phrases=20):
        cols = self.get_feature_names()
        ds = pd.Series([np.nan] * len(cols), index=cols)
        
        for col in cols:
            if col in data.keys():
                ds[col] = data[col]

        df = self.clf.preprocess(ds.to_frame().T)
        
        pred = self.clf.predict(df)
        
        topics = pd.Series([np.nan] * self.topics_count).to_frame()
        for i in range(self.topics_count):
            if pred[0,i] == 1:
                topics.loc[i,0] = i
        #return pred[0].T
        topics.rename(columns={0: 'topic'}, inplace=True)
        topics.dropna(inplace=True)
        topics['topic'] = topics['topic'].astype(int)
        
        phrases = topics.merge(self.phrases, how='inner', on='topic')
        if phrases.shape[0] == 0:
            phrases = self.phrases.copy()
        else:
            phrases.drop(index=phrases[phrases.topic==0].index, inplace=True)

        return phrases \
            .sort_values(by=['freq','rented_mean','listing_count'],
                         ascending=[False,False,True]) \
            .head(n_phrases)[['topic','phrases','freq','listing_count','rented_mean']]

        # return phrases \
        #     .sort_values(by=['freq','rented_mean','listing_count'],
        #                  ascending=[False,False,True]) \
        #     .head(n_phrases)[['topic','phrases','freq','listing_count','rented_mean']]

