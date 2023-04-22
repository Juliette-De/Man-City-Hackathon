import streamlit as st

st.title('Substitution recommender')



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_data import fawsl, events, lineups, lineups_positions, matches, obv, columns, categorical, ohe
from functions import build_test, preprocessing, predict_best_subs




### Features engineering


## Adding position of the player substitued in

subs = events[events['type.id']==19]

subs = subs.merge(lineups_positions[['match_id', 'counterpart_id', 'lineup.player_id', 'position_id']],
                  how='left',
                  left_on=['match_id', 'player.id', 'substitution.replacement.id'], 
                  right_on=['match_id', 'counterpart_id', 'lineup.player_id'])

subs = subs.rename(columns={'position.id': 'position_off',
                            'position_id': 'position_in'})

## Adding On-Ball-Values

subs = subs.merge(events.groupby(['match_id', 'player.id'])['obv_total_net'].sum().reset_index().rename(
    columns={'obv_total_net': 'obv_off'}), # 11*2*6+40 = 172 sum
                  how='left')

subs = subs.merge(events.groupby(['match_id', 'player.id'])['obv_total_net'].sum().reset_index().rename(
    columns={'player.id':'substitution.replacement.id', 'obv_total_net': 'obv_in'}),
                  how='left')

subs['obv'] = subs['obv_in'] - subs['obv_off']


## Adding summed On-Ball-Values

subs = subs.merge(obv.rename(columns={'obv_total_net': 'sum_obv_off'}), how='left')
subs = subs.merge(obv.rename(columns={'obv_total_net': 'sum_obv_in',
                                      'player.id':'substitution.replacement.id'}),
                  how='left')





### Model


train = subs[subs['match_id'] != matches['AstonVilla']] # Exclude AstonVilla game from the training dataset
X_train = train[columns]
Y_train = train['obv_in']



## One hot encoding

# get_dummies : not the same number of features in train and test data: https://stackoverflow.com/questions/67865253/valueerror-x-has-10-features-but-decisiontreeclassifier-is-expecting-11-featur

train_array_hot_encoded = ohe.fit_transform(X_train[categorical])
X_train = preprocessing(train_array_hot_encoded, X_train)
X_train.head()


model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
model.fit(X_train, Y_train)


minute_to_filter = st.slider('minute', 40, 90, 45)  # min: 0h, max: 23h, default: 17h

st.dataframe(predict_best_subs(model, minute_to_filter))