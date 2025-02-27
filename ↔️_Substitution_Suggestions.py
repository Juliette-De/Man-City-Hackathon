import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import streamlit as st
from PIL import Image

from load_data import fawsl, events, lineups, lineups_positions, matches, position, subs, columns, categorical, ohe, explanation
from functions import preprocessing, predict_best_subs, stats_player, highlight



### Model

## Prepare training dataset

train = subs[subs['match_id'] != matches['AstonVilla']] # Exclude AstonVilla game from the training dataset
X_train = train[columns]
Y_train = train['obv_in']


## One hot encoding

# get_dummies : not the same number of features in train and test data: https://stackoverflow.com/questions/67865253/valueerror-x-has-10-features-but-decisiontreeclassifier-is-expecting-11-featur
train_array_hot_encoded = ohe.fit_transform(X_train[categorical])
X_train = preprocessing(train_array_hot_encoded, X_train)

model = make_pipeline(StandardScaler(with_mean=False),
                      LinearRegression())
model.fit(X_train, Y_train)




### UI


st.set_page_config(page_title="Substitution recommender", page_icon="⚽")

minute_to_filter = st.slider('Manchester City - Aston Villa minute of the match', 40, 90, 45)  # min: 40, max: 90, default: 45

st.markdown('*<p style="text-align:right; font-size:smaller">(for demonstration purposes only: the track bar simulates the course of the WSL Manchester City - Aston Villa (1-1) match of January 21, 2023)</p>*',
           unsafe_allow_html=True)

st.title('Substitution Suggestions')


predicted_subs = predict_best_subs(model, minute_to_filter)



def arrows():
    empty, arrow1, arrow2 = st.columns([1.5, 1, 2])
    with arrow1:
        st.image(Image.open('pictures/red.png'), width=100)
    with arrow2:
        st.image(Image.open('pictures/green.png'), width=100)
        


for i in range(len(predicted_subs)):
    
    arrows()
    col1, col2 = st.columns(2)
    s = predicted_subs.loc[i]
    
    obv_in = [round(s['predicted_obv_in'], 2), s['sum_obv_in']]
    xg_in = ['', s['xg_in']]
    shots_in = ['', s['shots_in']]
    pressures_in = ['', s['pressures_in']]
    fouls_won_in = ['', s['fouls_won_in']]
    fouls_committed_in = ['', s['fouls_committed_in']]
    passing_in = ['', s['passing_in']]
    interceptions_in = ['', s['interceptions_in']]
    
    # Stats for the player to be substitued off
    stats1 = stats_player(s)
    
    obv_row = stats1.index[0]
    col_in = ['Expected', stats1.columns[1]]
    
    # Stats for the player to be substitued in
    
    if s['position_off'] == position['goalkeepers']:
        stats2 = pd.DataFrame(np.array([obv_in]),
                             [obv_row],
                             col_in).astype({col_in[1]: 'float'})
    
    elif s['position_off'] in position['defenders']:
        stats2 = pd.DataFrame(np.array([obv_in,
                                        passing_in,
                                        pressures_in,
                                        interceptions_in,
                                        fouls_committed_in]),
                             [obv_row,
                              'Passing %',
                              'Pressures',
                              'Interceptions',
                              'Fouls Committed'],
                             col_in).astype({col_in[1]: 'float'})
    
    elif s['position_off'] in position['midfielders']:
        stats2 = pd.DataFrame(np.array([obv_in,
                                        xg_in,
                                        shots_in,
                                        pressures_in,
                                        fouls_won_in,
                                        fouls_committed_in,                              
                                        passing_in]),
                             [obv_row,
                              'xG',
                              'Shots',
                              'Pressures',
                              'Fouls Won',
                              'Fouls Committed',
                              'Passing %',],
                             col_in).astype({col_in[1]: 'float'})
        
    else: # Forward
        stats2 = pd.DataFrame(np.array([obv_in, xg_in, shots_in, pressures_in, fouls_won_in]),
                             [obv_row, 'xG', 'Shots', 'Pressures', 'Fouls Won'],
                             col_in).astype({col_in[1]: 'float'})
        
    int_col = [value for value in ['Shots', 'Pressures', 'Fouls Won', 'Fouls Committed', 'Interceptions'] if value in stats1.index]
    
    
    with col1:
        try:
            st.image(Image.open('pictures/' + str(s['player.id']) + '.png'), use_container_width=True)
        except IOError:
            st.image(Image.open('pictures/default.png'), use_container_width=True)
        st.markdown('## <center>' + s['player_name_off'] + '</center>', unsafe_allow_html=True)  
        st.dataframe(stats1.style.format("{:.2f}").format(precision=0, subset=(int_col, stats1.columns[0]) # Game column
                                                         ).apply(highlight, m=minute_to_filter, axis=None))
     
    with col2:
        try:
            st.image(Image.open('pictures/' + str(s['substitution.replacement.id']) + '.png'),
                     use_container_width=True)
        except IOError:
            st.image(Image.open('pictures/default.png'), use_container_width=True)
        st.markdown('## <center>' + s['player_name_in'] + '</center>', unsafe_allow_html=True)
        st.dataframe(stats2.style.format({'Avg last 5 games': "{:.2f}"}))
        
    
    st.markdown('### <center> On-Ball Value Difference Prediction: :green[+' +
                str(round(s['predicted_obv'], 2)) + ']</center>',
                unsafe_allow_html=True)
        
    st.markdown("***")


st.markdown(explanation)
