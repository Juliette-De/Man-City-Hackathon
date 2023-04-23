import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import streamlit as st
from PIL import Image

from load_data import fawsl, events, lineups, lineups_positions, matches, total, columns, categorical, ohe
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


## Adding On-Ball-Values (per match)

subs = subs.merge(events.groupby(['match_id', 'player.id'])['obv_total_net'].sum().reset_index().rename(
    columns={'obv_total_net': 'obv_off_match'}), # 11*2*6+40 = 172 sum
                  how='left')

subs = subs.merge(events.groupby(['match_id', 'player.id'])['obv_total_net'].sum().reset_index().rename(
    columns={'player.id':'substitution.replacement.id', 'obv_total_net': 'obv_in'}),
                  how='left')

subs['obv'] = subs['obv_in'] - subs['obv_off_match']



## Adding summed On-Ball-Values

subs = subs.merge(total.rename(columns={'obv_total_net': 'obv_off'}), how='left')
subs = subs.merge(total[['player.id', 'obv_total_net']].rename(columns={'obv_total_net': 'sum_obv_in',
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




### UI


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


col = ['Match', 'Avg last 5 games']

def highlight(x):

    df1 = pd.DataFrame(index=x.index, columns=x.columns) # background-color: 

    
    #rewrite values by boolean masks
    df1.loc[['Sum of On-Ball Values'], 'Match'] = np.where((x.loc[['Sum of On-Ball Values']]['Match'] > x.loc[['Sum of On-Ball Values']][col[1]]),
                                                           'color: green;',
                                                           'font-weight: bold; color: red;')
    
    for i in df1.index[1:]:
        
        if i in ['xG', 'Shots']:
            if x.loc[i,'Match']*(90/minute_to_filter) < x.loc[i, col[1]]:
                df1.loc[[i], 'Match'] = 'color: red'
            elif x.loc[i,'Match'] > x.loc[i, col[1]]:
                df1.loc[[i], 'Match'] = 'color: green'
        
        if i in ['Fouls Committed', '']:
            if x.loc[i, 'Match'] > x.loc[i, col[1]]:
                df1.loc[[i], 'Match'] = 'color: red'
            elif x.loc[i,'Match']*(90/minute_to_filter) < x.loc[i, col[1]]:
                df1.loc[[i], 'Match'] = 'color: green'
  
    
    return df1


import matplotlib.pyplot as plt

def plot_graph(player):
    fig, ax = plt.subplots()
    series_obv = events[(events['match_id'] == 3856030) &
                        (events['minute']<minute_to_filter) &
                        (events['player.id']==player)
                       ].groupby('minute')['obv_total_net'].fillna(0).cumsum()
    series_obv = series_obv.set_axis(np.arange(len(series_obv))).plot(xlabel = 'Minute',
                                                                      ylabel = 'Cumulative OBV',
                                                                      ax=ax)
    return fig



import matplotlib.pyplot as plt
for i in range(len(predicted_subs)):
    
    arrows()
    col1, col2 = st.columns(2)
    s = predicted_subs.loc[i]
    
    obv = [s['obv_off_match'], s['obv_off']]
    xg = [s['xg_off_match'], s['xg_off']]
    shots = [s['shots_off_match'], s['shots_off']]
    fouls_committed = [s['fouls_committed_off_match'], s['fouls_committed_off']]
    
    # 
    obv_in = [round(s['predicted_obv_in'], 2), s['sum_obv_in']]
    xg_in = ['', s['xg_in']]
    shots_in = ['', s['shots_in']]
    fouls_committed_in = ['', s['fouls_committed_in']]
    
    col_in = ['Expected', col[1]]
    
    if s['position_off'] == 1: # Goalkeeper
        stats1 = pd.DataFrame(np.array([obv]),
                             ['Sum of On-Ball Values'],
                             col)
        stats2 = pd.DataFrame(np.array([obv_in]),
                             ['Sum of On-Ball Values'],
                             col_in).astype({col_in[1]: 'float'})
        int_col = []
    
    elif s['position_off'] in np.arange(2,9): # Back # Add fouls won?
        stats1 = pd.DataFrame(np.array([obv, fouls_committed]),
                             ['Sum of On-Ball Values', 'Fouls Committed'],
                             col)
        stats2 = pd.DataFrame(np.array([obv_in, fouls_committed_in]),
                             ['Sum of On-Ball Values', 'Fouls Committed'],
                             col_in).astype({col_in[1]: 'float'})
        int_col = ['Fouls Committed']
    
    elif s['position_off'] in np.concatenate((np.arange(9,17),np.arange(18,21))): # Midfield
        stats1 = pd.DataFrame(np.array([obv, xg, shots, fouls_committed]),
                             ['Sum of On-Ball Values', 'xG', 'Shots', 'Fouls Committed'],
                             col)
        stats2 = pd.DataFrame(np.array([obv_in, xg_in, shots_in, fouls_committed_in]),
                             ['Sum of On-Ball Values', 'xG', 'Shots', 'Fouls Committed'],
                             col_in).astype({col_in[1]: 'float'})
        int_col = ['Shots', 'Fouls Committed']
        
    else: #s['position_off'].isin(np.concatenate(([17],np.arange(21, 26)))): # Forward
        stats1 = pd.DataFrame(np.array([obv, xg, shots, fouls_committed]),
                             ['Sum of On-Ball Values', 'xG', 'Shots', 'Fouls Committed'],
                             col)
        stats2 = pd.DataFrame(np.array([obv_in, xg_in, shots_in, fouls_committed_in]),
                             ['Sum of On-Ball Values', 'xG', 'Shots', 'Fouls Committed'],
                             col_in).astype({col_in[1]: 'float'})
        int_col = ['Shots', 'Fouls Committed']

    # Setup a DataFrame with corresponding hover values
    tooltips_df = pd.DataFrame(index = stats1.index, columns = stats1.columns)
    tooltips_df.iloc[0,0] = 'test'
    
    
    with col1:
        st.image(Image.open('pictures/' + str(s['player.id']) + '.png'))
        st.markdown('## <center>' + s['player_name_off'] + '</center>', unsafe_allow_html=True)  
        #st.pyplot(plot_graph(s['player.id']))
        st.dataframe(stats1.style.format("{:.2f}").format(precision=0,
                                                          subset=(int_col, 'Match')
                                                         ).apply(highlight,
                                                                 axis=None))
        # #.set_tooltips(tooltips_df))
     
    with col2:
        st.image(Image.open('pictures/' + str(s['substitution.replacement.id']) + '.png'))
        st.markdown('## <center>' + s['player_name_in'] + '</center>', unsafe_allow_html=True)
        st.dataframe(stats2.style.format({'Avg last 5 games': "{:.2f}"}))
        
    
    st.markdown('### <center> On-Ball Value Difference Prediction: :green[+' +
                str(round(s['predicted_obv'], 2)) + ']</center>',
                unsafe_allow_html=True)
        
    st.markdown("***")


st.markdown('On-Ball Value: *the net change in expected goal difference (change in likelihood of scoring - change in likelihood of conceding) over the next 2 possession chains as a result of the event.* \n\n'
           'Statistics in the "' + col[1] + '" column are given per 90 minutes.\n\n')