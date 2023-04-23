import streamlit as st

st.title('Substitution recommender')



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    columns={'obv_total_net': 'obv_off'}), # 11*2*6+40 = 172 sum
                  how='left')

subs = subs.merge(events.groupby(['match_id', 'player.id'])['obv_total_net'].sum().reset_index().rename(
    columns={'player.id':'substitution.replacement.id', 'obv_total_net': 'obv_in'}),
                  how='left')

subs['obv'] = subs['obv_in'] - subs['obv_off']



## Adding summed On-Ball-Values

subs = subs.merge(total.rename(columns={'obv_total_net': 'sum_obv_off'}), how='left')
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


minute_to_filter = st.slider('minute', 40, 90, 45)  # min: 40, max: 90, default: 45


subs = predict_best_subs(model, minute_to_filter)


def arrows():
    empty, arrow1, arrow2 = st.columns([1.5, 1, 2])
    with empty:
        st.write("")
    with arrow1:
        st.image(Image.open('pictures/red.png'), width=100)
    with arrow2:
        st.image(Image.open('pictures/green.png'), width=100)


def highlight(x):

    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    
    #rewrite values by boolean masks
    df1['Match'] = np.where((x['Match'] > x['Last 5 matches']) & (x.index == 'Sum of On-Ball Values'),
                            'color: {}'.format('green'),
                            'color: {}'.format('red'))
    df1.loc['Fouls Committed'] = 'background-color:'
    return df1


for i in range(len(subs)):
    
    arrows()
    col1, col2 = st.columns(2)
    s = subs.loc[i]
    
    if s['position_off'] == 1: # Goalkeeper
        stats = pd.DataFrame({'Match':[s['obv_off']],
                              'Last 5 matches': [s['sum_obv_off']]},
                             ['Sum of On-Ball Values', 'test'])
    
    elif s['position_off'] in np.arange(2,9): # Back
        # Foul won
        stats = pd.DataFrame({'Match':[s['obv_off'], s['fouls_committed_off']],
                              'Last 5 matches': [s['sum_obv_off'], s['fouls_committed_off_total']]},
                             ['Sum of On-Ball Values', 'Fouls Committed'])
    
    elif s['position_off'] in np.concatenate((np.arange(9,17),np.arange(18,21))): # Midfield
        stats = pd.DataFrame({'Match':[s['obv_off'], s['shots_off'], s['fouls_committed_off']],
                              'Last 5 matches': [s['sum_obv_off'], s['shots_off_total'], s['fouls_committed_off_total']]},
                             ['Sum of On-Ball Values', 'Shots', 'Fouls Committed'])
        
    else: #s['position_off'].isin(np.concatenate(([17],np.arange(21, 26)))): # Forward
        stats = pd.DataFrame({'Match':[s['obv_off'], s['shots_off'], s['fouls_committed_off']],
                              'Last 5 matches': [s['sum_obv_off'], s['shots_off_total'], s['fouls_committed_off_total']]},
                             ['Sum of On-Ball Values', 'Shots', 'Fouls Committed'])

    
    with col1:
        st.image(Image.open('pictures/' + str(s['player.id']) + '.png'))
        st.markdown('## <center>' + s['player_name_off'] + '</center>', unsafe_allow_html=True)
        st.dataframe(stats.style.format(
            "{:.2f}").apply(highlight, axis=None))
     
    
    with col2:
        st.image(Image.open('pictures/' + str(s['substitution.replacement.id']) + '.png'))
        st.markdown('## <center>' + s['player_name_in'] + '</center>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({'Prediction':[round(s['predicted_obv_in'], 2), ''],
                               'Last 5 matches': [round(s['sum_obv_in'], 2), round(s['fouls_committed_in'], 2)]},
                              ['Sum of On-Ball Values', 'Fouls Committed']))#.style.format(
            #"{:.2f}"))
                 
                 
                 #.style.format(
            #"{:.2f}"))
            
            #"{:.2f}").format(na_rep='')
        
    
    st.markdown('### <center> On-Ball Value Difference Prediction: :green[+' +
                str(round(subs.loc[i, 'predicted_obv'], 2)) + ']</center>',
                unsafe_allow_html=True)
        
    st.markdown("***")


st.markdown('*On-Ball Value: the net change in expected goal difference (change in likelihood of scoring - change in likelihood of conceding) over the next 2 possession chains as a result of the event.*')