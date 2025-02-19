import streamlit as st
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from load_data import events



### Parameters

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

team_to_filter = st.selectbox("Game",
                              ("Arsenal WFC", "Leicester City WFC", "Aston Villa", "Tottenham Hotspur Women", "Liverpool WFC", "Brighton & Hove Albion WFC"),
                              index=2, # default: Aston Villa
                              label_visibility=st.session_state.visibility,
                              disabled=st.session_state.disabled)

minute_to_filter = st.slider('Match time (in minutes)', 45, 90, 45)  # min and default: 45, max: 90
st.markdown('*<p style="text-align:right; font-size:smaller">(for demonstration purposes only: the track bar simulates the course of the game)</p>*',
            unsafe_allow_html=True)

# Goal differential for the opponent
GD = int(events.loc[(events['team.name'] == team_to_filter) &
                      (events['minute'] <= minute_to_filter), 'GD'].iloc[-1])
GD_ManCity = -GD

if GD_ManCity>0:
    st.markdown('Goal diffential: +' + str(GD_ManCity))
else:
    st.markdown('Goal diffential: ' + str(GD))



### Predictions
              
def load_predictions():
    with open('./StatsBomb/Data/predictions.csv') as data_file:    
        predictions = pd.read_csv(data_file)  
    return predictions

def get_high_risk_players(team: str, minutes: int, goal_diff: int, predictions: pd.DataFrame):
    predictions['goal_diff_diff'] = np.abs(predictions['goal_diff'] - goal_diff)
    filtered_predictions = predictions.loc[(predictions['team_name'].str.contains(team)) &
                                           (predictions['time'] >= minutes)].sort_values(
        ['goal_diff_diff', 'sub_risk'],
        ascending=[True, False])
    return filtered_predictions[["player_name", "team_name", "player_out_position", 'sub_risk']].drop_duplicates(['player_name', 'team_name'])[:3]

predictions = load_predictions() 

players = get_high_risk_players(team_to_filter, minute_to_filter, GD, predictions)

players = players.replace({'Emma Stina Blackstenius': 'Stina Blackstenius',
                           'Emma Wilhelmina Koivisto': 'Emma Koivisto',
                           'Laura Madison Blindkilde Brown': 'Laura Blindkilde Brown',
                           'Jemma Elizabeth Purfield': 'Jemma Purfield',
                           'Carlotte Wubben-Moy': 'Lotte Wubben-Moy',
                           'Caitlin Jade Foord': 'Caitlin Foord'}).drop_duplicates()


### Outcome

def display_player(player):
    try:
        st.image(Image.open('pictures/' + player['player_name'] + '.png'), use_container_width=True)
    except IOError:
        st.image(Image.open('pictures/default.png'), use_container_width=True)
    #image_path = 'pictures/' + player['player_name'] + '.png'
    #if not os.path.isfile(image_path):
    st.markdown("""
    ### <p height:200px><center>""" + player['player_name'] + '</center></p>', unsafe_allow_html=True)
    st.markdown("""
    ##### <p height:100px><center>""" + player['player_out_position'] + '</center></p>', unsafe_allow_html=True)

st.markdown('***')
st.markdown('## Substitute Predictions')
c = st.columns(3)
for i in np.arange(len(players)):
    with c[i]:
        display_player(players.iloc[i])
