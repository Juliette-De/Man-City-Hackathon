import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

minute_to_filter = st.slider('Match Time (minutes)', 45, 90, 45)  # min: 40, max: 90, default: 45

goal_diff_to_filter = st.slider('Goal Differential', -2, 2, 0) 

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

team_to_filter = st.selectbox(
    "Opponent Team",
    ("Arsenal WFC", "Leicester City WFC", "Aston Villa", "Tottenham Hotspur Women", "Liverpool WFC", "Brighton & Hove Albion WFC"),
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

def display_player(player):
    st.markdown("""
    #### <p height:200px><center>""" + player['player_name'] + '</center></p>', unsafe_allow_html=True)
    st.markdown("""
    #### <p height:100px><center>""" + player['team'] + '</center></p>', unsafe_allow_html=True)
    st.markdown("""
    #### <p height:100px><center>""" + player['player_out_position'] + '</center></p>', unsafe_allow_html=True)

def load_predictions():
    with open('./StatsBomb/Data/predictions.csv') as data_file:    
        predictions = pd.read_csv(data_file)  
    return predictions

def get_high_risk_players(team: str, minutes: int, goal_diff: int, predictions: pd.DataFrame):
    filtered_predictions = predictions.loc[(predictions['team_name'].str.contains(team)) & (predictions['time'] >= minutes) & (np.abs(predictions['goal_diff'] - goal_diff) < 1.5)]
    return filtered_predictions[["player_name", "team_name", "player_out_position", 'sub_risk']].sort_values(['sub_risk'], ascending=False).drop_duplicates()[:3]

st.markdown('***')
st.markdown('## Substitute Predictions')

predictions = load_predictions() 

players = get_high_risk_players(team_to_filter, minute_to_filter, goal_diff_to_filter, predictions)

for i in np.arange((len(players)//3)+1):
    with st.container():
        c = st.columns(3)
        for j in np.arange(min(len(players)-3*i, 3)):
            with c[j]:
                display_player(players.iloc[3*i+j])