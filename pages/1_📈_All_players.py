import streamlit as st
import numpy as np

from PIL import Image

#import pandas as pd

from load_data import events, total, position, explanation
from functions import stats_match, stats_player, highlight



st.set_page_config(layout="wide")

minute_to_filter = st.slider('Manchester City - Aston Villa minute of the match', 40, 90, 45)  # min: 40, max: 90, default: 45

st.markdown('*<p style="text-align:right; font-size:smaller">(for demonstration purposes only: the track bar simulates the course of the WSL Manchester City - Aston Villa (1-1) match of January 21, 2023)</p>*',
           unsafe_allow_html=True)



st.title('All players')


all_players = stats_match(minute_to_filter)

# Add name and total stats
    
#sub_colums = ['player.id', 'player_name', 'xg', 'shots', 'fouls_committed']
all_players = all_players.merge(total.rename(
    columns = {i: i+'_off' for i in ['player_name', 'obv', 'xg', 'shots', 'fouls_won', 'fouls_committed', 'interceptions']}), # already position_off
                                on='player.id',
                                how='left').sort_values('obv_off_match', ascending=True)



    
def plot_graph(player):
    fig, ax = plt.subplots()
    series_obv = events[(events['match_id'] == 3856030) &
                        (events['minute']<minute_to_filter) &
                        (events['player.id']==player)
                       ].groupby('minute')['obv_total_net'].mean().plot(xlabel = 'Minute',
                                                                                ylabel = 'Cumulative OBV',
                                                                                ax=ax,
                                                                        ylim=(-0.2, 0.7))# without goalkeeper
    return fig



goalkeepers = all_players[all_players['position'] == position['goalkeepers']] # position_off false for Chloé Kelly
defenders = all_players[all_players['position'].isin(position['defenders'])]
midfielders = all_players[all_players['position'].isin(position['midfielders'])]
forwards = all_players[all_players['position'].isin(position['forwards'])]                


def display_player(player):
    stats1 = stats_player(player)
    int_col = [value for value in ['Shots', 'Fouls Won', 'Fouls Committed', 'Interceptions'] if value in stats1.index]
    st.image(Image.open('pictures/' + str(player['player.id']) + '.png')) # width=300 slightly small, 400 too large
    #st.write(player['player_name_off'])
    st.markdown("""
    #### <p height:200px><center>""" + player['player_name_off'] + '</center></p>', unsafe_allow_html=True)
    #st.pyplot(plot_graph(player['player.id']))
    st.dataframe(stats1.style.format("{:.2f}").format(precision=0,
                                                      subset=(int_col, stats1.columns[0])
                                                     ).apply(highlight, m=minute_to_filter, axis=None),
                 use_container_width = True)


st.markdown('## Forwards')

for i in np.arange((len(forwards)//3)+1):
    with st.container():
        c = st.columns(3)
        for j in np.arange((len(forwards)-3*i)):
            with c[j]:
                display_player(forwards.iloc[3*i+j])
            

st.markdown('***')
st.markdown('## Midfielders')

for i in np.arange((len(midfielders)//3)+1):
    with st.container():
        c = st.columns(3)
        for j in np.arange(min(len(midfielders)-3*i, 3)):
            with c[j]:
                display_player(midfielders.iloc[3*i+j])

                
st.markdown('***')
st.markdown('## Defenders')

for i in np.arange((len(defenders)//3)+1):
    with st.container():
        c = st.columns(3)
        for j in np.arange(min(len(defenders)-3*i, 3)):
            with c[j]:
                display_player(defenders.iloc[3*i+j])


st.markdown('***')
st.markdown('## Goalkeeper')


with st.container(): # no more than 3 goalkeepers
    c = st.columns(3)
    for j in np.arange(min(len(goalkeepers), 3)):
        with c[j]:
            display_player(goalkeepers.iloc[j])
            
            
st.markdown("***")
st.markdown(explanation)