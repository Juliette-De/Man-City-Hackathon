### Loading data

import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


with open('StatsBomb/Data/FAWSL_22_23.json') as data_file:    
    data = json.load(data_file)  

fawsl = pd.json_normalize(data).sort_values('match_date')


events = pd.DataFrame()
lineups = pd.DataFrame()
lineups_positions = pd.DataFrame()
lineups_events = pd.DataFrame()


matches = {'Arsenal' : 3852832,
           'AstonVilla' : 3856030,
           'Brighton' : 3855980,
           'LeicesterCity' : 3855947,
           'Liverpool' : 3855961,
           'Tottenham' : 3856040}


for i in matches:
    
    with open('StatsBomb/Data/ManCity_' + i + '_events.json') as data_file:    
        data = json.load(data_file)
    normalized_data = pd.json_normalize(data).sort_values(['minute', 'second'])
    normalized_data['match_id'] = matches[i]
    events = pd.concat([events, normalized_data])

    
    with open('StatsBomb/Data/ManCity_' + i + '_lineups.json') as data_file:    
        data = json.load(data_file)
        
    # Unnesting lineups
    normalized_data = pd.json_normalize(data,
                                        record_path='lineup',
                                        meta = ['team_id', 'team_name'])
    normalized_data['match_id'] = matches[i]
    lineups = pd.concat([lineups, normalized_data])
        
    # Unnesting positions
    normalized_data = pd.json_normalize(data,
                                        record_path=['lineup', 'positions'],
                                        meta = ['team_id', 'team_name', ['lineup', 'player_id']])
    normalized_data['match_id'] = matches[i]
    lineups_positions = pd.concat([lineups_positions, normalized_data])
    
    # Unnesting events
    normalized_data = pd.json_normalize(data, record_path=['events'], meta = ['team_id', 'team_name'])
    normalized_data['match_id'] = matches[i]
    lineups_events = pd.concat([lineups_events, normalized_data])
    
    # Unnesting formations: pd.json_normalize(data, record_path=['formations'], meta = ['team_id', 'team_name',])



# Convert column from object to int

lineups_positions = lineups_positions.astype({'lineup.player_id': 'int'})



## Add home and away team ids for each event
    
events = events.merge(fawsl[['match_id', 'home_team.home_team_id', 'away_team.away_team_id']],
                      how='left')


## Add opponent team id for each event

events['opponent.id'] = np.where(events['team.id'] == events['home_team.home_team_id'],
                                 events['away_team.away_team_id'],
                                 events['home_team.home_team_id'])



## Add the current score


# Flag all goals (Shot + Own Goal For)

events.loc[(events['shot.outcome.id'] == 97) | (events['type.id'] == 25), 'goal'] = 1
events['goal'] = events['goal'].fillna(0)



# Add score (goals for and against) at any time during the game

home = (events['team.id'] == events['home_team.home_team_id'])
away = (events['team.id'] == events['away_team.away_team_id'])

events.loc[home, 'home_team.goals'] = events[home].groupby(['match_id', 'team.id'])['goal'].cumsum()
events.loc[away, 'away_team.goals'] = events[away].groupby(['match_id', 'team.id'])['goal'].cumsum()

events['home_team.goals'] = events['home_team.goals'].fillna(method='ffill').fillna(0)
events['away_team.goals'] = events['away_team.goals'].fillna(method='ffill').fillna(0)

events['GF'] = np.where(home,
                        events['home_team.goals'],
                        events['away_team.goals'])

events['GA'] = np.where(home,
                        events['away_team.goals'],
                        events['home_team.goals'])



## Add cumulative xG (for and against) at any time during the game

events.loc[home, 'home_team.xg'] = events[home].fillna({'shot.statsbomb_xg':0}).groupby(
    ['match_id', 'team.id'])['shot.statsbomb_xg'].cumsum()
events.loc[away, 'away_team.xg'] = events[away].fillna({'shot.statsbomb_xg':0}).groupby(
    ['match_id', 'team.id'])['shot.statsbomb_xg'].cumsum()

events['home_team.xg'] = events['home_team.xg'].fillna(method='ffill').fillna(0)
events['away_team.xg'] = events['away_team.xg'].fillna(method='ffill').fillna(0)

events['xgF'] = np.where(home,
                        events['home_team.xg'],
                        events['away_team.xg'])

events['xgA'] = np.where(home,
                        events['away_team.xg'],
                        events['home_team.xg'])



## Add status (win / draw / loose), goal and xG differences

events.loc[events['GF']>events['GA'], 'status'] = 'W'
events.loc[events['GF']==events['GA'], 'status'] = 'D'
events.loc[events['GF']<events['GA'], 'status'] = 'L'
events['GD'] = events['GF'] - events['GA']
events['xgD'] = events['xgF'] - events['xgA']




obv = events.groupby('player.id')['obv_total_net'].sum().reset_index()



lineups_AstonVilla = lineups[lineups['match_id'] == matches['AstonVilla']]
lineups_AstonVilla.loc['player_name'] = lineups_AstonVilla.player_nickname.combine_first(lineups_AstonVilla.player_name)



columns = ['minute', 'second', 'team.id', 'opponent.id',
           'GF', 'GA', 'GD', 'status',
           'xgF', 'xgA', 'xgD',
           'player.id', 'position_off', 'sum_obv_off', 'obv_off',
           'substitution.replacement.id', 'position_in', 'sum_obv_in']

categorical = ['team.id', 'opponent.id', 'status',
               'player.id', 'position_off',
               'substitution.replacement.id', 'position_in']


ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")