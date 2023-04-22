import pandas as pd

from load_data import events, obv, lineups_AstonVilla, columns, categorical, ohe



def build_test(m):
    
    """
    m: minute of the game
    """

    # matches['AstonVilla'] = 3856030
    events_AstonVilla = events[(events['match_id'] == 3856030) & (events['minute']<m)]


    ### Players in game

    # Every minute, compute the obv of all players on the pitch

    substitued = events_AstonVilla.loc[events_AstonVilla['type.id']==19, 'player.id']

    players_on_pitch = events_AstonVilla[(events_AstonVilla['team.id'] == 746) &
                                         (~events_AstonVilla['player.id'].isin(substitued))].groupby(
        'player.id').aggregate({'position.id': pd.Series.mode,
                                'obv_total_net' : 'sum'}).reset_index().rename(
        columns={'position.id': 'position_off',
                 'obv_total_net': 'obv_off'})

    players_on_pitch = players_on_pitch.merge(obv.rename(columns={'obv_total_net': 'sum_obv_off'}),
                                              how='left')


    ### Players on the bench

    bench = lineups_AstonVilla[(lineups_AstonVilla['team_id'] == 746) &
                               (~lineups_AstonVilla['player_id'].isin(events_AstonVilla['player.id'].unique()))]


    ## Add the positions on the players on the bench

    bench = bench.merge(events.groupby('player.id')['position.id'].agg(pd.Series.mode), how='left',
                       left_on = 'player_id', right_on = 'player.id').rename(
        columns={'position.id': 'position_in'})

    # Players that didn't play during the 6 available games
    bench.loc[bench['player_nickname'] == 'Vicky Losada', 'position_in'] = 14
    bench.loc[bench['player_name'] == 'Alexandra MacIver', 'position_in'] = 1


    ## Add the OBV on the players on the bench

    bench = bench.merge(obv.rename(columns={'player.id': 'player_id', 'obv_total_net': 'sum_obv_in'}),
                        how='left').fillna({'sum_obv_in': 0})


    bench = bench.rename(columns={'player_id': 'substitution.replacement.id'}) 



    ### Dataframes : Matrices players on the pitch x player on the bench

    X_test = players_on_pitch.merge(bench[['substitution.replacement.id', 'position_in', 'sum_obv_in']],
                                    how='cross')

    X_test = events_AstonVilla.loc[(events_AstonVilla['team.id'] == 746),
                          ['minute', 'second', 'team.id', 'opponent.id', 'GF', 'GA', 'GD', 'status',
                           'xgF', 'xgA', 'xgD']].tail(1).merge(X_test, how='cross')
    
    return X_test




def preprocessing(array_hot_encoded, df):
    #Convert it to df
    data_hot_encoded = pd.DataFrame(array_hot_encoded,
                                    index=df.index,
                                    columns = ohe.get_feature_names_out())
    
    #Extract only the columns that didnt need to be encoded
    data_other_cols = df.drop(columns=categorical)
    
    #Concatenate the two dataframes : 
    return pd.concat([data_hot_encoded, data_other_cols], axis=1)




def predict_best_subs(model, m):
    
    X_test = build_test(m)[columns]
    
    test_array_hot_encoded = ohe.transform(X_test[categorical])
    X_test = preprocessing(test_array_hot_encoded, X_test)
    
    
    # Predict on the model already trained
    
    train_predictions = model.predict(X_test)
    
    X_test['predicted_obv_in'] = train_predictions
    X_test['predicted_obv'] = X_test['predicted_obv_in'] - X_test['obv_off']

    
    # Rank by best OBV
    
    best_subs = X_test.sort_values('predicted_obv', ascending=False)


    # Keep only relevant substitutions
    
    best_subs['player'] = best_subs[[col for col in X_test if col.startswith('player.id')]].sum(axis=1)
    best_subs = best_subs[best_subs['player']>0]

    best_subs['substitution.replacement'] = best_subs[[col for col in X_test if col.startswith('substitution.replacement.id')]].sum(axis=1)
    best_subs = best_subs[best_subs['substitution.replacement']>0]



    # Extract information from one-hot-encoded columns and convert to int

    best_subs['player.id'] = [int(float(x[10:])) for x in best_subs[
        [col for col in X_test if col.startswith('player.id')]].idxmax(axis=1)]

    best_subs['position_in'] = [int(float(x[12:])) for x in best_subs[
        [col for col in X_test if col.startswith('position_in')]].idxmax(axis=1)]

    best_subs['substitution.replacement.id'] = [int(float(x[28:])) for x in best_subs[
        [col for col in X_test if col.startswith('substitution.replacement.id')]].idxmax(axis=1)]

    best_subs['position_off'] = [int(float(x[13:])) for x in best_subs[
        [col for col in X_test if col.startswith('position_off')]].idxmax(axis=1)]


    best_subs = best_subs.merge(events.groupby('player.id')['position.id'].unique(),
                                how='left',
                                left_on = 'substitution.replacement.id', right_on = 'player.id').rename(
        columns={'position.id': 'possible_position_in'})

    
    # Filter by position
    best_subs = best_subs[[x in y for x,y in zip(best_subs['position_off'], best_subs['possible_position_in'])]
                         & (best_subs['predicted_obv']>0)]

    
    # Add name of the two players

    best_subs = best_subs.merge(lineups_AstonVilla[['player_id', 'player_name']].rename(
        columns = {'player_id': 'player.id',
                   'player_name': 'player_name_off'}),
                                how='left')
    
    best_subs = best_subs.merge(lineups_AstonVilla[['player_id', 'player_name']].rename(
        columns = {'player_id': 'substitution.replacement.id',
                   'player_name': 'player_name_in'}),
                                how='left')

    return best_subs[['player_name_off', 'position_off',
                      'player_name_in', 'position_in', 'possible_position_in',
                      'predicted_obv']]
