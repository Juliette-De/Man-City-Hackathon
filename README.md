# Hays x Man City Football Club Hackathon 2023

This application offers an in-game football substitution recommendation system. It is intended to be used as a decision-making aid during a football game.

## Quick start

The application is deployed at [this address](https://substitution-recommender.streamlit.app).

Alternatively, to run it locally:

- Clone this github repository or upload all of its files to the folder where you want to place this project.

- Install the necessary packages from the requirements.txt file. In the terminal, replacing path with the path of your dedicated folder:
```
pip install -r path/requirements.txt
```

- Launch the application:
```
streamlit run path/↔️_Substitution_Suggestions.py
```


## Features

This application offers the following three features, each on one page:
- The "↔️ Substitutions Suggestions" page features a substitution recommendation system based on StatsBomb's On-Ball Values. Every minute a model tests different combinations of possible substitutions and brings out those that would be the most profitable.
  - On the one hand, it makes it possible to identify the players who are underperforming during the game;
  - On the other hand, it suggests a potential substitute - based on her past performances and the context of the game.
- The "📈 All players" page offers an overview of the performance of all players on the pitch
- The "⏱️ Opponent Substitution Prediction" page presents the probable future substitutions of the opponent.


## Next steps

- Consider not only the cumulative On-Ball Values but also a Time Series of their evolution during the match.
- Include tracking data to also take into account off-ball events (pressing, etc.).



## Background

This application was created as part of a hackathon organized by Hays and Man City Football Club.
