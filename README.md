# Snake Game AI
in this project i made a snake game (game.py), then trained an ai to play the game using pytorch reinforced learning (gameai.py, agent.py, brain.py)

## video about the project
<a href="https://youtu.be/F2BhlaFetIU" target="_blank">
   <img src="https://i9.ytimg.com/vi/F2BhlaFetIU/mqdefault_custom_2.jpg?v=6a2d555b&sqp=CKDCutEG&rs=AOn4CLBmJoCaW15YNaPPfdWLvnMXsCGnHQ" alt="Watch the video" width="600" height="auto" />
</a>


## THE GAME (game.py)
you can play this game by copying the repo and

## GAME THAT THE AI PLAYS (gameai.py)
launching this file won't show nothing, it is not meant to be launched

## THE AGENT THAT PLAYS (agent.py)
launching this file will open the window (one that'1l open when starting game.py) but you won't play , instead the agent plays the game (in this case it'll train the model)

## THE BRAIN (brain.py)
here lies the Neural Network , yeah that's it

## mutliple?
i tried running training using multiple snakes at once , failed miserably.

## MODELS
inside this folder you'll find multiple models (from 1 to 500+ games played) you can load them and train them in the agent.py (adding that next) , or launching the agent that'll play the game using the model selected 


# HOW TO RUN

## REQUIREMENTS
```
pip install pytorch numpy pygame
```

## RUN
to train a model
```
python agent.py
```
else to play the game 
```
python game.py
```
