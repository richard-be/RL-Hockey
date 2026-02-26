# RL Project Team inthesac
Richard Below, Giorgi Gogelashvili, Jakob Kleine

This Repository contains code for the four models we implemented for the Reinforcement Learning Course in WS26/27, along with their respective training pipelines and additions.
The ensemble model combining TD3, SAC and CrossQ ranked 6th out of 178 participants.

## TD3
The implementation is based on CleanRL's implementations of TD3, RND, and PER: 
- TD3: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
- RND: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
- PER: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py

Own modifications were marked with Notes whereever possible. The use of pink noise was implemented by Richard Below. Smart auto-completion by Github co-pilot was used for the implementation. 

## SAC

The implementation is based on CleanRL SAC implementation for continuous actions. We added colored noise for action exploration and Randomized Ensembled Double Q-Learning for a higher Update-to-data ratio.

## CrossQ
The implementation is based on [CrossQ](https://arxiv.org/abs/1902.05605) and recently proposed [CrossQ + Weight Normalization](https://arxiv.org/abs/2502.07523) papers. 


## Ensemble
Ensemble models combine predictions of all three model variants in different manners:

* **Random Action Ensemble**: Uniformly sampling a model.

* **Mean Action Ensemlbe**: Averaging predicted actions.

* **Weighted Mean Ensemble**: Taking the mean of the predicted actions, weighted by their critic's predicted value.

* **Greedy Mean Ensemble**: Taking the action of the model with the highest predicted value from its critic.


## Folders
**experiments** contains config files that define the different TD3 training configurations.

**jobs** contains the batch files used for scheduling jobs on the cluster

**models** contains the weights of the models used for the tournament

**scripts** contains python executables

**singularity/recipes** contains the recipes for building singularity images 

**src** contains the source code

Most of the folders have subfolders for the different models we implemented
