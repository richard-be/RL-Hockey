import numpy as np
from typing import Protocol


class Critic(Protocol):

    def estimate(self, observation: np.array, action: np.array) -> float:
        ...

class Actor(Protocol):

    def act(self, observation: np.array) -> np.array:
        ...


class ActorCritic:
    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic

    def act(self, observation: np.array) -> np.array:
        return self.actor.act(observation=observation)


    def estimate(self, observation: np.array, action: np.array) -> float:
        return self.critic.act(observation=observation, action=action)


class RandomActorEnsemble:

    def __init__(self, actors: list[Actor]) -> None:
        self.actors = actors

    def act(self, observation: np.array) -> np.array:
        actor = np.random.choice(self.actors)
        return actor.act(observation)


class MeanActionEnsemble:

    def __init__(self, actors: list[Actor]) -> None:
        self.actors = actors

    def act(self, observation: np.array) -> np.array:
        actions = np.array([actor.act(observation) for actor in self.actors])
        mean_action = actions.mean(axis=0)
        return mean_action
    

class GreedyEnsemble:
    def __init__(self, actor_critics: list[ActorCritic]) -> None:
        self.actor_critics = actor_critics

    def act(self, observation: np.array) -> np.array:
        actions = [act_crit.act(observation) for act_crit in self.actor_critics]
        max_idx = np.argmax(np.array([act_crit.estimate(observation=observation, action=action) 
                                      for act_crit, action in zip(self.actor_critics, actions)]))
        return actions[max_idx]