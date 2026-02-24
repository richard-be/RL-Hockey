import numpy as np
from typing import Protocol



class Critic(Protocol):

    def estimate(self, observation: np.array, action: np.array) -> float:
        ...



class CriticEnsemble:

    def __init__(self, critics: list[Critic]):
        self.critics = critics

    def estimate(self, observation: np.array, action: np.array) -> float:
        return min([critic.estimate(observation=observation,
                                    action=action) for critic in self.critics])



class Actor(Protocol):

    def act(self, observation: np.array) -> np.array:
        ...



class ActorCritic:

    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic

    def act(self, observation: np.array) -> np.array:
        return self.actor.act(observation)


    def estimate(self, observation: np.array, action: np.array) -> float:
        return self.critic.estimate(observation, action)



class RandomActorEnsemble:

    def __init__(self, actors: list[Actor]) -> None:
        self.actors = actors

    def act(self, observation: np.array) -> np.array:
        actor = np.random.choice(self.actors)
        return actor.act(observation)


<<<<<<< HEAD
=======
class WeightedMeanEnsemble: 
    def __init__(self, actor_critics: list[ActorCritic], temperature: float = 1) -> None:
        self.actor_critics = actor_critics
        self.temperature = temperature

    def act(self, observation: np.array) -> np.array:
        actions = [act_crit.act(observation) for act_crit in self.actor_critics]
        q_values = np.array([act_crit.estimate(observation=observation, action=action) for act_crit, action in zip(self.actor_critics, actions)])
        weights = self.softmax(q_values) 
        return np.sum(actions * weights, axis=0)

    def softmax(self, xs): 
        xs = xs / self.temperature
        max_value = np.max(xs) 
        exps = np.exp(xs - max_value)
        return exps / np.sum(exps)
>>>>>>> main

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



