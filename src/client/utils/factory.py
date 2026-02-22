import torch
import numpy as np

from crossq.models.actor import GaussianPolicyConfig, GaussianPolicy
from crossq.models.critic import QNetwork as CrossQCritic
from td3.algorithm.td3 import Actor as TD3Actor
from td3.algorithm.td3 import QNetwork as TD3Critic
from sac.agent.sac import Actor as SACActor
from sac.agent.sac import SoftQNetwork as SACCritic

from utils.actors import Actor, ActorCritic, Critic, GreedyEnsemble, MeanActionEnsemble, RandomActorEnsemble

def crossq_constructor(env) -> GaussianPolicy:
    config = GaussianPolicyConfig(input_dim=np.prod(env.observation_space.shape),
                          action_dim=np.prod(env.action_space.shape))
    actor = GaussianPolicy(min_action=torch.from_numpy(env.action_space.low), 
                        max_action=torch.from_numpy(env.action_space.high), 
                        config=config)
    return actor


def sac_constructor(env) -> SACActor:
    actor = SACActor(env)
    actor.eval()
    actor._act = actor.act
    def act_(obs):
        with torch.no_grad():
            actor.eval()
            action, _, _ = actor._act(torch.from_numpy(obs).to(torch.float32).unsqueeze(0))
            actor.train()
            return action.squeeze(0).detach().cpu().numpy()
    actor.act = act_
    return actor

ACTOR_CONSTRUCTORS = {
    "sac": sac_constructor,
    "td3": TD3Actor,
    "crq": crossq_constructor, 
}

def load_actor_weights(actor: torch.nn.Module, weight_path: str, device: str) -> None:
    actor_state = torch.load(weight_path, map_location=device)
    if (isinstance(actor_state, tuple) or isinstance(actor_state, list)) and len(actor_state) == 3:   
        actor_state = actor_state[0] 
    actor.load_state_dict(actor_state)

ACTOR_WEIGHT_LOADER_FUNCTIONS = {
    "sac": load_actor_weights,
    "td3": load_actor_weights,
    "crq": load_actor_weights, 
}


def construct_actor(algorithm: str,
                    weight_path: str,
                    env,
                    device: str = "cpu") -> Actor:

    
    if not hasattr(env, "single_observation_space"):
        env.single_observation_space = env.observation_space
    if not hasattr(env, "single_action_space"):
        env.single_action_space = env.action_space

    assert algorithm in ACTOR_CONSTRUCTORS, f"Unknown algorithm {algorithm}. Supported algorithms: {list(ACTOR_CONSTRUCTORS.keys())}"

    actor_constructor, weight_loader = ACTOR_CONSTRUCTORS[algorithm], ACTOR_WEIGHT_LOADER_FUNCTIONS[algorithm]

    actor = actor_constructor(env)
    weight_loader(actor, weight_path=weight_path, device=device)
    return actor


CRITIC_CONSTRUCTORS = {
    "sac": SACActor,
    "td3": TD3Actor,
    "crq": crossq_constructor, 
}

def construct_critic(algorithm: str,
                    weight_path: str,
                    env,
                    device: str = "cpu") -> Critic:
    # TODO
    ...


def construct_actor_critic(algorithm: str,
                        weight_path: str,
                        env,
                        device: str = "cpu") -> ActorCritic:
    actor = construct_actor(algorithm=algorithm,
                            weight_path=weight_path,
                            env=env,
                            device=device)
    critic = construct_critic(algorithm=algorithm,
                              weight_path=weight_path,
                              env=env,
                              device=device)
    return ActorCritic(actor=actor, ciritc=critic)

ENSEMBELE_PIECE_CONSTRUCTORS = {
    "random": construct_actor,
    "mean": construct_actor,
    "greedy": construct_actor_critic
}

ENSEMBELE_CONSTRUCTORS = {
    "random": RandomActorEnsemble,
    "mean": MeanActionEnsemble,
    "greedy": GreedyEnsemble
}

def construct_ensemble(algorithm: str,
                       sub_algorithms: dict[str, str],
                       env,
                       device: str = "cpu") -> Actor:

    assert algorithm in ENSEMBELE_PIECE_CONSTRUCTORS, f"Unknown ensemble algorithm {algorithm}. Supported algorithms: {list(ENSEMBELE_PIECE_CONSTRUCTORS.keys())}"

    piece_constructor = ENSEMBELE_PIECE_CONSTRUCTORS[algorithm]
    sub_agents = []
    for sub_algorithm, weight_path in sub_algorithms.items():
        sub_agents.append(piece_constructor(sub_algorithm, weight_path, env, device))
    
    constructor = ENSEMBELE_CONSTRUCTORS[algorithm]

    return constructor(sub_agents)