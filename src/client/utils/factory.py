import torch
import numpy as np

from src.crossq.models.actor import GaussianPolicyConfig, GaussianPolicy
from src.crossq.models.critic import QNetwork as CrossQCritic
from src.crossq.models.feedforward import NNConfig
from src.td3.algorithm.models import Actor as TD3Actor
from src.td3.algorithm.models import QNetwork as TD3Critic
from src.sac.agent.sac import Actor as SACActor
from src.sac.agent.sac import SoftQNetwork as SACCritic


from functools import partial
from src.client.utils.actors import Actor, ActorCritic, Critic, GreedyEnsemble, MeanActionEnsemble, RandomActorEnsemble, WeightedMeanEnsemble, CriticEnsemble

def crossq_constructor(env) -> GaussianPolicy:
    config = GaussianPolicyConfig(input_dim=np.prod(env.single_observation_space.shape),
                          action_dim=np.prod(env.single_action_space.shape))
    actor = GaussianPolicy(min_action=torch.from_numpy(env.single_action_space.low), 
                        max_action=torch.from_numpy(env.single_action_space.high), 
                        config=config)
    return actor


# def sac_constructor(env) -> SACActor:
#     actor = SACActor(env)
#     actor.eval()
#     actor._act = actor.act
#     def act_(obs):
#         with torch.no_grad():
#             actor.eval()
#             action, _, _ = actor._act(torch.from_numpy(obs).to(torch.float32).unsqueeze(0))
#             actor.train()
#             return action.squeeze(0).detach().cpu().numpy()
#     actor.act = act_
#     return actor

ACTOR_CONSTRUCTORS = {
    "sac": SACActor,
    "td3": TD3Actor,
    "crq": crossq_constructor, 
}

def load_actor_weights(actor: torch.nn.Module, weight_path: str, device: str) -> None:
    print("loading weights from", weight_path)
    actor_state = torch.load(weight_path, map_location=device)
    if (isinstance(actor_state, tuple) or isinstance(actor_state, list)) and len(actor_state) > 1:   
        actor_state = actor_state[0]
    actor.load_state_dict(actor_state)


def load_sac_actor_weights(actor: torch.nn.Module, weight_path: str, device: str) -> None:
    actor_state = torch.load(weight_path, map_location=device, weights_only=False)
    if (isinstance(actor_state, tuple) or isinstance(actor_state, list)) and len(actor_state) > 1:   
        actor_state: torch.nn.Module = actor_state[0]
    actor.load_state_dict(actor_state.state_dict())

ACTOR_WEIGHT_LOADER_FUNCTIONS = {
    "sac": load_sac_actor_weights,
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




def crossq_critic_constructor(env) -> GaussianPolicy:
    config = NNConfig(input_dim=np.prod(env.observation_space.shape) + np.prod(env.action_space.shape),
                          output_dim=1, hidden_dim=512)
    
    critic = CrossQCritic(config=config)

    def _estimate(observation: np.array, action: np.array) -> float:
        with torch.no_grad():
            critic.eval()
            return critic.q_value(s=torch.from_numpy(observation).to(torch.float32).unsqueeze(0),
                                 a=torch.from_numpy(action).to(torch.float32).unsqueeze(0)).item()

    critic.estimate = _estimate
    return critic


def critic_constructor(env, module_class: torch.nn.Module) -> GaussianPolicy:
    actor = module_class(env)

    def _estimate(observation: np.array, action: np.array) -> float:
        with torch.no_grad():
            return actor.forward(torch.from_numpy(observation).to(torch.float32).unsqueeze(0),
                                 torch.from_numpy(action).to(torch.float32).unsqueeze(0)).item()

    actor.estimate = _estimate
    return actor


CRITIC_CONSTRUCTORS = {
    "sac": partial(critic_constructor, module_class=SACCritic),
    "td3": partial(critic_constructor, module_class=TD3Critic),
    "crq": crossq_critic_constructor, 
}


def load_critic_weights(critics: list[torch.nn.Module], weight_path: str, device: str) -> None:

    model_states = torch.load(weight_path, map_location=device, weights_only=True)
   
    if (isinstance(model_states, tuple) or isinstance(model_states, list)) and len(model_states) >= 1:   
        model_states = model_states[1:]
    else:
        raise ValueError("Critic weights not Found")
    for critic, model_state in zip(critics, model_states):
        critic.load_state_dict(model_state)

    return CriticEnsemble(critics)


def load_sac_weights(critics: list[torch.nn.Module], weight_path: str, device: str) -> None:

    model_states = torch.load(weight_path, map_location=device, weights_only=False)
   
    if (isinstance(model_states, tuple) or isinstance(model_states, list)) and len(model_states) >= 1:   
        model_states: list[torch.nn.Module] = model_states[1:]
    for critic, model_state in zip(critics, model_states):
        critic.load_state_dict(model_state.state_dict())

    return CriticEnsemble(critics)


CRITIC_WEIGHT_LOADER_FUNCTIONS = {
    "sac": load_sac_weights,
    "td3": load_critic_weights,
    "crq": load_critic_weights, 
}

def construct_critic(algorithm: str,
                    weight_path: str,
                    env,
                    device: str = "cpu") -> Critic:
    if not hasattr(env, "single_observation_space"):
        env.single_observation_space = env.observation_space
    if not hasattr(env, "single_action_space"):
        env.single_action_space = env.action_space

    assert algorithm in CRITIC_CONSTRUCTORS, f"Unknown Critic algorithm {algorithm}. Supported algorithms: {list(CRITIC_CONSTRUCTORS.keys())}"
    critic_constructor, weight_loader = CRITIC_CONSTRUCTORS[algorithm], CRITIC_WEIGHT_LOADER_FUNCTIONS[algorithm]

    critics = [critic_constructor(env), critic_constructor(env)]
 
    critics = weight_loader(critics, weight_path, device)

    return critics


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
    return ActorCritic(actor=actor, critic=critic)

ENSEMBELE_PIECE_CONSTRUCTORS = {
    "random": construct_actor,
    "mean": construct_actor,
    "greedy": construct_actor_critic,
    "weighted": construct_actor_critic
}

ENSEMBELE_CONSTRUCTORS = {
    "random": RandomActorEnsemble,
    "mean": MeanActionEnsemble,
    "greedy": GreedyEnsemble,
    "weighted": WeightedMeanEnsemble, 
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