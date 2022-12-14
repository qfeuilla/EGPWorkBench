import torch
from torch import nn
# from procgen import ProcgenEnv
from common.storage import Storage
from common.logger import Logger

class BaseAgent(object):
    """
    Class for the basic agent objects.
    To define your own agent, subclass this class and implement the functions below.
    """

    def __init__(self, 
                 env, #: ProcgenEnv,
                 env_test, #: ProcgenEnv,
                 policy: nn.Module,
                 logger: Logger,
                 storage: Storage,
                 storage_test: Storage,
                 device: torch.device,
                 num_checkpoints: int):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.env_test = env_test
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.storage_test = storage_test
        self.device = device
        self.num_checkpoints = num_checkpoints
        
        self.t = 0
        
    def predict(self, obs):
        """
        Predict the action with the given input 
        """
        pass
        
    def update_policy(self):
        """
        Train the neural network model
        """
        pass
        
    def train(self, num_timesteps):
        """
        Train the agent with collecting the trajectories
        """
        pass

    def evaluate(self):
        """
        Evaluate the agent
        """
        pass
