from .base_agent import BaseAgent
from common.logger import Logger
from common.storage import Storage
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
from torch import nn
import numpy as np
# from procgen import ProcgenEnv
from typing import List

import os
from torchvision import transforms
from PIL import Image

def get_goal_target(asset_index : int, is_test : bool) -> torch.Tensor:    
    # Get asset
    asset_path = f"../procgenEGP/procgen/data/assets/kenney/Items{'_test' if is_test else ''}/"
    
    target_file = os.listdir(asset_path)[asset_index]

    target_imgs = Image.open(asset_path + target_file).resize((8,8))
    convert_tensor = transforms.ToTensor()

    target_imgs = torch.FloatTensor(convert_tensor(target_imgs)[:3, :, :] /255.0)
    return target_imgs.unsqueeze(dim=0)

class PPO(BaseAgent):
    def __init__(self,
                 env, #: ProcgenEnv,
                 env_test, #: ProcgenEnv,
                 policy: nn.Module,
                 logger: Logger,
                 storage: Storage,
                 storage_test: Storage,
                 device: torch.device,
                 game_assets: List[int],
                 n_checkpoints: int,
                 is_test: bool = False,
                 n_steps: int = 128,
                 n_envs: int = 8,
                 epoch: int = 3,
                 mini_batch_per_epoch: int = 8,
                 mini_batch_size: int = 32*8,
                 gamma: float = 0.99,
                 lmbda: float = 0.95,
                 learning_rate: float = 2.5e-4,
                 grad_clip_norm: float = 0.5,
                 eps_clip: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 normalize_adv: bool = True,
                 normalize_rew: bool = True,
                 use_gae: bool = True,
                 **kwargs):

        super(PPO, self).__init__(env, env_test, policy, logger, storage, storage_test, device, n_checkpoints)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.goal_targets = torch.cat([get_goal_target(i, is_test) for i in game_assets])

    # def predict(self, obs, hidden_state, done, target):
    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        with torch.autocast("cuda"):
            self.policy.train()
            for e in range(self.epoch):
                recurrent = self.policy.is_recurrent()
                generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                            recurrent=recurrent)
                for sample in generator:
                    obs_batch, hidden_state_batch, act_batch, done_batch, \
                        old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, target = sample
                    mask_batch = (1-done_batch)
                    dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch, target)

                    # Clipped Surrogate Objective
                    log_prob_act_batch = dist_batch.log_prob(act_batch)
                    ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                    pi_loss = -torch.min(surr1, surr2).mean()

                    # Clipped Bellman-Error
                    clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                    v_surr1 = (value_batch - return_batch).pow(2)
                    v_surr2 = (clipped_value_batch - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                    # Policy Entropy
                    entropy_loss = dist_batch.entropy().mean()
                    loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                    loss.backward()

                    # Let model to handle the large batch-size with small gpu-memory
                    if grad_accumulation_cnt % grad_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    grad_accumulation_cnt += 1
                    pi_loss_list.append(pi_loss.item())
                    value_loss_list.append(value_loss.item())
                    entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def generate_wrong_target(self, target_idx_to_update: np.array) -> torch.Tensor:
        tmp_targets_idx = np.arange(self.goal_targets.shape[0])
        tmp_targets = torch.clone(self.goal_targets)
        
        wrong_target_idx = np.random.choice(tmp_targets_idx[~np.isin(tmp_targets_idx, target_idx_to_update)], 
                                        len(target_idx_to_update), 
                                        replace=True)
        tmp_targets[target_idx_to_update] = tmp_targets[wrong_target_idx]
        return tmp_targets

    def train(self, num_timesteps: int):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0

        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()

            # Generate wrong targets for 25% of the envs
            wrong_label_idx = np.random.randint(low=0, high=self.n_envs, size=self.n_envs // 4)
            tmp_targets = self.generate_wrong_target(wrong_label_idx)

            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, tmp_targets)
                next_obs, rew, done, info = self.env.step(act)
                
                # Change reward if get rewarded for collecting the wrong target
                rew[wrong_label_idx] = np.abs(rew[wrong_label_idx]) * -1

                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value, tmp_targets)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, tmp_targets)
            self.storage.store_last(obs, hidden_state, last_val, tmp_targets) 
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed(rew_batch, done_batch)
            self.logger.write_summary(summary)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt+1) * save_every):
                torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
                           '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()

# def train(self, num_timesteps: int):
#         save_every = num_timesteps // self.num_checkpoints
#         checkpoint_cnt = 0

#         obs = self.env.reset()
#         hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
#         done = np.zeros(self.n_envs)
#         while self.t < num_timesteps:
#             # Run Policy
#             self.policy.eval()

#             wrong_label_idx = np.random.randint(low=0, high=obs.shape, size=4)

#             for _ in range(self.n_steps):
#                 act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, self.goal_targets)
#                 next_obs, rew, done, info = self.env.step(act)
#                 self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value, self.goal_targets)
#                 obs = next_obs
#                 hidden_state = next_hidden_state
#             _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, self.goal_targets)
#             self.storage.store_last(obs, hidden_state, last_val, self.goal_targets)
#             # Compute advantage estimates
#             self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

#             # Optimize policy & valueq
#             summary = self.optimize()
#             # Log the training-procedure
#             self.t += self.n_steps * self.n_envs
#             rew_batch, done_batch = self.storage.fetch_log_data()
#             self.logger.feed(rew_batch, done_batch)
#             self.logger.write_summary(summary)
#             self.logger.dump()
#             self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
#             # Save the model
#             if self.t > ((checkpoint_cnt+1) * save_every):
#                 torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
#                            '/model_' + str(self.t) + '.pth')
#                 checkpoint_cnt += 1
#         self.env.close()


    def test(self, num_timesteps: int):
        obs = self.env_test.reset()
        hidden_state = np.zeros((self.n_envs, self.storage_test.hidden_state_size))
        done = np.zeros(self.n_envs)
        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, self.goal_targets)
                next_obs, rew, done, info = self.env_test.step(act)
                self.storage_test.store(obs, hidden_state, act, rew, done, info, log_prob_act, value, self.goal_targets)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, self.goal_targets)
            self.storage_test.store_last(obs, hidden_state, last_val, self.goal_targets)

        self.env_test.close()