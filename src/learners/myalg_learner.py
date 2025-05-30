import copy
import time

import torch as th
import torch
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qgroupmix import GroupMixer
from modules.mixers.qgroupmix_atten import GroupMixerAtten
from modules.mixers.qgattenmix import GAttenMixer
from modules.mixers.qghypermix import GHyperMixer
from modules.mixers.myalg_mixer import MyAlgMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num


def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        return target_mac_out


def calculate_n_step_td_target(mixer, target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        if mixer == "myalg_mixer":
            target_max_qvals = target_mixer(target_max_qvals, batch["state"], batch["obs"])
        else:
            target_max_qvals = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class MyAlgLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.obsLossWeight = 0.02
        self.actionLossWeight = 0.2
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        elif args.mixer == "myalg_mixer":
            self.mixer = MyAlgMixer(args)
        else:
            raise "mixer error"

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        self.mac_optimiser = Adam(params=self.mac.parameters(), lr=self.args.lr, weight_decay=getattr(self.args, "weight_decay", 0))
        
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)

    def compute_kl_divergence(self, p: th.Tensor, q: th.Tensor, dim: int = -1, epsilon: float = 1e-10) -> th.Tensor:
        if p.shape != q.shape:
            raise ValueError(f"tensor shape not matching: p.shape={p.shape}, q.shape={q.shape}")
        global_min = min(th.min(p), th.min(q))
        if global_min < 0:
            p_shifted = p - global_min + epsilon
            q_shifted = q - global_min + epsilon
        else:
            p_shifted = p + epsilon
            q_shifted = q + epsilon
        p_normalized = p_shifted / (p_shifted.sum(dim=dim, keepdim=True) + epsilon)
        q_normalized = q_shifted / (q_shifted.sum(dim=dim, keepdim=True) + epsilon)
        kl_div = th.sum(p_normalized * th.log(p_normalized / (q_normalized + epsilon) + epsilon), dim=dim)
        return kl_div
    def compute_MSE(self, p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if p.shape != q.shape:
            raise ValueError(f"tensor shape not matching: p.shape={p.shape}, q.shape={q.shape}")
        if p.shape[dim] != 1:
            raise ValueError(f"last d should be 1, but get {p.shape[dim]}")
        diff_squared = (p - q) ** 2
        MSE_dist = torch.mean(torch.sum(diff_squared, dim=dim))
        MSE_dist = MSE_dist.unsqueeze(dim)
        return MSE_dist
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_real = batch["actions_real"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
    
        self.mac.set_train_mode()
        self.mac.init_hidden(batch.batch_size)
        n_actions = batch["avail_actions"].shape[-1]  
        n_agents = batch["avail_actions"].shape[-2]
        mac_out = th.zeros(batch.batch_size, batch.max_seq_length, n_agents, n_actions, device=self.device)
        mac_out_obsReal = th.zeros_like(mac_out)
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                mac_out_obsReal[:, t] = self.mac.forward(batch, t=t, noise=False)
            mac_out_obsReal[avail_actions == 0] = -9999999
        
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_out[:, t] = self.mac.forward(batch, t=t)
        mac_out[avail_actions == 0] = -9999999
        obs_error = th.mean((mac_out - mac_out_obsReal) ** 2)
    
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = self.pool.apply_async(calculate_target_q, (self.target_mac, batch, True, self.args.thread_num)).get()
            else:
                target_mac_out = calculate_target_q(self.target_mac, batch)
            cur_max_actions = mac_out.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
    
            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(calculate_n_step_td_target, (self.args.mixer, self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma, self.args.td_lambda, True, self.args.thread_num, False, None)).get()
            else:
                targets = calculate_n_step_td_target(self.args.mixer, self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma, self.args.td_lambda)
    
        self.mixer.train()

        if self.args.mixer == "myalg_mixer":
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets
        masked_td_error = 0.5 * td_error.pow(2) * mask.expand_as(td_error)
        TD_loss = masked_td_error.sum() / mask.sum()
        
        loss = TD_loss + self.obsLossWeight * obs_error
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
    
        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                mask_elems = mask.sum().item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env
        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
