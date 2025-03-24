from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import torch
import torch as th
import numpy as np
import time


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelMyAlg:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker,
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)
        if self.args.evaluate:
            print("Waiting the environment to start...")
            time.sleep(5)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.last_test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        if self.args.use_cuda and not self.args.cpu_inference:
            self.batch_device = self.args.device
        else:
            self.batch_device = "cpu" if self.args.buffer_cpu_only else self.args.device
        print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& self.batch_device={}".format(
            self.batch_device))
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.batch_device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        if (self.args.use_cuda and self.args.cpu_inference) and str(self.mac.get_device()) != "cpu":
            self.mac.cpu()  # copy model to cpu

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "obs_real":[]
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["obs_real"].append(data["obs_real"])

        self.batch.update(pre_transition_data, ts=0, mark_filled=True)

        self.t = 0
        self.env_steps_this_run = 0

    def tamper_action(self, avail_actions, action, n_agents_to_tamper=1, tamper_prob=0.1):
        b, a, v = avail_actions.shape
        tampered_action = action.clone()
        for batch_idx in range(b):
            if th.rand(1).item() > tamper_prob:
                continue 
            if n_agents_to_tamper == "random":
                num_to_tamper = th.randint(1, a + 1, (1,)).item()
            else:
                num_to_tamper = min(n_agents_to_tamper, a)
            agent_indices = th.randperm(a)[:num_to_tamper].to(avail_actions.device)
            for agent_idx in agent_indices:
                avail_mask = avail_actions[batch_idx, agent_idx]
                available_indices = th.where(avail_mask == 1)[0]
                current_action = tampered_action[batch_idx, agent_idx]
                available_indices = available_indices[available_indices != current_action]
                if len(available_indices) > 0:
                    new_action = available_indices[th.randint(0, len(available_indices), (1,)).item()]
                    tampered_action[batch_idx, agent_idx] = new_action

        return tampered_action
    
    
    
    
    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)
        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions= self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                     bs=envs_not_terminated, test_mode=test_mode)
            actionsNoObsNoise= self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                     bs=envs_not_terminated, test_mode=test_mode, noise=False)

            tamper_avail_actions = self.batch["avail_actions"][:, self.t]
            actionsNoObsNoise_real = actionsNoObsNoise
            prob_tamperAction = self.args.action_tamper
            n_agents_to_tamper = self.args.n_agents_to_tamper
            actionsNoObsNoise_tampered = self.tamper_action(tamper_avail_actions[envs_not_terminated], actionsNoObsNoise, n_agents_to_tamper, prob_tamperAction)
            actions = self.tamper_action(tamper_avail_actions[envs_not_terminated], actions, n_agents_to_tamper, prob_tamperAction)
            cpu_actions = actions.to("cpu").numpy()
            cpu_actionsNoObsNoise = actionsNoObsNoise.to("cpu").numpy()
            cpu_actionsNoObsNoise_real = actionsNoObsNoise_real.to("cpu").numpy()

            actions_chosen = {
                "actions": np.expand_dims(cpu_actions, axis=1),
            }
            actionsNoObsNoise_chosen = {
                "actionsNoObsNoise": np.expand_dims(cpu_actionsNoObsNoise, axis=1),
            }
            actionsNoObsNoise_real_chosen = {
                "actionsNoObsNoise_real": np.expand_dims(cpu_actionsNoObsNoise_real, axis=1),
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.batch.update(actionsNoObsNoise_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.batch.update(actionsNoObsNoise_real_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", [cpu_actions[action_idx], self.args.obs_component, self.args.obs_tamper]))
                    action_idx += 1  # actions is not a list over every env

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [], 
                "obs_real":[] 
            }
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["obs_real"].append(data["obs_real"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch
        # return clear_no_reward_sub_trajectory(self.batch)

    def save_replay(self):
        print("----------------------------Replay----------------------------")
        if self.args.save_replay:
            for parent_conn in self.parent_conns:
                parent_conn.send(("save_replay", None))
            for parent_conn in self.parent_conns:
                _ = parent_conn.recv()

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_min", np.min(returns), self.t_env)
        self.logger.log_stat(prefix + "return_max", np.max(returns), self.t_env)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
        print(self.logger.stats)



def tamper_obs(obs, prob, noise, obs_component):
    tampered_obs = []
    component_lengths = []
    offsets = [0]
    for comp in obs_component:
        if isinstance(comp, (int, float)):
            length = comp
        elif isinstance(comp, (tuple, list)):
            length = np.prod(comp)
        else:
            raise ValueError(f"Unknown component type in obs_component: {comp}")
        component_lengths.append(length)
        offsets.append(offsets[-1] + length)
    offsets.pop()
    
    for arr in obs:
        if np.random.rand() < prob:
            is_all_zeros = all(x == 0 for x in arr)
            if is_all_zeros:
                tampered_obs.append(arr)
            else:
                tampered_arr = arr.copy()
                parts = []
                for i, length in enumerate(component_lengths):
                    start = offsets[i]
                    end = start + length
                    parts.append(tampered_arr[start:end])

                for i, part in enumerate(parts):
                    comp = obs_component[i]
                    if isinstance(comp, int):
                        if i == 0:  # move_feats
                            for j in range(len(part)):
                                if np.random.rand() < noise:
                                    part[j] = 1 - part[j] if part[j] in [0, 1] else part[j]
                        elif i == len(parts) - 1:  # own_feats
                            gaussian_noise = np.random.normal(loc=0.0, scale=noise)
                            part[0] = np.clip(part[0] + gaussian_noise, 0, 1)
                            current_one_idx = None
                            for idx in range(1, len(part)):
                                if part[idx] == 1.0:
                                    current_one_idx = idx - 1
                                    break
                            possible_indices = list(range(len(part) - 1))
                            if current_one_idx is not None and len(possible_indices) > 1:
                                possible_indices.remove(current_one_idx)
                            new_one_idx = np.random.choice(possible_indices)
                            part[1:] = 0.0
                            part[1 + new_one_idx] = 1.0
                    else:  # enemy_feats, ally_feats
                        group_size = comp[-1] if isinstance(comp, (tuple, list)) else 8
                        for j in range(0, len(part), group_size):
                            chunk = part[j:j + group_size]
                            if len(chunk) == group_size:
                                chunk[0] = 1.0 if chunk[0] == 0.0 else 0.0
                                for k in range(1, min(5, group_size)):
                                    gaussian_noise = np.random.normal(loc=0.0, scale=noise)
                                    chunk[k] = np.clip(chunk[k] + gaussian_noise, -1, 1)
                                if group_size > 5:
                                    current_one_idx = None
                                    for idx in range(5, group_size):
                                        if chunk[idx] == 1.0:
                                            current_one_idx = idx - 5
                                            break
                                    possible_indices = list(range(group_size - 5))
                                    if current_one_idx is not None and len(possible_indices) > 1:
                                        possible_indices.remove(current_one_idx)
                                    new_one_idx = np.random.choice(possible_indices)
                                    chunk[5:] = 0.0
                                    chunk[5 + new_one_idx] = 1.0
                            part[j:j + group_size] = chunk
                
                tampered_arr = np.concatenate(parts)
                tampered_obs.append(tampered_arr)
        else:
            tampered_obs.append(arr)
    
    return tampered_obs


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data[0]
            OBS_component = data[1]
            OBStamper_prob = data[2][0]
            OBStamper_noise = data[2][1]
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            obs_real = obs
            obs = tamper_obs(obs, OBStamper_prob, OBStamper_noise, OBS_component)
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "obs_real": obs_real,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
                "obs_real": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_replay":
            remote.send(env.save_replay())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
