import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_

class HyperLayers(nn.Module):
    def __init__(self, input_dim, hypernet_embed, n_agents, embed_dim):
        super(HyperLayers, self).__init__()
        self.w1 = nn.Sequential(
            nn.Linear(input_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, n_agents * embed_dim)
        )
        self.b1 = nn.Sequential(
            nn.Linear(input_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, embed_dim)
        )
        self.w2 = nn.Sequential(
            nn.Linear(input_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, embed_dim)
        )
        self.b2 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        
        for i in range(self.n_heads):  # multi-head attention
            selector_nn = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(self.hypernet_embed, self.embed_dim, bias=False))
            self.selector_extractors.append(selector_nn)  # query
            if self.args.nonlinear:  # add qs
                self.key_extractors.append(nn.Linear(self.obs_dim + 1, self.embed_dim, bias=False))  # key
            else:
                self.key_extractors.append(nn.Linear(self.obs_dim, self.embed_dim, bias=False))  # key
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                              nn.ReLU(),
                                              nn.Linear(self.hypernet_embed, self.n_heads))
        if self.args.state_bias:
            # V(s) instead of a bias for the last layers
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

    def forward(self, x):
        return x

class MyAlgMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(MyAlgMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_groups = args.n_groups
        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed
        self.state_ally_feats_size = args.state_ally_feats_size
        self.state_enemy_feats_size = args.state_enemy_feats_size
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        self.n_heads = args.n_heads
        
        self.abs = abs
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        assert self.qmix_pos_func == "abs"

        self.groups = nn.ModuleList([
            MultiHeadAttention(args) for _ in range(self.n_groups)
        ])
        self.hyper = HyperLayers(self.input_dim, self.hypernet_embed, self.n_groups, self.embed_dim)

    def get_masks(self, ally_states):
        if self.n_groups == 1:
            return [th.ones(ally_states.shape[0], ally_states.shape[1], self.n_agents).to(ally_states.device)]
        # 只取初始时间的类型（无人死亡）
        type_matrix = ally_states[:, :, :, -self.n_groups:]
        types = [th.where(type_matrix[:, 0, :, i] == 1) for i in range(self.n_groups)]
        masks = [th.zeros(ally_states.shape[0], ally_states.shape[1], self.n_agents).to(ally_states.device) for _ in range(self.n_groups)]
        for i in range(self.n_groups):
            masks[i][types[i][0], :, types[i][1]] = 1
        return masks

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)

    def forward(self, qvals, states, obs):

        b, t, _ = qvals.size()
        
        split_indices = [self.n_agents * self.state_ally_feats_size, self.n_agents * self.state_ally_feats_size + self.n_enemies * self.state_enemy_feats_size]
        ally_states, enemy_states, last_actions = np.split(states, split_indices, axis=2)
        ally_states = ally_states.reshape(b, t, self.n_agents, self.state_ally_feats_size)
        
        masks = self.get_masks(ally_states)  # [b, t, n_agents]
        states = states.reshape(-1, self.state_dim)
        
        group_qvals = []
        attend_mag_regs = []
        head_entropies = []
        for i, group in enumerate(self.groups):
            masked_qvals = (qvals * masks[i]).reshape(b * t, 1, self.n_agents)
            masked_obs = (obs * masks[i].unsqueeze(-1)).reshape(self.n_agents, b * t, self.obs_dim)
            
            all_head_selectors = [sel_ext(states) for sel_ext in group.selector_extractors]  # (head_num, b*t, embed_dim)
            all_head_keys = [[k_ext(enc) for enc in masked_obs] for k_ext in group.key_extractors]  # (head_num, agent_num, b*t, embed_dim)
            
            # calculate attention per head
            head_qs = []
            head_attend_logits = []
            head_attend_weights = []
            for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):\
                # curr_head_keys: (agent_num, b*t, embed_dim)
                # curr_head_selector: (b*t, embed_dim)
                # (b*t, 1, embed_dim) * (b*t, embed_dim, agent_num)
                attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                          th.stack(curr_head_keys).permute(1, 2, 0))
                # attend_logits: (b*t, 1, agent_num)
                scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
                attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (b*t, 1, agent_num)
                # (b*t, 1, agent_num) * (b*t, 1, agent_num)
                head_q = (masked_qvals * attend_weights).sum(dim=2)
                head_qs.append(head_q)
                head_attend_logits.append(attend_logits)
                head_attend_weights.append(attend_weights)
            
            # regularize magnitude of attention logits
            attend_mag_reg = self.args.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
            head_entropie = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]
            attend_mag_regs.append(attend_mag_reg)
            head_entropies.append(head_entropie)
            
            if self.args.state_bias:
                # State-dependent bias
                v = group.V(states).view(-1, 1)  # (b*t, 1)
                # head_qs: [head_num, b*t, 1]
                if self.args.weighted_head:
                    w_head = th.abs(group.hyper_w_head(states))  # w_head: (b*t, head_num)
                    w_head = w_head.view(-1, self.n_heads, 1)  # w_head: (b*t, head_num, 1)
                    y = th.stack(head_qs).permute(1, 0, 2)  # head_qs: (head_num, b*t, 1); y: (b*t, head_num, 1)
                    y = (w_head * y).sum(dim=1) + v  # y: (b*t, 1)
                else:
                    y = th.stack(head_qs).sum(dim=0) + v  # y: (b*t, 1)
            else:
                if self.args.weighted_head:
                    w_head = th.abs(group.hyper_w_head(states))  # w_head: (b*t, head_num)
                    w_head = w_head.view(-1, self.n_heads, 1)  # w_head: (b*t, head_num, 1)
                    y = th.stack(head_qs).permute(1, 0, 2)  # head_qs: (head_num, b*t, 1); y: (b*t, head_num, 1)
                    y = (w_head * y).sum(dim=1)  # y: (b*t, 1)
                else:
                    y = th.stack(head_qs).sum(dim=0)  # y: (b*t, 1)
            # Reshape and return
            q_group = y.view(b * t, 1, 1)
            group_qvals.append(q_group)
        # Combine group Q-values
        group_qvals = th.cat(group_qvals, dim=2)  # (b*t, 1, n_groups)
        w1 = self.hyper.w1(states).view(-1, self.n_groups, self.embed_dim) # (b*t, n_groups, emb)
        b1 = self.hyper.b1(states).view(-1, 1, self.embed_dim)
        w2 = self.hyper.w2(states).view(-1, self.embed_dim, 1) # (b*t, emb, 1)
        b2 = self.hyper.b2(states).view(-1, 1, 1)
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        hidden = F.elu(th.matmul(group_qvals, w1) + b1) # (b*t, 1, emb)
        qtot = th.matmul(hidden, w2) + b2 # (b*t, 1, 1)
        
        return qtot.view(b, t, -1)