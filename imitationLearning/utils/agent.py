import torch

import utils
from .other import device
from model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False, obs_shape=None,train=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        # obs_space_tmp, self.preprocess_obss_tmp = utils.get_obss_preprocessor_tmp(obs_space)
        if obs_shape is not None:
            obs_space['image'] = obs_shape
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
        if not train:
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        if not train:
            self.acmodel.eval()
            if hasattr(self.preprocess_obss, "vocab"):
                self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

    def reset_memory(self):
        self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

    def action_dist(self, obs):
        preprocessed_obss = self.preprocess_obss([obs], device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)
        return dist

    def action_prob(self, obs):
        preprocessed_obss = self.preprocess_obss(obs, device=device)

        if self.acmodel.recurrent:
            dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
        else:
            dist, _ = self.acmodel(preprocessed_obss)
        return dist.logits
