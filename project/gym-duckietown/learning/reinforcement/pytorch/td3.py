import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpg import ActorCNN, CriticCNN, ActorDense, CriticDense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add additional critic and target
# add clipped noise to
# policy is learned differently

class TD3(object):
    def __init__(self, state_dim, action_dim, min_action, max_action, target_action_noise, clip_range, net_type):
        super(TD3, self).__init__()
        print("Starting TD3 init")
        assert net_type in ["cnn", "dense"]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.target_action_noise = target_action_noise
        self.clip_range = clip_range

        if net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorDense(state_dim, action_dim, max_action).to(device)
        else:
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action).to(device)
            self.actor_target = ActorCNN(action_dim, max_action).to(device)

        print("Initialized Actor")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        print("Initialized Target+Opt [Actor]")
        if net_type == "dense":
            self.critic1 = CriticDense(state_dim, action_dim).to(device)
            self.critic2 = CriticDense(state_dim, action_dim).to(device)
            self.critic_target1 = CriticDense(state_dim, action_dim).to(device)
            self.critic_target2 = CriticDense(state_dim, action_dim).to(device)
        else:
            self.critic1 = CriticCNN(action_dim).to(device)
            self.critic2 = CriticCNN(action_dim).to(device)
            self.critic_target1 = CriticCNN(action_dim).to(device)
            self.critic_target2 = CriticCNN(action_dim).to(device)
        print("Initialized Critic")
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters())
        print("Initialized Target+Opt [Critic]")

    def predict(self, state):

        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            noise = np.random.normal(
                0,
                self.target_action_noise,
                size=self.action_dim).clip(-self.clip_range, self.clip_range)
            target_action = (self.actor_target(next_state) + noise).clip(self.min_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic_target1(next_state, target_action)
            target_Q2 = self.critic_target2(next_state, target_action)
            target_Q = reward + (done * discount * torch.min(target_Q1, target_Q2)).detach()

            # Get current Q estimate
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)

            # Compute critic loss
            critic_loss1 = F.mse_loss(current_Q1, target_Q)
            critic_loss2 = F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer1.zero_grad()
            self.critic_optimizer2.zero_grad()
            critic_loss1.backward(retain_graph=True)
            critic_loss2.backward()
            self.critic_optimizer1.step()
            self.critic_optimizer2.step()

            # Compute actor loss
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic1.state_dict(), '{}/{}_critic1.pth'.format(directory, filename))
        torch.save(self.critic2.state_dict(), '{}/{}_critic2.pth'.format(directory, filename))
        print("Saved Critic")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic1.load_state_dict(torch.load('{}/{}_critic1.pth'.format(directory, filename), map_location=device))
        self.critic2.load_state_dict(torch.load('{}/{}_critic2.pth'.format(directory, filename), map_location=device))
