import torch
import gym
import torch.nn.functional as F
import numpy as np
from utils.ActorNet import ActorNet
from utils.Buffer import ReplayBuffer
from utils.CriticNet import CriticNet
from utils.OUActionNoise import OUActionNoise
from utils.utils import plot_reward_curve

class Agent():
    def __init__(self, lr1, lr2, input_dims, tau, n_actions, gamma=0.99, max_size=1000000, fc1_dims=400, fc2_dims=300, batch_size=64):
        self.lr1 = lr1
        self.lr2 = lr2
        self.input_dims = input_dims
        self.tau = tau
        self.n_actions = n_actions
        self.gamma = gamma
        self.max_size = max_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.actor = ActorNet(lr1, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='actor')
        self.critic = CriticNet(lr2, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='critic')
        self.target_actor = ActorNet(lr1, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNet(lr2, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation, explore):
        self.actor.eval()
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        if explore:
            mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
            self.actor.train()
            out = mu_prime.cpu().detach().numpy()[0]
        else:
            self.actor.train()
            out = mu.cpu().detach().numpy()[0]    
        return out
    
    def remember(self, state, action, reward, _state, done):
        self.memory.store_transition(state, action, reward, _state, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return
        states, actions, rewards, _states, done = self.memory.sample_memory(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        _states = torch.tensor(_states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(_states)
        _critic_value = self.target_critic.forward(_states, target_actions)
        critic_value = self.critic.forward(states, actions)
        _critic_value[done] = 0.0
        _critic_value = _critic_value.view(-1)

        target = rewards + self.gamma*_critic_value
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau == None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)


if __name__ == '__main__':
    env_train = gym.make('LunarLanderContinuous-v2')
    agent = Agent(lr1=0.0001, lr2=0.001, input_dims=env_train.observation_space.shape, tau=0.001, batch_size=256, fc1_dims=400, fc2_dims=300, n_actions=env_train.action_space.shape[0])
    n_games = 500
    demo_games = 10
    filename = 'LunarLander_lr1_' + str(agent.lr1) + '_lr2_' + str(agent.lr2) + '_games_' + str(n_games)
    figure_file = 'plots/' + filename + '.png'
    best_score = env_train.reward_range[0]
    score_history = []

##Train##    
    for i in range(n_games):
        explore = True
        observation = env_train.reset()[0]
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation, explore)
            _observation, reward, done, info, _ = env_train.step(action)
            agent.remember(observation, action, reward, _observation, done)
            agent.learn()
            score += reward
            observation = _observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' %score, 'average score %.1f' % avg_score)

    env_train.close()

    x = [i+1 for i in range(n_games)]
    plot_reward_curve(x, score_history, figure_file)

    print('Demoing Model')
##Demo##
    env_test =  gym.make('LunarLanderContinuous-v2', render_mode='human')
    for i in range(demo_games):
        explore = False
        observation = env_test.reset()[0]
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation, explore)
            _observation, reward, done, info, _ = env_test.step(action)
            agent.remember(observation, action, reward, _observation, done)
            agent.learn()
            score += reward
            observation = _observation












