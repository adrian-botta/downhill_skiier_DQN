import matplotlib.pyplot as plt

class rl_metrics_logger:
    def __init__(self):
        self.logging_episode_number = []
        self.logging_time_step_count = []
        self.logging_epsilon_value = []
        self.logging_total_reward = []
        self.logging_replay_loss = []
    
    def store(self, episode, time_step, epsilon, total_reward, model_loss):
        self.logging_episode_number.append(episode)
        self.logging_time_step_count.append(time_step)
        self.logging_epsilon_value.append(epsilon)
        self.logging_total_reward.append(total_reward)
        self.logging_replay_loss.append(model_loss)
    
    def plot_metrics(self):
        #plot loss
        plt.figure(figsize=(10,10))
        plt.plot(self.logging_replay_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        #plot timestep
        plt.figure(figsize=(10,10))
        plt.plot(self.logging_time_step_count)
        plt.title('final timestep')
        plt.ylabel('timestep')
        plt.xlabel('epoch')
        plt.show()
        #plot reward
        plt.figure(figsize=(10,10))
        plt.plot(self.logging_total_reward)
        plt.title('total reward')
        plt.ylabel('reward')
        plt.xlabel('epoch')
        plt.show()