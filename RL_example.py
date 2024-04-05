import gymnasium as gym

env = gym.make('CartPole-v1',render_mode="human")
print(env.action_space.shape)
#env = gym.make("FrozenLake-v1",render_mode="human", is_slippery=False)
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation) # position, velocity, angular position, angular velocity
    #if terminated or truncated:
    #    observation, info = env.reset()

    if truncated or abs(observation[0]>2.4): # kill after time or after position > 2.4
        observation, info = env.reset() 

    

env.close()