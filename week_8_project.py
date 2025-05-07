"""
Catch the Apple – A simple PyGame-based environment for RL projects

Description:
  In this game, a paddle is controlled by an agent at the bottom of the screen.
  Apples fall from the top, and the goal is to catch as many apples as possible.
  The environment returns a reward of +1 for every apple caught and -1 for every apple missed.
  The game ends after a fixed number of frames (episodes) or when a specific number of misses is reached.

Agent Actions:
  0 – Move Left
  1 – Stay still
  2 – Move Right

Instructions for Students:
  1. Review the environment code and the provided RL agent scaffold.
  2. Implement your reinforcement learning algorithm (e.g., Q-learning or DQN) to choose actions based on state information.
  3. Adjust hyperparameters (learning rate, exploration vs. exploitation) and observe training performance.
  4. Use the state information (e.g., paddle x-position, apple x and y positions) as input for your model.
  5. Experiment with different reward structures and terminal conditions.

Requirements:
  pip install pygame

Run this file in PyCharm or any other Python IDE.
"""

import pygame
import random
import sys

# Game settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 30

# Game colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Paddle settings
PADDLE_WIDTH = 80
PADDLE_HEIGHT = 10
PADDLE_Y = SCREEN_HEIGHT - 40
PADDLE_SPEED = 20

# Apple settings
APPLE_SIZE = 20
APPLE_SPEED = 5

# Episode termination settings
MAX_FRAMES = 600  # Approx 20 seconds if FPS=30
MAX_MISSES = 5


class CatchAppleEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Catch the Apple - RL Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.paddle_x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2
        self.apple_x = random.randint(0, SCREEN_WIDTH - APPLE_SIZE)
        self.apple_y = 0
        self.frames = 0
        self.misses = 0
        self.score = 0
        state = self._get_state()
        return state

    def _get_state(self):
        """
        Returns the current state.
        You can modify this to include additional features.
        State: (paddle_x, apple_x, apple_y)
        """
        return (self.paddle_x, self.apple_x, self.apple_y)

    def step(self, action):
        """
        Applies the given action and advances the game by one frame.

        Actions:
          0: Move Left
          1: Stay Still
          2: Move Right

        Returns:
          state: New state after taking the action.
          reward: Reward received.
          done: Flag indicating whether the episode is finished.
        """
        # Process action
        if action == 0:
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:
            self.paddle_x += PADDLE_SPEED
        # Clamp paddle within screen bounds
        self.paddle_x = max(0, min(SCREEN_WIDTH - PADDLE_WIDTH, self.paddle_x))

        # Move apple
        self.apple_y += APPLE_SPEED

        reward = -1  # small negative reward for each frame

        # Check if apple is caught or missed
        if self.apple_y + APPLE_SIZE >= PADDLE_Y:
            # Check if apple is within paddle horizontal range
            if self.paddle_x <= self.apple_x <= self.paddle_x + PADDLE_WIDTH:
                reward = 10  # reward for catching apple
                self.score += 1
            else:
                reward = -10  # penalty for missing apple
                self.misses += 1
            # Reset apple position after catch/miss
            self.apple_x = random.randint(0, SCREEN_WIDTH - APPLE_SIZE)
            self.apple_y = 0

        # Increase frame count
        self.frames += 1

        # Check if episode is done
        done = False
        if self.frames >= MAX_FRAMES or self.misses >= MAX_MISSES:
            done = True

        next_state = self._get_state()
        return next_state, reward, done

    def render(self):
        """Renders the current game state."""
        self.screen.fill(WHITE)
        # Draw paddle
        pygame.draw.rect(self.screen, BLACK, (self.paddle_x, PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT))
        # Draw apple
        pygame.draw.circle(self.screen, RED, (self.apple_x + APPLE_SIZE // 2, self.apple_y + APPLE_SIZE // 2),
                           APPLE_SIZE // 2)
        # Draw score and misses
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        miss_text = self.font.render(f"Misses: {self.misses}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(miss_text, (10, 30))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


# =============================
# Sample RL Agent Scaffold
# =============================
class RandomAgent:
    """
    A simple agent that randomly chooses an action.
    Replace or extend this with your RL algorithm.
    """

    def select_action(self, state):
        return random.choice([0, 1, 2])


def run_episode(env, agent, render=True):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # For simplicity, we use a random agent here.
        # Replace this with your action selection mechanism from your RL model.
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if render:
            env.render()
        # Process PyGame events (to allow window closing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                env.close()
                sys.exit()
    return total_reward


if __name__ == "__main__":
    # This block runs a few episodes using the RandomAgent.
    env = CatchAppleEnv()
    agent = RandomAgent()
    episodes = 5
    for ep in range(episodes):
        total = run_episode(env, agent, render=True)
        print(f"Episode {ep + 1}: Total Reward = {total}")
    env.close()

    # Instructions for further work:
    # 1. Replace the RandomAgent with an RL agent you develop (e.g., Q-learning, DQN).
    # 2. Use the state (paddle_x, apple_x, apple_y) as input and update your model based on rewards.
    # 3. Experiment with reward shaping, state representation, and hyperparameters.
    # 4. Analyze and discuss your results in relation to literature on RL and human-AI interaction.
