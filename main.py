import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from constants import actions


class SmartPolicyNetwork(nn.Module): #neural network class for learning from policy
    def __init__(self, state_dim, action_dim): #state_dim is input features, represents the state of the environment; action_dim is the possible actions for the agent
        super(SmartPolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x): #information about how the input data in the neural network
        return self.network(x)


class MDPCellularSimulation:
    def __init__(self, width=20, height=20):  
        self.width = width
        self.height = height
        self.cell = {
            "position": (random.randint(0, width - 1), random.randint(0, height - 1)),
            "health": 10,
            "money": 0,
        } #randomly initialise the cell position within the boundaries.
        self.visited_blocks = set() #set of visited grid blocks
        self.actions = [
            (0, 1),  # Down
            (0, -1),  # Up
            (1, 0),  # Right
            (-1, 0),  # Left
            (0, 0),  # Stay
        ] #list of actions for the cell
        self.food_blocks = set(self.generate_resources(75)) #generate food blocks in the cell
        self.money_blocks = set(self.generate_resources(50))   #generate money blocks in the cell
        self.global_knowledge = {
            "survival_rate": 0.5,
            "total_food_collected": 0,
            "total_money_collected": 0,
        } #dictinoary to tracks and store the global strategies and stores the information that by actions or episodes.
        self.device = torch.device("cpu") #becuase mac doesnt hace gpu
        self.policy_network = SmartPolicyNetwork(
            state_dim=9, action_dim=len(self.actions)
        ).to(self.device) #initialsing the policy network.
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.0001) #optimiszer to update the parameters during training to for better decision making.

    def generate_resources(self, count): #function to generate resources
        resources = set()
        while len(resources) < count:
            pos = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            resources.add(pos)
        return resources

    def get_state_tensor(self): #function to construct the state reporesentation of the current environment, which is used as an input for the PolicyNetwork
        position = self.cell["position"]  # get the agents position in the grid
        nearest_food_distance = self._distance_to_nearest(self.food_blocks, position) # get the distance to the nearest food
        nearest_money_distance = self._distance_to_nearest(self.money_blocks, position) # for money
        state_features = [
            position[0] / self.width, #normalising the x coordinate
            position[1] / self.height, #normalising the x coordinate
            self.cell["health"] / 10, #
            self.cell["money"] / 10,
            len(self.food_blocks) / 50,
            len(self.money_blocks) / 50,
            self.global_knowledge["survival_rate"], # current survival rate of the agent
            nearest_food_distance,  # Distance to nearest food
            nearest_money_distance,  # Distance to nearest money
        ]
        return torch.tensor(state_features, dtype=torch.float32)

    def validate_position(self, position):
        x, y = position
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return x, y

    def calculate_reward(self, new_position, method=None):
        reward = 0

        # Strong penalty for dying
        if self.cell["health"] <= 0:
            reward -= 50
        # Rewards for collecting food or money
        if new_position in self.food_blocks:
            reward += 50 if method == "chosen" else 25
            self.global_knowledge["total_food_collected"] += 1
            self.food_blocks.remove(new_position)
        if new_position in self.money_blocks:
            reward += 30 if method == "chosen" else 15
            self.global_knowledge["total_money_collected"] += 1
            self.money_blocks.remove(new_position)
        # Reward for maintaining health
        reward += self.cell["health"] / 10
        # Encourage exploration (reward for visiting new blocks)
        if new_position not in self.visited_blocks:
            reward += 5 if method == "chosen" else 2  # Reward for exploration
            self.visited_blocks.add(new_position)

        # Slight penalty for being idle (staying in place)
        if new_position == self.cell["position"]:
            reward -= 10 if method == "chosen" else 5

        # Encourage movement towards resource-rich areas
        distance_to_food = self._distance_to_nearest(self.food_blocks, new_position)
        distance_to_money = self._distance_to_nearest(self.money_blocks, new_position)

        if distance_to_food is not None:
            reward += max(0, 5 - distance_to_food)  # Reward closer proximity to food

        if distance_to_money is not None:
            reward += max(0, 3 - distance_to_money)  # Reward closer proximity to money

        return reward

    def _distance_to_nearest(self, resource_blocks, position):
        if not resource_blocks:
            return None
        return min(
            np.sqrt((position[0] - x) ** 2 + (position[1] - y) ** 2)
            for x, y in resource_blocks
        )

    def choose_action(self, epsilon=0.1):
        if random.random() < epsilon:  # Exploration
            return random.choice(self.actions), "random"
        else:  # Exploitation
            state_tensor = self.get_state_tensor().to(self.device)
            with torch.no_grad():
                action_probs = self.policy_network(state_tensor)

            # Clamp probabilities to avoid numerical instability
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)

            # Sample an action based on the policy probabilities
            action_index = torch.multinomial(action_probs, 1).item()
            return self.actions[action_index], "chosen"

    def step(self, action, method=None): #steps for the agent and calculating the reward according to the step taken.
        x, y = self.cell["position"]
        dx, dy = action
        new_position = (x + dx, y + dy)
        new_position = self.validate_position(new_position)
        self.cell["health"] -= 1
        self.cell["position"] = new_position
        if new_position in self.food_blocks:
            self.cell["health"] += 5
        if new_position in self.money_blocks:
            self.cell["money"] += 1
        reward = self.calculate_reward(new_position, method)
        return reward

    def train(self, episodes=5000, epsilon=1.0, epsilon_decay=0.999):
        reward_history = []
        for episode in range(episodes):
            self.reset_environment()
            trajectory = []
            total_reward = 0

            for _ in range(100):
                action, method = self.choose_action(epsilon)
                print(actions[action])
                reward = self.step(action, method)
                log_prob = torch.log(
                    torch.clamp(
                        self.policy_network(self.get_state_tensor())[
                            self.actions.index(action)
                        ],
                        min=1e-8,
                        max=1.0,
                    )
                )
                trajectory.append((log_prob, reward))
                total_reward += reward

                if self.cell["health"] <= 0:
                    break
            self.update_policy(trajectory)
            reward_history.append(total_reward)
            epsilon = max(0.1, epsilon * epsilon_decay)
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    def update_policy(self, trajectory):
        returns = []
        G = 0
        for _, reward in reversed(trajectory):
            G = reward + 0.9 * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for (log_prob, _), G in zip(trajectory, returns):
            policy_loss.append(-log_prob * G)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).mean()
        policy_loss.backward()
        print(self.optimizer.step())

    def save_model(self, path="policy_network.pth"):
        torch.save(self.policy_network.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path="policy_network.pth"):
        self.policy_network.load_state_dict(torch.load(path))
        self.policy_network.to(self.device)
        print(f"Model loaded from {path}.")

    def reset_environment(self):
        self.cell = {
            "position": (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            ),
            "health": 10,
            "money": 0,
        }
        self.food_blocks = set(self.generate_resources(75))
        self.money_blocks = set(self.generate_resources(50))
        self.visited_blocks.clear()


def main():
    pygame.init()
    WIDTH, HEIGHT, TILE_SIZE = 800, 800, 40
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cell Simulation")
    sim = MDPCellularSimulation()

    print("Choose a mode:")
    print("1: Observe behavior before training")
    print("2: Train the model")
    print("3: Load and observe trained behavior")
    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "2":
        print("Training in progress...")
        sim.train()
        sim.save_model()
    elif mode == "3":
        sim.load_model()
    else:
        print("Observing random behavior...")

    clock = pygame.time.Clock()
    running, play_simulation = True, False
    while running:
        screen.fill((128, 128, 128))
        for food in sim.food_blocks:
            pygame.draw.rect(
                screen,
                (0, 255, 0),
                (food[0] * TILE_SIZE, food[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE),
            )
        for money in sim.money_blocks:
            pygame.draw.rect(
                screen,
                (255, 215, 0),
                (money[0] * TILE_SIZE, money[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE),
            )
        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (
                sim.cell["position"][0] * TILE_SIZE,
                sim.cell["position"][1] * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            ),
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    play_simulation = not play_simulation
                if event.key == pygame.K_r:
                    sim.reset_environment()

        if play_simulation:
            action, _ = sim.choose_action()
            sim.step(action)
            if sim.cell["health"] <= 0:
                print("The cell has died. Resetting environment...")
                sim.reset_environment()
                play_simulation = False  # Pause simulation after reset
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
