import random
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import LearningAgent, Patch

class LearningModel(Model):
    def __init__(self, N, width, height, env_type, bias_strength, learning_rate, noise=0.05, seed=None):
        super().__init__(seed=seed)

        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)

        self.env_type = env_type
        self.bias_strength = bias_strength
        self.learning_rate = learning_rate
        self.noise = noise

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.create_agents()
        self.create_patches()

        self.datacollector = DataCollector(
            agent_reporters={"ht": "ht_belief", "phi": "current_phi"}
        )

    def create_agents(self):
        for i in range(self.num_agents):
            ht0 = np.random.normal(0.5, 0.15)
            a = LearningAgent(i, self, i, ht0)
            self.grid.place_agent(a, (0, i))
            self.schedule.add(a)

    def create_patches(self):
        props = self.get_env_props(self.env_type)
        for x in range(self.grid.width):
            self.fill_column(x, props)

    def get_env_props(self, env):
        if env == "balanced":
            return {"HT": 0.25, "HN": 0.25, "UT": 0.25, "UN": 0.25}
        if env == "healthy":
            return {"HT": 0.50, "HN": 0.25, "UT": 0.25, "UN": 0.00}
        if env == "unhealthy":
            return {"HT": 0.25, "HN": 0.00, "UT": 0.50, "UN": 0.25}
        if env == "healthy_b":
            return {"HT": 0.4444, "HN": 0.2222, "UT": 0.2222, "UN": 0.1111}
        if env == "unhealthy_b":
            return {"HT": 0.2222, "HN": 0.1111, "UT": 0.4444, "UN": 0.2222}
        return {"HT": 0.25, "HN": 0.25, "UT": 0.25, "UN": 0.25}

    def fill_column(self, x, props):
        types = []
        for t, p in props.items():
            types.extend([t] * int(self.grid.height * p))
        while len(types) < self.grid.height:
            types.append(random.choice(list(props.keys())))
        random.shuffle(types)

        for y in range(self.grid.height):
            patch = Patch(f"patch_{x}_{y}", self, types[y])
            self.grid.place_agent(patch, (x, y))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()