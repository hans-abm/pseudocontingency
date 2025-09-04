import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

from agent import LearningAgent, Patch

class LearningModel(Model):
    """Model of agents learning about food healthiness through pseudocontingencies."""
    
    def __init__(self, N=156, width=100, height=156, bias_strength=0.5, 
                 env_type='unhealthy', noise=0.05, seed=None, learning_rate=0.9):
        super().__init__(seed=seed)
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)
        self.env_type = env_type
        self.noise = noise
        self.bias_strength = bias_strength
        self.learning_rate = learning_rate

        if seed is not None:
            random.seed(seed)

        self._create_agents()
        self._distribute_patches(self.env_type)
        
        self.datacollector = DataCollector(
            agent_reporters={
                'ht': 'ht_belief',
                'phi': 'current_phi'
            }
        )

    def _get_environment(self, env_type):
        """Get food type proportions based on environment type."""
        if env_type == 'balanced':
            return {'HT': 0.25, 'HN': 0.25, 'UT': 0.25, 'UN': 0.25}
        elif env_type == 'healthy':
            return {'HT': 0.50, 'HN': 0.25, 'UT': 0.25, 'UN': 0.00}
        elif env_type == 'unhealthy':
            return {'HT': 0.25, 'HN': 0.00, 'UT': 0.50, 'UN': 0.25}
        elif env_type == 'healthy_b':
            return {'HT': 0.4444, 'HN': 0.2222, 'UT': 0.2222, 'UN': 0.1111}
        elif env_type == 'unhealthy_b':
            return {'HT': 0.2222, 'HN': 0.1111, 'UT': 0.4444, 'UN': 0.2222}

    def _distribute_patches(self, env_type):
        """Distribute patches across grid based on environment type."""
        proportions = self._get_environment(env_type)
        for x in range(self.grid.width):
            self._fill_column(x, proportions)

    def _fill_column(self, x, proportions):
        """Fill a column with patches based on proportions."""
        patch_types = []
        for ptype, prop in proportions.items():
            num_patches = int(self.grid.height * prop)
            patch_types.extend([ptype] * num_patches)
        while len(patch_types) < self.grid.height:
            patch_types.append(random.choice(list(proportions.keys())))
        
        random.shuffle(patch_types)
        for y in range(self.grid.height):
            patch = Patch(f'patch_{x}_{y}', self, patch_types[y])
            self.grid.place_agent(patch, (x, y))

    def _create_agents(self):
        """Create agents from data."""
        df = pd.read_csv('data/Study2_data_processed.csv', delimiter=";")
        df = df.map(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

        for i, row in df.iterrows():
            if i < self.num_agents:
                agent = LearningAgent(
                    unique_id=i, 
                    model=self, 
                    row=i, 
                    ht_belief=np.random.normal(0.5, 0.15),
                    BMI=row['BMI']
                )

                self.grid.place_agent(agent, (0, i))
                self.schedule.add(agent)

    def step(self):
        """Execute one model step."""
        self.datacollector.collect(self)
        self.schedule.step()

    def plot_env(self):
        """Visualize the environment grid."""
        grid_matrix = []
        for y in range(self.grid.height):
            row = []
            for x in range(self.grid.width):
                cell_content = self.grid.get_cell_list_contents([(x, y)])
                patch = next((obj for obj in cell_content 
                            if isinstance(obj, Patch)), None)
                row.append(patch.get_color() if patch else "white")
            grid_matrix.append(row)

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['red', 'blue', 'purple', 'green']
        cmap = mcolors.ListedColormap(colors)
        bounds = list(range(len(colors) + 1))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        matrix = [[colors.index(color) if color in colors else -1 
                  for color in row] for row in grid_matrix]
        ax.imshow(matrix, cmap=cmap, norm=norm)
        plt.show()