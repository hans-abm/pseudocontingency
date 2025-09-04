from mesa import Agent
import random
import numpy as np


class LearningAgent(Agent):
    """Agent that learns health-taste relationships through pseudocontingencies."""
    
    def __init__(self, unique_id, model, row, ht_belief, BMI):
        super().__init__(unique_id, model)
        self.row = row
        self.ptype = None
        self.food_observed = []
        self.ht_belief = ht_belief
        self.current_phi = 0.0
        self.BMI = BMI
        
        # Model parameters
        self.bias_strength = model.bias_strength
        self.learning_rate = model.learning_rate
        self.noise = model.noise
        
        # Track frequencies
        self.health_counts = {"Healthy": 0, "Unhealthy": 0}
        self.taste_counts = {"Tasty": 0, "NotTasty": 0}
        self.joint_counts = {"HT": 0, "HN": 0, "UT": 0, "UN": 0}

    def move(self):
        """Move agent to next cell in row."""
        x, y = self.pos
        new_x = (x + 1) % self.model.grid.width
        self.model.grid.move_agent(self, (new_x, y))

    def observe(self):
        """Observe current food type and update beliefs."""
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        self.ptype = next((content.type for content in cell_contents 
                          if isinstance(content, Patch)), None)
        
        options = ["HT", "HN", "UT", "UN"]
        
        if self.ptype:
            if random.random() < self.noise:
                observed_type = random.choice(options)
            else:
                observed_type = self.ptype

            self.food_observed.append(observed_type)
            self._update_counts(observed_type)
            self._update_pc_belief()

    def _update_counts(self, food_type):
        """Update marginal and joint frequency counts."""
        self.joint_counts[food_type] += 1
        
        if food_type[0] == 'H':
            self.health_counts["Healthy"] += 1
        else:
            self.health_counts["Unhealthy"] += 1
            
        if food_type[1] == 'T':
            self.taste_counts["Tasty"] += 1
        else:
            self.taste_counts["NotTasty"] += 1

    def _update_pc_belief(self):
        """Update beliefs using pseudocontingency mechanism."""
        if not self.food_observed:
            return
        
        total_obs = sum(self.joint_counts.values())
        if total_obs == 0:
            return
            
        # Construct contingency table
        contingency_table = np.array([
            [self.joint_counts["HT"], self.joint_counts["HN"]],
            [self.joint_counts["UT"], self.joint_counts["UN"]]
        ])
        
        # Calculate actual correlation
        actual_correlation = self._calculate_phi(contingency_table)
        self.current_phi = actual_correlation
        
        # Calculate pseudocontingency
        health_prop = self.health_counts["Healthy"] / total_obs
        taste_prop = self.taste_counts["Tasty"] / total_obs
        
        health_skew = health_prop - 0.5
        taste_skew = taste_prop - 0.5
        pc_effect = 4 * health_skew * taste_skew
        
        # Update belief
        new_belief = (1 - self.bias_strength) * actual_correlation + self.bias_strength * pc_effect
        new_belief = (new_belief + 1) / 2
        
        self.ht_belief = (1 - self.learning_rate) * self.ht_belief + self.learning_rate * new_belief
        
    def _calculate_phi(self, table):
        """Calculate phi coefficient from 2x2 contingency table."""
        if table.sum() == 0 or table[0].sum() == 0 or table[1].sum() == 0 or table[:,0].sum() == 0 or table[:,1].sum() == 0:
            return 0
            
        x = table[0,0] * table[1,1] - table[0,1] * table[1,0]
        y = table[0].sum() * table[1].sum() * table[:,0].sum() * table[:,1].sum()
        
        if y <= 0:
            return 0
            
        phi = x / np.sqrt(y)
        return phi

    def step(self):
        """Execute one step of the agent."""
        self.move()
        self.observe()


class Patch(Agent):
    """Food patch with specific type."""
    
    COLORS = {
        "HT": "green",   # Healthy and Tasty
        "HN": "blue",    # Healthy and Not Tasty
        "UT": "red",     # Unhealthy and Tasty
        "UN": "purple"   # Unhealthy and Not Tasty
    }
    
    def __init__(self, unique_id, model, patch_type):
        super().__init__(unique_id, model)
        self.type = patch_type
        
    def get_color(self):
        return self.COLORS.get(self.type, "white")