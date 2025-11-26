import random
import numpy as np
from mesa import Agent

class LearningAgent(Agent):
    def __init__(self, unique_id, model, row, ht_belief):
        super().__init__(unique_id, model)
        self.row = row
        self.ptype = None
        self.food_observed = []
        self.ht_belief = ht_belief
        self.current_phi = 0.0

        self.bias_strength = model.bias_strength
        self.learning_rate = model.learning_rate
        self.noise = model.noise

        self.health_counts = {"Healthy": 0, "Unhealthy": 0}
        self.taste_counts = {"Tasty": 0, "NotTasty": 0}
        self.joint_counts = {"HT": 0, "HN": 0, "UT": 0, "UN": 0}

    def move(self):
        x, y = self.pos
        new_x = (x + 1) % self.model.grid.width
        self.model.grid.move_agent(self, (new_x, y))

    def step(self):
        self.move()
        self.observe()

    def observe(self):
        cell = self.model.grid.get_cell_list_contents([self.pos])
        p = next((c.type for c in cell if isinstance(c, Patch)), None)
        if p:
            if random.random() < self.noise:
                p = random.choice(["HT", "HN", "UT", "UN"])
            self.food_observed.append(p)
            self.update_counts(p)
            self.update_belief()

    def update_counts(self, p):
        self.joint_counts[p] += 1
        if p[0] == "H":
            self.health_counts["Healthy"] += 1
        else:
            self.health_counts["Unhealthy"] += 1
        if p[1] == "T":
            self.taste_counts["Tasty"] += 1
        else:
            self.taste_counts["NotTasty"] += 1

    def update_belief(self):
        tot = sum(self.joint_counts.values())
        if tot == 0:
            return

        tab = np.array([
            [self.joint_counts["HT"], self.joint_counts["HN"]],
            [self.joint_counts["UT"], self.joint_counts["UN"]]
        ])

        self.current_phi = self.calc_phi(tab)

        hp = self.health_counts["Healthy"] / tot
        tp = self.taste_counts["Tasty"] / tot
        pc = 4 * (hp - 0.5) * (tp - 0.5)

        est = (1 - self.bias_strength) * self.current_phi + self.bias_strength * pc
        est = (est + 1) / 2
        self.ht_belief = (1 - self.learning_rate) * self.ht_belief + self.learning_rate * est

    def calc_phi(self, tab):
        if tab.sum() == 0:
            return 0
        a = tab[0,0] * tab[1,1] - tab[0,1] * tab[1,0]
        b = tab[0].sum() * tab[1].sum() * tab[:,0].sum() * tab[:,1].sum()
        if b <= 0:
            return 0
        return a / np.sqrt(b)

class Patch(Agent):
    COLORS = {
        "HT": "green",
        "HN": "blue",
        "UT": "red",
        "UN": "purple"
    }

    def __init__(self, unique_id, model, type):
        super().__init__(unique_id, model)
        self.type = type

    def get_color(self):
        return self.COLORS[self.type]