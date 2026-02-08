import os
from time import strftime
from itertools import product

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

from utils import load_cal, mae, ks
from agent import Patch


CELL_KEYS = ("HT", "HN", "UT", "UN")


def _normalize_weights(weights):
    if isinstance(weights, dict):
        vals = {
            "HT": float(weights["HT"]),
            "HN": float(weights["HN"]),
            "UT": float(weights["UT"]),
            "UN": float(weights["UN"]),
        }
    elif isinstance(weights, (list, tuple, np.ndarray)) and len(weights) == 4:
        vals = {
            "HT": float(weights[0]),
            "HN": float(weights[1]),
            "UT": float(weights[2]),
            "UN": float(weights[3]),
        }
    else:
        raise ValueError("weights must be dict with HT/HN/UT/UN or length-4 sequence")

    for k, v in vals.items():
        if v <= 0:
            raise ValueError(f"weight {k} must be > 0 for identifiability")

    total = sum(vals.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    return {k: v / total for k, v in vals.items()}


class FullWeightedLearningAgent(Agent):
    def __init__(self, unique_id, model, row, ht_belief):
        super().__init__(unique_id, model)
        self.row = row
        self.food_observed = []
        self.ht_belief = ht_belief

        self.cell_weights = model.cell_weights
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
            if np.random.random() < self.noise:
                p = np.random.choice(["HT", "HN", "UT", "UN"])
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
        tab = np.array([
            [self.joint_counts["HT"], self.joint_counts["HN"]],
            [self.joint_counts["UT"], self.joint_counts["UN"]],
        ])

        r = self.calc_cell_weighted_R(tab)
        est = (r + 1.0) / 2.0
        self.ht_belief = (1 - self.learning_rate) * self.ht_belief + self.learning_rate * est

    def calc_cell_weighted_R(self, tab):
        a = tab[0, 0]  # HT
        b = tab[0, 1]  # HN
        c = tab[1, 0]  # UT
        d = tab[1, 1]  # UN

        w = self.cell_weights
        denom = w["HT"] * a + w["HN"] * b + w["UT"] * c + w["UN"] * d
        if denom <= 0:
            return 0.0
        num = w["HT"] * a - w["HN"] * b - w["UT"] * c + w["UN"] * d
        return num / denom


class FullWeightedLearningModel(Model):
    def __init__(self, N, width, height, env_type, cell_weights, learning_rate, noise=0.0, seed=None):
        super().__init__(seed=seed)

        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)

        self.env_type = env_type
        self.cell_weights = _normalize_weights(cell_weights)
        self.learning_rate = learning_rate
        self.noise = noise

        if seed is not None:
            np.random.seed(seed)

        self.create_agents()
        self.create_patches()

        self.datacollector = DataCollector(
            agent_reporters={"ht": "ht_belief"}
        )

    def create_agents(self):
        for i in range(self.num_agents):
            ht0 = np.random.normal(0.5, 0.15)
            a = FullWeightedLearningAgent(i, self, i, ht0)
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
            types.append(np.random.choice(list(props.keys())))
        np.random.shuffle(types)

        for y in range(self.grid.height):
            patch = Patch(f"patch_{x}_{y}", self, types[y])
            self.grid.place_agent(patch, (x, y))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


def _run_reps(env, weights, lr, n_agents, steps, reps, width=24, height=None, noise=0.0, seed=None):
    h = height if height is not None else n_agents
    all_runs = []

    for i in range(reps):
        run_seed = None if seed is None else int(seed + i)
        m = FullWeightedLearningModel(
            N=n_agents,
            width=width,
            height=h,
            env_type=env,
            cell_weights=weights,
            learning_rate=lr,
            noise=noise,
            seed=run_seed,
        )
        for _ in range(steps):
            m.step()
        df = m.datacollector.get_agent_vars_dataframe().reset_index()
        last = df["Step"].max()
        all_runs.append(df[df["Step"] == last]["ht"].values.astype(float))

    mat = np.vstack(all_runs)
    avg = mat.mean(axis=0)
    return mat, avg


def _default_weight_grid(weight_range):
    if weight_range is None:
        raise ValueError("weight_range cannot be None")
    for w in weight_range:
        if w <= 0:
            raise ValueError("weight_range must contain only positive values")
    grid = []
    for w_ht, w_hn, w_ut, w_un in product(weight_range, repeat=4):
        grid.append({"HT": w_ht, "HN": w_hn, "UT": w_ut, "UN": w_un})
    return grid


def fit_group(df, weight_grid=None, weight_range=None, lr=0.9, steps=24, reps=1, n_agents=52, noise=0.0, seed=None):
    emp = {
        "healthy": df[df.env == "healthy"]["ht"].values,
        "unhealthy": df[df.env == "unhealthy"]["ht"].values,
        "balanced": df[df.env == "balanced"]["ht"].values,
    }

    if weight_grid is None:
        if weight_range is None:
            weight_range = np.round(np.linspace(0.1, 9.0, 5), 2)
        weight_grid = _default_weight_grid(weight_range)

    rows = []
    for w in weight_grid:
        w_norm = _normalize_weights(w)
        for env, emp_vals in emp.items():
            sims_mat, sims_avg = _run_reps(
                env=env,
                weights=w_norm,
                lr=lr,
                n_agents=n_agents,
                steps=steps,
                reps=reps,
                noise=noise,
                seed=seed,
            )

            rows.append({
                "env": env,
                "w_ht": float(w_norm["HT"]),
                "w_hn": float(w_norm["HN"]),
                "w_ut": float(w_norm["UT"]),
                "w_un": float(w_norm["UN"]),
                "mae": mae(sims_avg, emp_vals),
                "ks": ks(sims_avg, emp_vals),
                "sim_ht": sims_mat,
                "n_agents": int(n_agents),
            })

    g = pd.DataFrame(rows)
    g = g.sort_values(["env", "w_ht", "w_hn", "w_ut", "w_un"]).reset_index(drop=True)
    return g

def best_weights(g):
    tmp = (
        g.sort_values(["env", "mae"])
        .groupby("env")
        .first()
        .reset_index()
    )
    return tmp[["env", "w_ht", "w_hn", "w_ut", "w_un", "mae"]]


def run(
    weight_grid=None,
    weight_range=None,
    lr=0.9,
    steps=24,
    reps=10,
    n_agents=52,
    noise=0.0,
    seed=None,
):
    df = load_cal()

    g = fit_group(
        df,
        weight_grid=weight_grid,
        weight_range=weight_range,
        lr=lr,
        steps=steps,
        reps=reps,
        n_agents=n_agents,
        noise=noise,
        seed=seed,
    )

    ts = strftime("%y%m%d_%H%M%S")
    outdir = f"repo/results/integration_full_{ts}"
    os.makedirs(outdir, exist_ok=True)

    g.to_csv(f"{outdir}/group.csv", index=False)
    best = best_weights(g)
    best.to_csv(f"{outdir}/best_weights.csv", index=False)

    return g, best


if __name__ == "__main__":
    run()
