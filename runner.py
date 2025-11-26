from model import LearningModel
import numpy as np

def run_single(env, bias, lr, n_agents, steps, width=24, height=None, noise=0.0, seed=None):
    h = height if height is not None else n_agents
    m = LearningModel(
        N=n_agents,
        width=width,
        height=h,
        env_type=env,
        bias_strength=bias,
        learning_rate=lr,
        noise=noise,
        seed=seed
    )
    for _ in range(steps):
        m.step()
    return m

def run_reps(env, bias, lr, n_agents, steps, reps, width=24, height=None, noise=0.05):

    all_runs = []
    for _ in range(reps):
        m = run_single(env, bias, lr, n_agents, steps, width, height, noise)
        all_runs.append(final_ht(m))

    mat = np.vstack(all_runs)

    avg = mat.mean(axis=0)
    return avg

def run_multibias(env, biases, lr, steps, width=24, noise=0.05, seed=None):

    n = len(biases)
    m = LearningModel(
        N=n,
        width=width,
        height=n,
        env_type=env,
        bias_strength=biases[0],
        learning_rate=lr,
        noise=noise,
        seed=seed
    )

    for i, a in enumerate(m.schedule.agents):
        a.bias_strength = biases[i]

    for _ in range(steps):
        m.step()

    return m

def final_ht(model):
    df = model.datacollector.get_agent_vars_dataframe().reset_index()
    last = df["Step"].max()
    return df[df["Step"] == last]["ht"].values