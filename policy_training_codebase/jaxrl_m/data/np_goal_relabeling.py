import numpy as np


def geometric(traj, *, reached_proportion, discount):
    """
    Relabels with a geometric distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled geometrically from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = np.shape(traj["terminals"])[0]

    # geometrically select a future index for each transition i in the range [i + 1, traj_len)
    arange = np.arange(traj_len)
    is_future_mask = (arange[:, None] < arange[None]).astype(np.float32)
    d = discount ** (arange[None] - arange[:, None]).astype(np.float32)

    probs = is_future_mask * d
    # hack: last row is all 0s, and will cause division issues.
    # This is masked out by goal_reached_mask so it doesn't matter
    probs[-1, -1] = 1.0
    probs = probs / probs.sum(axis=1, keepdims=True)  # normalize
    goal_idxs = np.array(
        [
            np.random.choice(np.arange(traj_len), size=1, p=probs[i] / probs[i].sum())
            for i in range(traj_len)
        ],
        dtype=np.int32,
    )[:, 0]

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = np.random.uniform(size=traj_len) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = np.logical_or(
        goal_reached_mask, np.arange(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = np.where(goal_reached_mask, np.arange(traj_len), goal_idxs)

    # select goals
    traj["goals"] = np.take(traj["next_observations"], goal_idxs, axis=0)

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = np.where(goal_reached_mask, 0, -1).astype(np.int32)

    # add masks
    traj["masks"] = np.logical_not(traj["terminals"])

    # add distances to goal
    traj["goal_dists"] = goal_idxs - np.arange(traj_len)

    return traj


GOAL_RELABELING_FUNCTIONS = {
    "geometric": geometric,
}
