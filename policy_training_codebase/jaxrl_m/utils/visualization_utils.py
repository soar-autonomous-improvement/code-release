import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def make_single_trajectory_visual(
    q_estimates,
    values,
    obs_images,
    goal_images,
):
    def np_unstack(array, axis):
        arr = np.split(array, array.shape[axis], axis)
        arr = [a.squeeze() for a in arr]
        return arr

    def process_images(images):
        # assume image in C, H, W shape
        assert len(images.shape) == 4
        assert images.shape[-1] == 3

        interval = max(1, images.shape[0] // 4)

        sel_images = images[::interval]
        sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
        return sel_images

    fig, axs = plt.subplots(4, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)
    plt.xlim([0, q_estimates.shape[-1]])

    obs_images = process_images(obs_images)
    goal_images = process_images(goal_images)

    axs[0].imshow(obs_images)
    axs[1].imshow(goal_images)

    axs[2].plot(q_estimates[0, :], linestyle="--", marker="o")
    axs[2].plot(q_estimates[1, :], linestyle="--", marker="o")
    axs[2].set_ylabel("q values")

    axs[3].plot(values[0, :], linestyle="--", marker="o")
    axs[3].plot(values[1, :], linestyle="--", marker="o")
    axs[3].set_ylabel("values")

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    return out_image


def value_and_reward_visulization(trajs, agent, seed=0):

    rng = jax.random.PRNGKey(seed)
    n_trajs = len(trajs)
    visualization_images = []

    # for each trajectory
    for i in range(n_trajs):
        observations = trajs[i]["observations"]
        next_observations = trajs[i]["next_observations"]
        goals = {
            "image": np.repeat(
                trajs[i]["next_observations"]["image"][-1, ...][None, ...],
                repeats=trajs[i]["observations"]["image"].shape[0],
                axis=0,
            ),
            "proprio": np.repeat(
                trajs[i]["next_observations"]["proprio"][-1, ...][None, ...],
                repeats=trajs[i]["observations"]["proprio"].shape[0],
                axis=0,
            ),
        }

        actions = trajs[i]["actions"]

        q_pred = agent.forward_critic(
            (observations, goals), actions, rng=None, train=False
        )

        values = agent.forward_critic(
            (observations, goals),
            agent.forward_policy((observations, goals), rng, train=False).sample(
                seed=rng
            ),
            rng=None,
            train=False,
        )

        visualization_images.append(
            make_single_trajectory_visual(
                q_pred,
                values,
                observations["image"],
                goals["image"],
            )
        )

    return np.concatenate(visualization_images, 0)
