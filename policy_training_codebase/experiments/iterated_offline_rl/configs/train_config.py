import os
from ml_collections import ConfigDict

def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        num_steps=int(201000),
        log_interval=100,
        save_interval=5000,
        eval_interval=5000,
        num_val_batches=8,
        save_dir="gs://autonomous-improvement/jaxrl_log",
        resume_path="",
        seed=242,
    )

    base_data_config = dict(
        shuffle_buffer_size=25000,
        augment=True,
        augment_next_obs_goal_differently=False,
        cache=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    possible_structures = {
        "crl": ConfigDict(
            dict(
                agent="stable_contrastive_rl",
                agent_kwargs=dict(
                    critic_network_kwargs=dict(
                        hidden_dims=(256, 256, 256), use_layer_norm=True
                    ),
                    critic_kwargs=dict(init_final=1e-12, repr_dim=16, twin_q=True),
                    policy_network_kwargs=dict(
                        hidden_dims=(256, 256, 256), dropout_rate=0.1
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        std_parameterization="fixed",
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=2000000,
                    use_td=True,
                    gcbc_coef=0.2,
                    discount=0.99,
                    temperature=1.0,
                    target_update_rate=0.002,
                    shared_encoder=True,
                    early_goal_concat=False,
                    shared_goal_encoder=True,
                    use_proprio=False,
                ),
                dataset_kwargs=dict(
                    #goal_relabeling_strategy="delta_goals", # specified in the data config
                    #goal_relabeling_kwargs=dict(goal_delta=[0, 24]), # specified in the data config
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=False, act="swish"
                ),
                **base_real_config,
            )
        ),

        "diffusion_gcbc": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=2000000,
                ),
                dataset_kwargs=dict(
                    #goal_relabeling_strategy="delta_goals", # specified in the data config
                    #goal_relabeling_kwargs=dict(goal_delta=[0, 24]), # specified in the data config
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=4,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),

        "gaussian_gcbc": ConfigDict(
            dict(
                agent="gc_bc",
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                        dropout_rate=0.1,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        std_parameterization="fixed",
                        fixed_std=[1, 1, 1, 1, 1, 1, 0.1],
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                    # freeze_encoder=True,
                ),
                dataset_kwargs=dict(
                    relabel_actions=True,
                    action_merge_horizon=2,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",  # diff: bridge release use resnet-50
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=False, act="swish"
                ),
                **base_real_config,
            )
        ),

        "gc_iql": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                        dropout_rate=0.1,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        std_parameterization="fixed",
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                    discount=0.99,
                    expectile=0.7, # to ablate over
                    temperature=1.0, # to ablate over, lower is more RL
                    target_update_rate=0.002,
                    shared_encoder=True, # to ablate over
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    negative_proportion=0.3,
                ),
                dataset_kwargs=dict(
                    #goal_relabeling_strategy="uniform",
                    #goal_relabeling_kwargs=dict(reached_proportion=0.1),
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=False,
                    act="swish",
                ),
                **base_real_config,
            ),
        ),

        "vision_backbone_1": ConfigDict(
            dict(
                agent="vision_backbone_1",
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                        dropout_rate=0.1,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        std_parameterization="fixed",
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                    gcbc_loss_ratio=0.5,
                    freeze_encoder=True,
                ),
                dataset_kwargs=dict(
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",  # diff: bridge release use resnet-50
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=False, act="swish"
                ),
                **base_real_config,
            )
        ),

    }

    return possible_structures[config_string]
