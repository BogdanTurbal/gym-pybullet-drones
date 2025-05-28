import wandb
run = wandb.init()
artifact = run.use_artifact('bogdan-turbal-y/drone-swarm-rpm-artifacts/best_model_reward_129.62_improvement_1.55:v0', type='model')
artifact_dir = artifact.download()
print(artifact_dir)


# /Users/bohdan.turbal/Desktop/dimploma_thesis/gym-pybullet-drones/gym_pybullet_drones/examples/artifacts/model_eval_15:v1/model_eval_15_step_375000_reward_50.66_20250526_234035.zip
#/Users/bohdan.turbal/Desktop/dimploma_thesis/artifacts/best_model_reward_179.14:v1