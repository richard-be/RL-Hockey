import run_training
import tyro

# Define parameter grid
alphas = [0.0, 0.1, 0.2, 0.3]
betas = [0, 1, 2]

# Sequential loop

for alpha in alphas:
    for beta in betas:
        args = tyro.cli(run_training.Args, args=[
            "--total-timesteps", "50000",
            "--num-envs", "4",
            "--track",
            "--alpha", str(alpha),
            "--beta", str(beta),
            "--no-autotune"
        ])
        print(f"Running with alpha={alpha}, beta={beta}")
        run_training.main(args)
    print(f"Running with autotune, beta={beta}")
    args = tyro.cli(run_training.Args, args=[
        "--total-timesteps", "50000",
        "--num-envs", "4",
        "--track",
        "--beta", str(beta)
    ])
    run_training.main(args)