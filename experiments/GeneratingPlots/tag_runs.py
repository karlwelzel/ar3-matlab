import wandb

api = wandb.Api()
all_runs = api.runs(
    path="ar3-project/ar3-project",
    filters={"group": "Exp_Benchmark_3"},
)

for run in all_runs:
    print(f"{run.name=}")
    try:
        mgh_num = int(run.config["problem"])
    except ValueError:
        mgh_num = None

    original_tags = run.tags.copy()
    run.tags = []

    if mgh_num is not None:
        run.tags.append("benchmark")

    if mgh_num is None or mgh_num % 2 == 1:
        run.tags.append("training")

    if mgh_num is None:
        run.tags.append("extra")

    if "ill_cond" in run.config["problem"]:
        run.tags.append("regularized-cubic")

    if run.tags != original_tags:
        run.update()
