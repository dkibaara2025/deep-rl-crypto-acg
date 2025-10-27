from src.data_preprocessing import load_dataset, top_assets, build_features, build_arm_data
from src.model_training import TrainConfig, bandit_run_once
from src.curriculum_bandit import UCB1Config

CSV = "/content/export/tables/dataset_long_1D.csv"
df = load_dataset(CSV)
assets = top_assets(df, k=5)

# features
df_price = build_features(df, assets, augmented=False)
data_xy, arms, cov = build_arm_data(df_price, horizons=(1,3,7), win=64)

# train
cfg = TrainConfig(seed=1337, ucb_c=1.2, rounds=150, steps_per_pull=1, lr=2e-3, batch_size=256, device="cpu")
per_arm, agg, model, rounds_used = bandit_run_once(df_price, data_xy, arms, cfg)
print(agg)
