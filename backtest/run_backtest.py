# backtest/run_backtest.py
import os
import argparse
import pandas as pd

from envs.trading_env import TradingEnv
from envs.wrappers import ObsSelectWrapper

from policies.policies import always_zero_policy, buy_and_hold_policy
from policies.ddcontrol import ddcontrol_with_features

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def obs_idxs_from_mode(obs_mode: str):
    # 9维原始 obs：
    # [position, ret_1, ret_5, ret_20, vol_20, price_vs_ma20, dd, dd_increase, equity_log]
    m = (obs_mode or "S").upper()
    if m == "S":
        return [0, 1, 2, 3, 5]          # 5维
    if m == "M":
        return [0, 1, 2, 3, 4, 5]       # 6维
    if m == "L":
        return [0, 1, 2, 3, 4, 5, 7]    # 7维
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "close" not in df.columns:
        raise ValueError("CSV 必须包含 close 列")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"数据太短：len(df)={len(df)}，建议至少 200+")
    return df


def run_policy(df: pd.DataFrame, policy_fn, log_dir: str, log_name: str, env_kwargs: dict, obs_idxs, policy_kwargs: dict):
    """baseline 用：非 VecEnv"""
    os.makedirs(log_dir, exist_ok=True)

    env = TradingEnv(df=df, log_dir=log_dir, log_name=log_name, **env_kwargs)
    env = ObsSelectWrapper(env, obs_idxs)

    obs, _ = env.reset()
    done = False
    state = {}

    while not done:
        action = policy_fn(obs, state=state, **policy_kwargs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print("Saved:", os.path.join(log_dir, log_name))


def run_ppo_backtest(df: pd.DataFrame, log_dir: str, log_name: str, env_kwargs: dict, obs_idxs,
                     model_path: str, vecnorm_path: str, deterministic: bool):
    """PPO 专用：VecEnv + VecNormalize（标准做法，100%不会再报错）"""
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        env = TradingEnv(df=df, log_dir=log_dir, log_name=log_name, **env_kwargs)
        env = ObsSelectWrapper(env, obs_idxs)
        return env

    venv = DummyVecEnv([make_env])

    # 加载 VecNormalize，并绑定 venv
    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(model_path)

    obs = venv.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        done = bool(dones[0])

    venv.close()
    print("Saved:", os.path.join(log_dir, log_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/gongye_fulian_features.csv")
    ap.add_argument("--log_dir", type=str, default="logs")

    ap.add_argument("--policy", type=str, default="all", choices=["all", "baseline", "ppo"])
    ap.add_argument("--model_path", type=str, default="models/ppo_601138_obsS_shieldON_p2.zip")
    ap.add_argument("--vecnorm_path", type=str, default="models/vecnorm_601138_obsS_shieldON_p2.pkl")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--obs_mode", type=str, default="S", choices=["S", "M", "L"])

    args = ap.parse_args()

    df = load_data(args.data)
    obs_idxs = obs_idxs_from_mode(args.obs_mode)

    max_step_change = 0.1
    ENV_KWARGS = dict(
        cost_rate=0.001,
        max_step_change=max_step_change,
        train_random_start_pos=False,
        print_first_n=0,
        use_action_shield=True,
    )

    if args.policy in ("all", "baseline"):
        run_policy(
            df, always_zero_policy, args.log_dir, "behavior_baseline_always0.csv",
            env_kwargs=ENV_KWARGS, obs_idxs=obs_idxs,
            policy_kwargs=dict(max_step_change=max_step_change),
        )

        run_policy(
            df, buy_and_hold_policy, args.log_dir, "behavior_baseline_buyhold.csv",
            env_kwargs=ENV_KWARGS, obs_idxs=obs_idxs,
            policy_kwargs=dict(max_step_change=max_step_change),
        )

        run_policy(
            df, ddcontrol_with_features, args.log_dir, "behavior_baseline_ddcontrol.csv",
            env_kwargs=ENV_KWARGS, obs_idxs=obs_idxs,
            policy_kwargs=dict(
                max_step_change=max_step_change,
                base_pos=0.6,
                min_pos=0.1,
                dd_low=0.02,
                dd_high=0.08,
                vol_low=0.02,
                vol_high=0.04,
                vol_cut=0.6,
                ma_band=0.01,
                recover_step=0.01,
                min_rebalance_gap=0.03,
                block_add_when_below_ma=True
            ),
        )

    if args.policy in ("all", "ppo"):
        run_ppo_backtest(
            df=df,
            log_dir=args.log_dir,
            log_name="behavior_ppo.csv",
            env_kwargs=ENV_KWARGS,
            obs_idxs=obs_idxs,
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            deterministic=bool(args.deterministic),
        )


if __name__ == "__main__":
    main()
