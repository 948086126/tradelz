# train_ppo.py
import os
import json
import argparse
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.trading_env import TradingEnv
from envs.wrappers import ObsSelectWrapper


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "close" not in df.columns:
        raise ValueError("CSV must contain close")
    df = df.dropna().reset_index(drop=True)
    if len(df) < 300:
        raise ValueError("数据太短，建议 >= 300")
    return df


def obs_idxs_from_mode(obs_mode: str):
    m = (obs_mode or "S").upper()
    if m == "S":
        return [0, 1, 2, 3, 5]
    if m == "M":
        return [0, 1, 2, 3, 4, 5]
    if m == "L":
        return [0, 1, 2, 3, 4, 5, 7]
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def make_env(df: pd.DataFrame, env_kwargs: dict, obs_idxs):
    def _thunk():
        env = TradingEnv(df=df, **env_kwargs)
        env = ObsSelectWrapper(env, obs_idxs)
        env = Monitor(env)
        return env
    return _thunk


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_env_kwargs(phase: int, use_action_shield: bool, log_dir: str, log_name: str):
    # Phase1：更强的参与驱动（part_penalty/idle_signal 稍强）
    if phase == 1:
        return dict(
            cost_rate=0.001,
            max_step_change=0.10,
            use_action_shield=bool(use_action_shield),

            ddi_penalty=2.0,
            ddi_free=0.012,

            turnover_penalty=0.0005,
            smooth_penalty=0.0025,
            cap_penalty=0.35,

            hold_a=0.0005,
            hold_b=0.0006,

            opp_cost=0.030,
            trend_bonus=0.012,

            idle_base=0.00005,
            idle_signal=0.00085,

            part_penalty=0.080,
            part_power=1.6,
            min_participation=0.10,

            desired_ema_alpha=0.35,

            train_random_start_pos=True,
            random_pos_prob=0.8,
            random_pos_low=0.08,
            random_pos_high=0.60,

            log_dir=log_dir,
            log_name=log_name,
            print_first_n=0,
        )

    # Phase2：更稳健（更重视少交易、撞风控更痛、参与惩罚变温和）
    if phase == 2:
        return dict(
            cost_rate=0.001,
            max_step_change=0.10,
            use_action_shield=bool(use_action_shield),

            ddi_penalty=2.2,
            ddi_free=0.010,

            #你现在 PPO 太偏向减仓，通常是这些惩罚太强：
            #turnover_penalty 太大（不敢加仓）
            #smooth_penalty 太大（不敢调整仓位)
            #cap_penalty 太大（太怕碰风控)
            #turnover_penalty: 0.0006 → 0.00045
            #smooth_penalty: 0.0040 → 0.0030
            #cap_penalty: 0.65 → 0.45
            turnover_penalty=0.0006,
            smooth_penalty=0.0040,
            cap_penalty=0.65,

            hold_a=0.0005,
            hold_b=0.0006,

            opp_cost=0.018,
            trend_bonus=0.010,

            idle_base=0.00005,
            idle_signal=0.00055,

            part_penalty=0.050,
            part_power=1.5,
            min_participation=0.08,

            desired_ema_alpha=0.35,

            train_random_start_pos=True,
            random_pos_prob=0.6,
            random_pos_low=0.08,
            random_pos_high=0.55,

            log_dir=log_dir,
            log_name=log_name,
            print_first_n=0,
        )

    raise ValueError(f"Unknown phase: {phase}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/gongye_fulian_features.csv")
    ap.add_argument("--symbol", type=str, default="601138")
    ap.add_argument("--obs_mode", type=str, default="S")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--total_steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--shield", action="store_true")
    ap.add_argument("--no_shield", action="store_true")

    ap.add_argument("--init_model", type=str, default="", help="Phase2 常用：加载 best_model 继续训")
    ap.add_argument("--use_vecnorm", action="store_true")
    ap.add_argument("--run_id", type=str, default="", help="用于Web任务唯一标识，防止覆盖")

    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_rl", exist_ok=True)

    df = load_df(args.data)

    obs_mode = args.obs_mode.upper()
    obs_idxs = obs_idxs_from_mode(obs_mode)

    use_action_shield = True
    if args.no_shield:
        use_action_shield = False
    elif args.shield:
        use_action_shield = True

    n = len(df)
    split = int(n * 0.8)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_eval = df.iloc[split:].reset_index(drop=True)

    suffix = f"_{args.run_id}" if args.run_id else ""
    run_tag = f"{args.symbol}_obs{obs_mode}_{'shieldON' if use_action_shield else 'shieldOFF'}_p{args.phase}{suffix}"

    train_log = f"train_{run_tag}.csv"
    eval_log = f"eval_{run_tag}.csv"

    env_kwargs = build_env_kwargs(args.phase, use_action_shield, log_dir="logs_rl", log_name=train_log)

    eval_env_kwargs = dict(env_kwargs)
    eval_env_kwargs["train_random_start_pos"] = False
    eval_env_kwargs["random_pos_prob"] = 0.0
    eval_env_kwargs["log_name"] = eval_log

    train_env = DummyVecEnv([make_env(df_train, env_kwargs, obs_idxs)])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(df_eval, eval_env_kwargs, obs_idxs)])
    eval_env = VecMonitor(eval_env)

    if args.use_vecnorm:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False
        eval_env.norm_reward = False

    ent_coef = 0.012 if args.phase == 1 else 0.008

    if args.init_model:
        model = PPO.load(args.init_model, env=train_env, device="auto")
        model.tensorboard_log = "logs_rl/tb"
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            tensorboard_log="logs_rl/tb",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
            use_sde=True,
            sde_sample_freq=4,
        )

    save_json(
        f"logs_rl/runmeta_{run_tag}.json",
        dict(
            data=args.data,
            symbol=args.symbol,
            obs_mode=obs_mode,
            obs_idxs=obs_idxs,
            phase=args.phase,
            total_steps=args.total_steps,
            seed=args.seed,
            use_action_shield=use_action_shield,
            env_kwargs=env_kwargs,
            use_vecnorm=bool(args.use_vecnorm),
            init_model=args.init_model,
        ),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="logs_rl",
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_steps, callback=eval_callback)

    out_path = f"models/ppo_{run_tag}.zip"
    model.save(out_path)
    print("Saved:", out_path)

    if args.use_vecnorm:
        vn_path = f"models/vecnorm_{run_tag}.pkl"
        train_env.save(vn_path)
        print("Saved VecNormalize:", vn_path)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
