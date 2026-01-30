import numpy as np
import pandas as pd

print("numpy:", np.__version__)
print("pandas:", pd.__version__)

from envs.trading_env import TradingEnv
print("TradingEnv imported OK")