# webapp/schemas.py
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class BacktestRunRequest(BaseModel):
    compare_shield: bool = False

    # ✅ 新增：PPO/回测用的 observation 子集模式
    obs_mode: str = Field("S", description="PPO obs子集: S(5维)/M(6维)/L(7维)")

    symbol: str = Field(..., description="A股代码，例如 601138")
    start_date: Optional[str] = Field(None, description="YYYYMMDD 或 YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYYMMDD 或 YYYY-MM-DD")

    policy_name: str = Field(..., description="always0 / buyhold / ddcontrol / ppo")
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    env_kwargs: Dict[str, Any] = Field(default_factory=dict)

    data_csv_path: str = Field(..., description="数据CSV路径，如 data/gongye_fulian_features.csv")


class BacktestRunResponse(BaseModel):
    run_id: str
    metrics: Dict[str, Any]
    csv_path: str
    equity_png: str
    pos_png: str


class BacktestRunItem(BaseModel):
    run_id: str
    created_at: str
    symbol: str
    start_date: Optional[str]
    end_date: Optional[str]
    policy_name: str
    metrics: Dict[str, Any]
    csv_path: str
    equity_png: str
    pos_png: str


class BacktestRunListResponse(BaseModel):
    items: List[BacktestRunItem]
class BacktestSingleResult(BaseModel):
    run_id: str
    metrics: Dict[str, Any]
    csv_path: str
    equity_png: str
    pos_png: str


class BacktestCompareResponse(BaseModel):
    compare: bool = True
    symbol: str
    policy_name: str
    shield_on: BacktestSingleResult
    shield_off: BacktestSingleResult
    diff: Dict[str, Any] = Field(default_factory=dict)

