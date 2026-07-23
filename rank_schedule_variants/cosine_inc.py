try:
    from .rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_cosine_rank,
    )
except ImportError:
    from rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_cosine_rank,
    )


class SingleDeviceAutoCosIncWithAuxAdam(SingleDeviceAutoScheduledRankWithAuxAdam):
    """auto_cos_inc_rank-compatible cosine increase: 200 -> 250."""

    DEFAULT_RANK_SCHEDULE = "cosine"
    DEFAULT_RANK = 200
    DEFAULT_RANK_START = 200
    DEFAULT_RANK_END = 250


SingleDeviceCosineIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
SingleDeviceFinalIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
