try:
    from .rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_logarithmic_rank,
    )
except ImportError:
    from rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_logarithmic_rank,
    )


class SingleDeviceAutoCosIncWithAuxAdam(SingleDeviceAutoScheduledRankWithAuxAdam):
    """auto_cos_inc_rank-compatible logarithmic increase: 200 -> 250."""

    DEFAULT_RANK_SCHEDULE = "logarithmic"
    DEFAULT_RANK = 200
    DEFAULT_RANK_START = 200
    DEFAULT_RANK_END = 250


SingleDeviceLogarithmicIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
SingleDeviceFinalIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
