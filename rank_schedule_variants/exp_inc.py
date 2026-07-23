try:
    from .rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_exponential_rank,
    )
except ImportError:
    from rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_exponential_rank,
    )


class SingleDeviceAutoCosIncWithAuxAdam(SingleDeviceAutoScheduledRankWithAuxAdam):
    """auto_cos_inc_rank-compatible exponential increase: 200 -> 250."""

    DEFAULT_RANK_SCHEDULE = "exponential"
    DEFAULT_RANK = 200
    DEFAULT_RANK_START = 200
    DEFAULT_RANK_END = 250


SingleDeviceExponentialIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
SingleDeviceFinalIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
