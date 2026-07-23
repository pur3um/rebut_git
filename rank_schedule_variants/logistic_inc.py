try:
    from .rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_logistic_rank,
    )
except ImportError:
    from rank_schedule_optimizer import (
        SingleDeviceAutoScheduledRankWithAuxAdam,
        get_logistic_rank,
    )


class SingleDeviceAutoCosIncWithAuxAdam(SingleDeviceAutoScheduledRankWithAuxAdam):
    """auto_cos_inc_rank-compatible logistic increase: 200 -> 250."""

    DEFAULT_RANK_SCHEDULE = "logistic"
    DEFAULT_RANK = 200
    DEFAULT_RANK_START = 200
    DEFAULT_RANK_END = 250


SingleDeviceLogisticIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
SingleDeviceFinalIncWithAuxAdam = SingleDeviceAutoCosIncWithAuxAdam
