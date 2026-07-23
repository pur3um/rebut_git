"""Drop-in auto_cos_inc_rank module using a logistic 200 -> 250 schedule."""

try:
    from .logistic_inc import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from logistic_inc import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
