"""Drop-in auto_cos_inc_rank module using an exponential 200 -> 250 schedule."""

try:
    from .exp_inc import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from exp_inc import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
