"""Drop-in auto_cos_inc_rank module using a linear 200 -> 250 schedule."""

try:
    from .linear_inc import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from linear_inc import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
