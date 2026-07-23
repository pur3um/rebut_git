"""Drop-in auto_cos_inc_rank module using fixed rank 200."""

try:
    from .fixed_rank import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from fixed_rank import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
