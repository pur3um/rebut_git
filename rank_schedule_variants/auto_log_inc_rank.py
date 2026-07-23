"""Drop-in auto_cos_inc_rank module using a logarithmic 200 -> 250 schedule."""

try:
    from .log_inc import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from log_inc import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
