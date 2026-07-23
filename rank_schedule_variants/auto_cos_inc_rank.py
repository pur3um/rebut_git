"""Drop-in auto_cos_inc_rank module using a cosine 200 -> 250 schedule."""

try:
    from .cosine_inc import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from cosine_inc import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
