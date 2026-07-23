"""Drop-in auto_cos_inc_rank module using a cosine 250 -> 200 schedule."""

try:
    from .cosine_dec import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from cosine_dec import SingleDeviceAutoCosIncWithAuxAdam

__all__ = ["SingleDeviceAutoCosIncWithAuxAdam"]
