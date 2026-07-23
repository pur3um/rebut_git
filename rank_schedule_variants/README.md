# auto_cos_inc_rank schedule variants

첨부된 `auto_cos_inc_rank.py`의 optimizer 구조와 클래스명에 맞추고,
rank schedule만 바꾼 버전입니다. 모든 변형은 drop-in 호환 클래스
`SingleDeviceAutoCosIncWithAuxAdam`을 제공합니다.

앞서 요청한 공정 비교 조건을 유지해 Newton-Schulz 반복 기본값은
`ns_steps=10`, 증가형 기본 범위는 `200 -> 250`입니다.

| 파일 | 기본 rank 동작 | 곡선 특성 |
| --- | --- | --- |
| `cosine_inc.py` | 200 -> 250 | 초반/후반이 느린 cosine 증가 |
| `exp_inc.py` | 200 -> 250 | 초반이 느리고 후반이 빠른 exponential 증가 |
| `log_inc.py` | 200 -> 250 | 초반이 빠르고 후반이 느린 logarithmic 증가 |
| `linear_inc.py` | 200 -> 250 | 일정한 속도의 linear 증가 |
| `logistic_inc.py` | 200 -> 250 | 초반/후반이 느린 S-curve 증가 |
| `cosine_dec.py` | 250 -> 200 | cosine 감소 |
| `fixed_rank.py` | 200 고정 | schedule 없음 |

기존 모듈명과 같은 import 형태로 쓰기 위한 진입 파일도 함께 제공합니다.

| 진입 파일 | 연결되는 구현 |
| --- | --- |
| `auto_cos_inc_rank.py` | `cosine_inc.py` |
| `auto_exp_inc_rank.py` | `exp_inc.py` |
| `auto_log_inc_rank.py` | `log_inc.py` |
| `auto_linear_inc_rank.py` | `linear_inc.py` |
| `auto_logistic_inc_rank.py` | `logistic_inc.py` |
| `auto_cos_dec_rank.py` | `cosine_dec.py` |
| `auto_fixed_rank.py` | `fixed_rank.py` |

## 파일별 사용법

폴더 전체를 기존 optimizer 모듈 위치에 복사한 뒤 원하는 파일에서 기존
`auto_cos_inc_rank.py`와 같은 클래스명을 import합니다. 예를 들어
exponential 증가:

```python
from rank_schedule_variants.auto_exp_inc_rank import SingleDeviceAutoCosIncWithAuxAdam

optimizer = SingleDeviceAutoCosIncWithAuxAdam(
    [
        {
            "params": muon_params,
            "use_muon": True,
            "warmup_steps": total_rank_schedule_steps,
        },
        {
            "params": adam_params,
            "use_muon": False,
        },
    ]
)
```

기존 optimizer registry가 다음 import를 사용한다면, 선택한 변형 파일을
`auto_cos_inc_rank.py` 자리에 두어 import문을 그대로 유지할 수 있습니다.

```python
from .auto_cos_inc_rank import SingleDeviceAutoCosIncWithAuxAdam
```

## 공정한 비교를 위한 설정

원본 `auto_cos_inc_rank.py`는 `rank_end=rank`, `warmup_steps=1`이므로
기본값만 쓰면 rank가 증가하지 않습니다. 이 비교용 구현은 곡선이 실제로
나타나도록 `rank_end=250`, `warmup_steps=100000`을 기본값으로 사용합니다.
실험에서는 모든 방법에 동일한
`warmup_steps=total_rank_schedule_steps`를 명시하십시오.

다음 값도 모든 증가 실험에서 동일하게 유지하십시오.

```python
{
    "ns_steps": 10,
    "rank": 200,
    "rank_start": 200,
    "rank_end": 250,
    "warmup_steps": total_rank_schedule_steps,
}
```

중요: 실제 적용 rank는 아래와 같이 `rank`를 하한으로 사용합니다.

```python
applied_rank = max(rank, scheduled_rank)
```

따라서 64에서 시작하려면 `rank_start=64`뿐 아니라 `rank=64`도 함께
설정해야 합니다. `rank=200, rank_start=64`이면 실제 적용 rank는 200
아래로 내려가지 않습니다.

감소 실험은 `rank=200`, `rank_start=250`, `rank_end=200`, 고정 실험은
세 rank 값을 모두 200으로 사용합니다.

곡선 강도는 다음 인자로 조정할 수 있습니다.

```python
{
    "exp_rate": 5.0,
    "log_rate": 9.0,
    "logistic_steepness": 12.0,
}
```

`rank_schedule_optimizer.py` 하나만 직접 사용하고 param group의
`rank_schedule`을 `cosine`, `exponential`, `logarithmic`, `linear`,
`logistic`, `fixed` 중 하나로 지정해도 됩니다. 감소는 원하는 곡선과 함께
`rank_start > rank_end`로 설정하면 됩니다.
