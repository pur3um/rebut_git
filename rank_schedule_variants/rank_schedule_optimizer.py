import math
import statistics
from typing import Optional

import torch


SUPPORTED_RANK_SCHEDULES = {
    "cosine",
    "exponential",
    "logarithmic",
    "linear",
    "logistic",
    "fixed",
}


def _normalized_step(step: int, warmup_steps: int) -> float:
    """Map optimizer steps 1..warmup_steps exactly to progress 0..1."""
    step = int(step)
    warmup_steps = int(warmup_steps)

    if warmup_steps <= 1:
        return 1.0
    if step <= 1:
        return 0.0
    if step >= warmup_steps:
        return 1.0
    return (step - 1) / float(warmup_steps - 1)


def _schedule_progress(
    t: float,
    schedule_type: str,
    exp_rate: float = 5.0,
    log_rate: float = 9.0,
    logistic_steepness: float = 12.0,
) -> float:
    """
    Return normalized schedule progress in [0, 1].

    exponential: slow early, fast late
    logarithmic: fast early, slow late
    logistic: slow-fast-slow, with exact endpoint normalization
    """
    schedule_type = str(schedule_type).lower()
    t = max(0.0, min(float(t), 1.0))

    if schedule_type == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * t))
    elif schedule_type == "exponential":
        rate = float(exp_rate)
        if abs(rate) < 1e-12:
            progress = t
        else:
            progress = math.expm1(rate * t) / math.expm1(rate)
    elif schedule_type == "logarithmic":
        rate = float(log_rate)
        if rate <= 0.0:
            raise ValueError("log_rate must be positive.")
        progress = math.log1p(rate * t) / math.log1p(rate)
    elif schedule_type == "linear":
        progress = t
    elif schedule_type == "logistic":
        steepness = float(logistic_steepness)
        if abs(steepness) < 1e-12:
            progress = t
        else:
            sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
            lower = sigmoid(-0.5 * steepness)
            upper = sigmoid(0.5 * steepness)
            current = sigmoid(steepness * (t - 0.5))
            progress = (current - lower) / (upper - lower)
    elif schedule_type == "fixed":
        progress = 0.0
    else:
        supported = ", ".join(sorted(SUPPORTED_RANK_SCHEDULES))
        raise ValueError(
            f"Unknown rank schedule '{schedule_type}'. Supported schedules: {supported}."
        )

    return max(0.0, min(float(progress), 1.0))


def get_scheduled_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    schedule_type: str,
    exp_rate: float = 5.0,
    log_rate: float = 9.0,
    logistic_steepness: float = 12.0,
) -> int:
    """Return an integer rank with exact start and end values."""
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    schedule_type = str(schedule_type).lower()

    if schedule_type == "fixed":
        return start_rank

    t = _normalized_step(step, warmup_steps)
    progress = _schedule_progress(
        t,
        schedule_type=schedule_type,
        exp_rate=exp_rate,
        log_rate=log_rate,
        logistic_steepness=logistic_steepness,
    )
    rank = start_rank + (end_rank - start_rank) * progress
    return int(round(rank))


def get_cosine_rank(step: int, start_rank: int, end_rank: int, warmup_steps: int) -> int:
    return get_scheduled_rank(
        step, start_rank, end_rank, warmup_steps, schedule_type="cosine"
    )


def get_exponential_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    exp_rate: float = 5.0,
) -> int:
    return get_scheduled_rank(
        step,
        start_rank,
        end_rank,
        warmup_steps,
        schedule_type="exponential",
        exp_rate=exp_rate,
    )


def get_logarithmic_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    log_rate: float = 9.0,
) -> int:
    return get_scheduled_rank(
        step,
        start_rank,
        end_rank,
        warmup_steps,
        schedule_type="logarithmic",
        log_rate=log_rate,
    )


def get_linear_rank(step: int, start_rank: int, end_rank: int, warmup_steps: int) -> int:
    return get_scheduled_rank(
        step, start_rank, end_rank, warmup_steps, schedule_type="linear"
    )


def get_logistic_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    logistic_steepness: float = 12.0,
) -> int:
    return get_scheduled_rank(
        step,
        start_rank,
        end_rank,
        warmup_steps,
        schedule_type="logistic",
        logistic_steepness=logistic_steepness,
    )


def get_fixed_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
) -> int:
    del step, end_rank, warmup_steps
    return int(start_rank)


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    assert G.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    compute_dtype = torch.bfloat16 if (use_bfloat16 and G.is_cuda) else torch.float32
    X = G.to(dtype=compute_dtype)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X.to(dtype=G.dtype)


@torch.no_grad()
def zeropower_via_lowrank_matrix_sign(
    G: torch.Tensor,
    steps: int = 10,
    rank: int = 200,
    oversample: int = 4,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    rescale: bool = False,
) -> torch.Tensor:
    assert G.ndim == 2, "Expected a 2D matrix after any conv flattening."

    if rank <= 0:
        return torch.zeros_like(G)

    X = G.float()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m, n = X.shape
    max_rank = min(m, n)
    sketch_dim = min(int(rank) + max(0, int(oversample)), max_rank)

    if sketch_dim >= max_rank:
        Z = zeropower_via_newtonschulz5(
            X,
            steps=steps,
            eps=eps,
            use_bfloat16=small_ns_bfloat16,
        ).float()
        if transposed:
            Z = Z.mT
        return Z.type_as(G)

    Omega = torch.randn(n, sketch_dim, device=X.device, dtype=X.dtype)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.mT @ X
    S = zeropower_via_newtonschulz5(
        B,
        steps=steps,
        eps=eps,
        use_bfloat16=small_ns_bfloat16,
    ).float()
    Z = Q @ S

    if rescale and sketch_dim > 0:
        Z = Z * math.sqrt(float(max_rank) / float(sketch_dim))
    if transposed:
        Z = Z.mT
    return Z.type_as(G)


def _round_up_to_multiple(value: int, multiple: int = 8) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(int(value) / float(multiple)) * multiple)


def _clamp_rank(value: int, floor_rank: int, ceil_rank: int) -> int:
    floor_rank = int(floor_rank)
    ceil_rank = max(floor_rank, int(ceil_rank))
    return max(floor_rank, min(int(value), ceil_rank))


@torch.no_grad()
def build_muon_search_matrix(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)
    return update


@torch.no_grad()
def preview_muon_search_matrix(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum_preview = momentum.detach().clone()
    grad_preview = grad.detach().clone()
    momentum_preview.lerp_(grad_preview, 1 - beta)
    update = grad_preview.lerp_(momentum_preview, beta) if nesterov else momentum_preview

    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)
    return update


@torch.no_grad()
def choose_auto_rank_start(
    update: torch.Tensor,
    floor_rank: int,
    probe_rank: int,
    energy_tau: float = 0.90,
    round_multiple: int = 8,
    eps: float = 1e-12,
) -> int:
    assert update.ndim == 2

    X = update.float()
    if X.size(-2) > X.size(-1):
        X = X.mT

    m, n = X.shape
    limit = min(m, n)
    floor_rank = max(1, min(int(floor_rank), limit))
    probe_rank = max(floor_rank, min(int(probe_rank), limit))

    Omega = torch.randn(n, probe_rank, device=X.device, dtype=X.dtype)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.mT @ X

    svals = torch.linalg.svdvals(B.float())
    if svals.numel() == 0:
        return floor_rank

    capture = torch.cumsum(svals.square(), dim=0) / (X.square().sum() + eps)
    if float(capture[-1].item()) < float(energy_tau):
        return probe_rank

    threshold = torch.tensor(float(energy_tau), device=capture.device, dtype=capture.dtype)
    rank_hat = int(torch.searchsorted(capture, threshold).item()) + 1
    rank_hat = _round_up_to_multiple(max(floor_rank, rank_hat), round_multiple)
    return _clamp_rank(rank_hat, floor_rank, probe_rank)


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,  #! 10?
    nesterov: bool = True,
    rank: int = 200,
    oversample: int = 4,
    lowrank_rescale: bool = False,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    step: int = 1,
    rank_start: int = 200,
    rank_end: int = 250,
    warmup_steps: int = 100000,
    current_rank: Optional[int] = None,
    rank_schedule: str = "cosine",
    exp_rate: float = 5.0,
    log_rate: float = 9.0,
    logistic_steepness: float = 12.0,
) -> torch.Tensor:
    update = build_muon_search_matrix(
        grad,
        momentum,
        beta=beta,
        nesterov=nesterov,
    )

    if current_rank is None:
        scheduled_rank = get_scheduled_rank(
            step=step,
            start_rank=rank_start,
            end_rank=rank_end,
            warmup_steps=warmup_steps,
            schedule_type=rank_schedule,
            exp_rate=exp_rate,
            log_rate=log_rate,
            logistic_steepness=logistic_steepness,
        )
        applied_rank = max(int(rank), int(scheduled_rank))
    else:
        applied_rank = int(current_rank)

    update = zeropower_via_lowrank_matrix_sign(
        update,
        steps=ns_steps,
        rank=applied_rank,
        oversample=oversample,
        eps=eps,
        small_ns_bfloat16=small_ns_bfloat16,
        rescale=lowrank_rescale,
    )
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceAutoScheduledRankWithAuxAdam(torch.optim.Optimizer):
    """
    Muon + auxiliary Adam with a selectable rank schedule.

    Subclasses select defaults through the DEFAULT_* class attributes. Every
    setting can still be overridden in the Muon parameter group.
    """

    DEFAULT_RANK_SCHEDULE = "cosine"
    DEFAULT_RANK = 200
    DEFAULT_RANK_START = 200
    DEFAULT_RANK_END = 250
    DEFAULT_WARMUP_STEPS = 100000

    def __init__(self, param_groups):
        normalized_groups = []

        for group in param_groups:
            assert "use_muon" in group, "Each param group must include use_muon=True/False."
            g = dict(group)

            if g["use_muon"]:
                g.setdefault("lr", 0.003)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 10)
                g.setdefault("nesterov", True)
                g.setdefault("rank", self.DEFAULT_RANK)
                g.setdefault("rank_start", self.DEFAULT_RANK_START)
                g.setdefault("rank_end", self.DEFAULT_RANK_END)
                g.setdefault("warmup_steps", self.DEFAULT_WARMUP_STEPS)
                g.setdefault("rank_schedule", self.DEFAULT_RANK_SCHEDULE)
                g.setdefault("exp_rate", 5.0)
                g.setdefault("log_rate", 9.0)
                g.setdefault("logistic_steepness", 12.0)
                g.setdefault("oversample", 4)
                g.setdefault("lowrank_rescale", False)
                g.setdefault("eps", 1e-6)
                g.setdefault("small_ns_bfloat16", False)
                g.setdefault("step", 0)
                g.setdefault("current_rank", g["rank_start"])
                g.setdefault("current_target_rank", g["rank_end"])
                g.setdefault("current_method", f"{g['rank_schedule']}_closed_form")

                g.setdefault("auto_init_rank_start", False)
                g.setdefault("init_probe_steps", 8)
                g.setdefault("init_energy", 0.90)
                g.setdefault("init_round_multiple", 8)
                g.setdefault("auto_rank_start_final", None)
                g.setdefault("_init_rank_candidates", [])

                if int(g["rank"]) <= 0:
                    raise ValueError("rank must be positive.")
                if int(g["rank_start"]) <= 0 or int(g["rank_end"]) <= 0:
                    raise ValueError("rank_start and rank_end must be positive.")
                if str(g["rank_schedule"]).lower() not in SUPPORTED_RANK_SCHEDULES:
                    supported = ", ".join(sorted(SUPPORTED_RANK_SCHEDULES))
                    raise ValueError(
                        f"Unknown rank schedule '{g['rank_schedule']}'. "
                        f"Supported schedules: {supported}."
                    )
                if (
                    bool(g["auto_init_rank_start"])
                    and int(g["rank_end"]) < int(g["rank"])
                ):
                    raise ValueError(
                        "auto_init_rank_start requires rank_end >= rank. "
                        "Disable it for a decreasing-rank schedule."
                    )
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)

            normalized_groups.append(g)

        super().__init__(normalized_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_muon_group(self, group):
        group_step = int(group.get("step", 0)) + 1
        group["step"] = group_step

        if bool(group["auto_init_rank_start"]) and group["auto_rank_start_final"] is None:
            self._update_auto_rank_start(group, group_step)

        scheduled_rank = get_scheduled_rank(
            step=group_step,
            start_rank=group["rank_start"],
            end_rank=group["rank_end"],
            warmup_steps=group["warmup_steps"],
            schedule_type=group["rank_schedule"],
            exp_rate=group["exp_rate"],
            log_rate=group["log_rate"],
            logistic_steepness=group["logistic_steepness"],
        )
        applied_rank = max(int(group["rank"]), int(scheduled_rank))

        group["current_rank"] = applied_rank
        group["current_target_rank"] = int(group["rank_end"])
        suffix = "auto_start" if bool(group["auto_init_rank_start"]) else "closed_form"
        group["current_method"] = f"{group['rank_schedule']}_{suffix}"

        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)

            update = muon_update(
                p.grad,
                state["momentum_buffer"],
                beta=group["momentum"],
                ns_steps=group["ns_steps"],
                nesterov=group["nesterov"],
                rank=group["rank"],
                oversample=group["oversample"],
                lowrank_rescale=group["lowrank_rescale"],
                eps=group["eps"],
                small_ns_bfloat16=group["small_ns_bfloat16"],
                current_rank=applied_rank,
            )
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update.reshape(p.shape), alpha=-group["lr"])

    def _update_auto_rank_start(self, group, group_step: int):
        if group_step <= int(group["init_probe_steps"]):
            step_candidates = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                search_matrix = preview_muon_search_matrix(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    nesterov=group["nesterov"],
                )
                candidate = choose_auto_rank_start(
                    search_matrix,
                    floor_rank=group["rank"],
                    probe_rank=group["rank_end"],
                    energy_tau=group["init_energy"],
                    round_multiple=group["init_round_multiple"],
                )
                step_candidates.append(int(candidate))

            if step_candidates:
                step_median = int(statistics.median(step_candidates))
                step_median = max(int(group["rank"]), step_median)
                step_median = min(int(group["rank_end"]), step_median)
                group["_init_rank_candidates"].append(step_median)

                provisional_start = int(statistics.median(group["_init_rank_candidates"]))
                group["rank_start"] = _clamp_rank(
                    _round_up_to_multiple(
                        provisional_start,
                        int(group["init_round_multiple"]),
                    ),
                    int(group["rank"]),
                    int(group["rank_end"]),
                )

        if group_step == int(group["init_probe_steps"]):
            if group["_init_rank_candidates"]:
                final_start = int(statistics.median(group["_init_rank_candidates"]))
                final_start = _clamp_rank(
                    _round_up_to_multiple(
                        final_start,
                        int(group["init_round_multiple"]),
                    ),
                    int(group["rank"]),
                    int(group["rank_end"]),
                )
            else:
                final_start = int(group["rank"])

            group["auto_rank_start_final"] = final_start
            group["rank_start"] = final_start

    def _step_adam_group(self, group):
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0

            state["step"] += 1
            update = adam_update(
                p.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["betas"],
                group["eps"],
            )
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])


# Backward-compatible name from the previously delivered lr_inc-based package.
SingleDeviceScheduledRankWithAuxAdam = SingleDeviceAutoScheduledRankWithAuxAdam


# Backward-compatible name used by the attached lr_inc.py.
SingleDeviceFinalIncWithAuxAdam = SingleDeviceAutoScheduledRankWithAuxAdam
