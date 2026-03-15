from __future__ import annotations

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_updates: int,
        tot_updates: int,
        lr: float,
        end_lr: float,
        power: float,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_updates = max(int(warmup_updates), 0)
        self.tot_updates = max(int(tot_updates), 1)
        self.lr = float(lr)
        self.end_lr = float(end_lr)
        self.power = float(power)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_step(self) -> int:
        return max(self.last_epoch, 0)

    def _compute_lr(self, step: int) -> float:
        if self.warmup_updates > 0 and step <= self.warmup_updates:
            return self.lr * step / float(self.warmup_updates)
        if step >= self.tot_updates:
            return self.end_lr
        if self.tot_updates <= self.warmup_updates:
            return self.end_lr

        lr_range = self.lr - self.end_lr
        pct_remaining = 1.0 - (step - self.warmup_updates) / (self.tot_updates - self.warmup_updates)
        return lr_range * (pct_remaining ** self.power) + self.end_lr

    def get_lr(self):
        lr = self._compute_lr(self._get_step())
        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        lr = self._compute_lr(self._get_step())
        return [lr for _ in self.optimizer.param_groups]
