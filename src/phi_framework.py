"""
phi_framework.py
----------------
Standalone phi training framework for PyTorch classification pipelines.

Phi is a per-class fidelity scalar that tracks constraint satisfaction
during training and scales loss amplification accordingly.

Low phi  -> class constraint is violated -> aggressive learning pressure
High phi -> class constraint is satisfied -> gentle reinforcement

The amplifier fires on every true label occurrence regardless of prediction
correctness. No confidence gate. The true label is always the cleanest signal.

Velocity modulation: if phi is rising fast the system backs off.
If phi is stuck or falling the system pushes harder. Alpha and beta
emerge from dynamics rather than being manually tuned constants.

Usage:
    from phi_framework import PhiTracker

    tracker = PhiTracker(
        num_classes=8,
        class_names=le.classes_,
        baseline_report_path="results/baseline_report.json",
    )

    for epoch in range(EPOCHS):
        for xb, yb in train_loader:
            out             = model(xb)
            loss_per_sample = criterion(out, yb)    # reduction='none'
            amplifier       = tracker.get_amplifier(yb)
            loss            = (loss_per_sample * amplifier).mean()
            loss.backward()
            optimizer.step()
            tracker.accumulate(out.argmax(1), yb)
        tracker.update()
        print(tracker.phi_state())
"""

import json
import torch
from typing import List, Optional


class PhiTracker:
    """
    Tracks per-class fidelity (phi) and computes loss amplifiers.

    Parameters
    ----------
    num_classes : int
        Number of classification classes.
    class_names : list of str
        Class names in label-encoded order.
    baseline_report_path : str, optional
        Path to baseline classification_report JSON.
        If provided, phi is initialized from baseline F1 scores.
        If None, phi is initialized uniformly at 0.5.
    alpha_max : float
        Maximum aggression when phi is low. Default 3.0.
        Raise for severe imbalance. Lower for noisy labels.
    beta_min : float
        Minimum reinforcement when phi is high. Default 0.3.
        Never set to 0 — always maintain subtle reinforcement.
    vel_scale : float
        Velocity modulation strength. Default 5.0.
        Do not lower below 3.0 — causes training instability.
    vel_clamp_lo : float
        Floor on velocity factor. Default 0.5.
        Required to prevent negative amplifiers.
    vel_clamp_hi : float
        Ceiling on velocity factor. Default 2.0.
        Required to prevent gradient explosion.
    ema_alpha : float
        EMA smoothing for phi update. Default 0.8.
        Higher = slower phi change = more stable.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        baseline_report_path: Optional[str] = None,
        alpha_max: float = 3.0,
        beta_min: float = 0.3,
        vel_scale: float = 5.0,
        vel_clamp_lo: float = 0.5,
        vel_clamp_hi: float = 2.0,
        ema_alpha: float = 0.8,
    ):
        self.num_classes   = num_classes
        self.class_names   = list(class_names)
        self.alpha_max     = alpha_max
        self.beta_min      = beta_min
        self.vel_scale     = vel_scale
        self.vel_clamp_lo  = vel_clamp_lo
        self.vel_clamp_hi  = vel_clamp_hi
        self.ema_alpha     = ema_alpha

        if baseline_report_path is not None:
            with open(baseline_report_path) as f:
                baseline = json.load(f)
            self.phi = torch.tensor(
                [baseline[cls]['f1-score'] for cls in self.class_names],
                dtype=torch.float32
            )
            print("Phi initialized from baseline F1:")
            for c, cls in enumerate(self.class_names):
                print(f"  {cls:<14}: {self.phi[c].item():.4f}")
        else:
            self.phi = torch.ones(num_classes, dtype=torch.float32) * 0.5
            print("Phi initialized uniformly at 0.5")

        self.phi_history   = [self.phi.clone()]
        self.class_correct = torch.zeros(num_classes)
        self.class_total   = torch.zeros(num_classes)

    def get_amplifier(self, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample loss amplifier from true labels and current phi.

        Fires on every true label occurrence regardless of prediction.
        Low phi class  -> high amplifier -> aggressive learning.
        High phi class -> low amplifier  -> gentle reinforcement.
        Velocity modulates: rising phi backs off, stuck phi pushes harder.

        Parameters
        ----------
        true_labels : torch.Tensor
            Ground truth class indices for the current batch.

        Returns
        -------
        torch.Tensor
            Per-sample amplifier, same length as true_labels.
        """
        amplifier = torch.ones(len(true_labels), device=true_labels.device)

        if len(self.phi_history) >= 2:
            phi_vel = self.phi_history[-1] - self.phi_history[-2]
        else:
            phi_vel = torch.zeros(self.num_classes)

        phi_norm = self.phi.clamp(0, 1)

        for i in range(len(true_labels)):
            c     = true_labels[i].item()
            phi_c = phi_norm[c].item()
            vel_c = phi_vel[c].item()

            base = self.alpha_max * (1 - phi_c) + self.beta_min * phi_c
            vf   = 1.0 - vel_c * self.vel_scale
            vf   = max(self.vel_clamp_lo, min(self.vel_clamp_hi, vf))

            amplifier[i] = base * vf

        return amplifier

    def accumulate(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor
    ) -> None:
        """
        Accumulate per-class correct predictions for this batch.
        Call after each training batch.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted class indices (argmax of logits).
        true_labels : torch.Tensor
            Ground truth class indices.
        """
        preds  = predictions.cpu()
        labels = true_labels.cpu()
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                self.class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                self.class_total[c]   += mask.sum().item()

    def update(self) -> None:
        """
        Update phi from accumulated epoch statistics via EMA.
        Call once per epoch after all batches. Resets accumulators.
        """
        for c in range(self.num_classes):
            if self.class_total[c] > 0:
                acc         = self.class_correct[c] / self.class_total[c]
                self.phi[c] = self.ema_alpha * self.phi[c] + (1 - self.ema_alpha) * acc

        self.phi_history.append(self.phi.clone())
        self.class_correct.zero_()
        self.class_total.zero_()

    def phi_state(self) -> dict:
        """Return current phi values as a readable dictionary."""
        return {
            self.class_names[c]: round(self.phi[c].item(), 3)
            for c in range(self.num_classes)
        }

    def most_violated(self, n: int = 3) -> List[str]:
        """Return the n classes with lowest phi."""
        sorted_idx = self.phi.argsort()
        return [self.class_names[i.item()] for i in sorted_idx[:n]]

    def all_satisfied(self, threshold: float = 0.92) -> bool:
        """Check whether all classes have phi above the threshold."""
        return all(
            self.phi[c].item() >= threshold
            for c in range(self.num_classes)
        )

    def unsatisfied_classes(self, threshold: float = 0.92) -> List[str]:
        """Return class names still below the satisfaction threshold."""
        return [
            self.class_names[c]
            for c in range(self.num_classes)
            if self.phi[c].item() < threshold
        ]
