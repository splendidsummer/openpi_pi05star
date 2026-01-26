#!/usr/bin/env python3
"""
Visualization utilities for openpi models and robot data.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Union
from pathlib import Path


def visualize_robot_images(
    images: Dict[str, np.ndarray],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Visualize multiple camera views from robot observations.

    Args:
        images: Dictionary mapping camera names to image arrays (H, W, C)
        title: Optional title for the figure
        save_path: Optional path to save the figure
        figsize: Optional figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided")

    # Calculate grid layout
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols

    if figsize is None:
        figsize = (5 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (cam_name, img) in enumerate(images.items()):
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(cam_name)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def plot_action_trajectory(
    actions: np.ndarray,
    action_names: Optional[List[str]] = None,
    title: str = "Action Trajectory",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot action trajectories over time.

    Args:
        actions: Action array of shape (time_steps, action_dim)
        action_names: Optional list of names for each action dimension
        title: Title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {actions.shape}")

    time_steps, action_dim = actions.shape

    if action_names is None:
        action_names = [f"Action {i}" for i in range(action_dim)]
    elif len(action_names) != action_dim:
        raise ValueError(f"Expected {action_dim} action names, got {len(action_names)}")

    fig, axes = plt.subplots(action_dim, 1, figsize=figsize, sharex=True)
    if action_dim == 1:
        axes = [axes]

    time_indices = np.arange(time_steps)

    for idx, (ax, name) in enumerate(zip(axes, action_names)):
        ax.plot(time_indices, actions[:, idx], linewidth=2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    return fig


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot training metrics over time (e.g., loss, accuracy).

    Args:
        metrics: Dictionary mapping metric names to lists of values
        title: Title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        log_scale: Whether to use log scale for y-axis

    Returns:
        matplotlib Figure object
    """
    if not metrics:
        raise ValueError("No metrics provided")

    fig, ax = plt.subplots(figsize=figsize)

    for metric_name, values in metrics.items():
        steps = np.arange(len(values))
        ax.plot(steps, values, label=metric_name, linewidth=2, marker='o', markersize=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")

    return fig


def compare_actions(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    action_names: Optional[List[str]] = None,
    title: str = "Predicted vs Ground Truth Actions",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Compare predicted actions against ground truth.

    Args:
        predicted: Predicted actions of shape (time_steps, action_dim)
        ground_truth: Ground truth actions of shape (time_steps, action_dim)
        action_names: Optional list of names for each action dimension
        title: Title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs ground_truth {ground_truth.shape}"
        )

    if predicted.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shape {predicted.shape}")

    time_steps, action_dim = predicted.shape

    if action_names is None:
        action_names = [f"Action {i}" for i in range(action_dim)]
    elif len(action_names) != action_dim:
        raise ValueError(f"Expected {action_dim} action names, got {len(action_names)}")

    fig, axes = plt.subplots(action_dim, 1, figsize=figsize, sharex=True)
    if action_dim == 1:
        axes = [axes]

    time_indices = np.arange(time_steps)

    for idx, (ax, name) in enumerate(zip(axes, action_names)):
        ax.plot(time_indices, ground_truth[:, idx], label="Ground Truth",
                linewidth=2, alpha=0.7, color='blue')
        ax.plot(time_indices, predicted[:, idx], label="Predicted",
                linewidth=2, alpha=0.7, color='red', linestyle='--')
        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig
