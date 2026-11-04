"""Render an LNS-RL search trajectory as an animated GIF."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROUTE_COLORS = (
    "#2563EB",
    "#DC2626",
    "#059669",
    "#7C3AED",
    "#EA580C",
    "#0891B2",
    "#DB2777",
    "#65A30D",
    "#4F46E5",
    "#CA8A04",
)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _point_transform(
    coordinates: np.ndarray, width: int, height: int
) -> list[tuple[int, int]]:
    margin_left, margin_right = 55, 35
    margin_top, margin_bottom = 105, 55
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    x_span = max(float(np.ptp(x_values)), 1.0)
    y_span = max(float(np.ptp(y_values)), 1.0)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    scale = min(plot_width / x_span, plot_height / y_span)
    used_width = x_span * scale
    used_height = y_span * scale
    x_offset = margin_left + (plot_width - used_width) / 2.0
    y_offset = margin_top + (plot_height - used_height) / 2.0
    return [
        (
            round(x_offset + (float(x) - float(x_values.min())) * scale),
            round(height - margin_bottom - (plot_height - used_height) / 2.0
                  - (float(y) - float(y_values.min())) * scale),
        )
        for x, y in coordinates
    ]


def _frame(
    instance_data: np.ndarray,
    routes: Sequence[Sequence[int]],
    iteration: int,
    distance: float,
    total_iterations: int,
    width: int,
    height: int,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    body_font = _font(17)
    label_font = _font(14)
    points = _point_transform(instance_data[:, :2], width, height)

    draw.text((28, 18), "LNS-RL search", fill="#0F172A", font=title_font)
    draw.text(
        (28, 51),
        f"Iteration {iteration:,} / {total_iterations:,}",
        fill="#334155",
        font=body_font,
    )
    draw.text(
        (width - 238, 51),
        f"Best distance: {distance:,.2f}",
        fill="#334155",
        font=body_font,
    )

    bar_left, bar_top, bar_right, bar_bottom = 28, 75, width - 28, 83
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_right, bar_bottom), radius=4, fill="#E2E8F0"
    )
    progress = iteration / max(total_iterations, 1)
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_left + (bar_right - bar_left) * progress, bar_bottom),
        radius=4,
        fill="#2563EB",
    )

    depot = points[0]
    for route_index, route in enumerate(routes):
        color = ROUTE_COLORS[route_index % len(ROUTE_COLORS)]
        route_points = [depot, *(points[int(node)] for node in route), depot]
        draw.line(route_points, fill=color, width=3, joint="curve")

    for point in points[1:]:
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#FFFFFF", outline="#0F172A", width=2)

    x, y = depot
    draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill="#0F172A")
    draw.text((x + 11, y - 9), "Depot", fill="#0F172A", font=label_font)
    return image


def save_search_animation(
    instance_data: np.ndarray,
    snapshots: Sequence[tuple[int, float, Sequence[Sequence[int]]]],
    output: str | Path,
    total_iterations: int,
    *,
    duration_ms: int = 180,
    width: int = 900,
    height: int = 650,
) -> None:
    """Save sampled best solutions from one search as an animated GIF."""
    if not snapshots:
        raise ValueError("At least one search snapshot is required")

    frames = [
        _frame(instance_data, routes, iteration, distance, total_iterations, width, height)
        for iteration, distance, routes in snapshots
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
