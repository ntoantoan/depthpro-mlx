"""Point-cloud rendering and camera paths."""

from .camera import ease_loop, look_at, orbit_frames
from .splat import fill_holes_farthest, render_view

__all__ = ["ease_loop", "fill_holes_farthest", "look_at", "orbit_frames", "render_view"]
