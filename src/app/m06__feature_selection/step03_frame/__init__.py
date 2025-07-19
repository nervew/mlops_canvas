# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/
#    m06__feature_selection/step03_frame/__init__.py

"""
step03_frame
────────────
Selección híbrida de características (forward + RFE),
ahora configurable con cualquier estimador.
"""

from .core.frame_selector import FrameSelector, frame_partitions

__all__ = ["FrameSelector", "frame_partitions"]
