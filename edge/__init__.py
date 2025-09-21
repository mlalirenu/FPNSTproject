
# Advanced Edge Detection Module:
# Provides multiple sophisticated edge detection methods for preservation masks.


from edge.directional_gradient import DirectionalGradientFusion
from edge.colorspace_saliency import ColorSpaceSaliencyMaps
from edge.bayesian_confidence import BayesianConfidenceBlender

__all__ = [
    'HEDEdgeDetector',
    'DirectionalGradientFusion',
    'ColorSpaceSaliencyMaps',
    'BayesianConfidenceBlender'
]

__version__ = "1.0.0"