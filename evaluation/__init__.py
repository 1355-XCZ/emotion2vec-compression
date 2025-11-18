"""
Evaluation module
Contains emotion2vec classifier, bitrate sweep, layer sweep and result analysis
"""

try:
    from .emotion_classifier import EmotionClassifierV2
    from .method_rate_sweep import rate_sweep_evaluation
    from .method_layer_sweep import layer_sweep_evaluation
    from .analyzer import ResultAnalyzer
except ImportError as e:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path
    parent = Path(__file__).parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    
    from emotion_classifier import EmotionClassifierV2
    from method_rate_sweep import rate_sweep_evaluation
    from method_layer_sweep import layer_sweep_evaluation
    from analyzer import ResultAnalyzer

__all__ = [
    'EmotionClassifierV2',
    'rate_sweep_evaluation',
    'layer_sweep_evaluation',
    'ResultAnalyzer',
]

