"""Provide the fusion of multiple image quality assessment metrics."""


def fuse_metrics_linear_combination(metrics, weights):
    """
    Fuse multiple image quality assessment metrics into a single score as a linear
    combination of the metrics.

    Parameters
    ----------
    metrics : list
        List of metric values.
    weights : list[float]
        List of weights for the metrics.

    Returns
    -------
    float
        Fused score.
    """
    return sum([m.score_val * w for m, w in zip(metrics, weights)])
