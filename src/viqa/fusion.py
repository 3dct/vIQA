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

    Examples
    --------
    >>> from viqa import fuse_metrics_linear_combination, PSNR, SSIM
    >>> metrics = [PSNR, SSIM]
    >>> PSNR.score_val = 20.0
    >>> SSIM.score_val = 0.5
    >>> weights = [0.5, 0.5]
    >>> fuse_metrics_linear_combination(metrics, weights)
    10.25
    """
    return sum(m.score_val * w for m, w in zip(metrics, weights, strict=True))
