python_sources = [
    '__init__.py',
    'fsim.py',
    'gsm.py',
    'mad.py',
    'msssim.py',
    'psnr.py',
    'rmse.py',
    'ssim.py',
    'vif.py',
    'vsi.py',
    'uqi.py',
    'qmeasure.py',
]

py.install_sources(
    python_sources,
    pure: false,
    subdir: 'viqa/fr_metrics',
)

subdir('stat_utils')
subdir('qmeasure_calc')