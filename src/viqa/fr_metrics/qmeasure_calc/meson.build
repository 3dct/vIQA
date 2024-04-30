# incdir_numpy = run_command('src/viqa/_build_utils/get_numpy_path.py', check: true).stdout().strip()
# incdir_numpy = 'C:/Users/p42938/.conda/envs/IQA_Library/Lib/site-packages/numpy/core/include'
incdir_numpy = '/opt/conda/lib/python3.11/site-packages/numpy/core/include'

sources = files(
'ImageHistogram.cpp',
'iAQMeasureCalculation.cpp',
'qmeasurecalcmodule.cpp',
)

py.extension_module(
    'qmeasurecalc',
    sources: sources,
    install: true,
    include_directories: incdir_numpy,
    subdir: 'viqa/fr_metrics/qmeasure_calc',
)