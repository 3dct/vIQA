incdir_numpy = run_command('../../_build_utils/get_numpy_path.py', check: true).stdout().strip()
# incdir_numpy = '/home/lukas/.virtualenvs/vIQA_Package/lib/python3.12/site-packages/numpy/_core/include'
# incdir_numpy = '/opt/python/cp312-cp312/lib/python3.12/site-packages/numpy/_core/include'

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
    subdir: 'viqa/nr_metrics/qmeasure_utils',
)