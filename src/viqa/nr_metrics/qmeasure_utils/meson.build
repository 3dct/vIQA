incdir_numpy = run_command('python', '../../_build_utils/get_numpy_path.py', check: true).stdout().strip()

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
    c_args: ['-std=c++20'],
)