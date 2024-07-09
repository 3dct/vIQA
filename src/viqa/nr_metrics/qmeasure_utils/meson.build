incdir_numpy = run_command('python', '../../_build_utils/get_numpy_path.py', check: true).stdout().strip()

sources = files(
'ImageHistogram.cpp',
'iAQMeasureCalculation.cpp',
'qmeasurecalcmodule.cpp',
)

if ccpp.get_id() == 'clang'
    extra_args = ['-std=c++20']
else
    extra_args = []
endif

py.extension_module(
    'qmeasurecalc',
    sources: sources,
    install: true,
    c_args: extra_args,
    include_directories: incdir_numpy,
    subdir: 'viqa/nr_metrics/qmeasure_utils',
)