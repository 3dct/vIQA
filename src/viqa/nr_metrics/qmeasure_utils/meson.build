incdir_numpy = run_command('python', '../../utils/_build/get_numpy_path.py', check: true).stdout().strip()
incdir_qmeasure = './qmeasure/src'
inc_dir = include_directories(incdir_numpy, incdir_qmeasure)

sources = files(
'qmeasure/src/ImageHistogram.cpp',
'qmeasure/src/QMeasureCalculation.cpp',
'qmeasurecalcmodule.cpp',
)

python_sources = files(
'qmeasurecalc.pyi',
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
    include_directories: inc_dir,
    subdir: 'viqa/nr_metrics/qmeasure_utils',
)

py.install_sources(
    python_sources,
    pure: false,
    subdir: 'viqa/nr_metrics/qmeasure_utils',
)
