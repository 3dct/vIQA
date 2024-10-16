incdir_numpy = run_command('python', '../../utils/_build/get_numpy_path.py', check: true).stdout().strip()

python_sources = files(
'statisticscalc.pyi',
)

py.extension_module(
    'statisticscalc',
    'statisticsmodule.c',
    install: true,
    include_directories: incdir_numpy,
    subdir: 'viqa/fr_metrics/stat_utils',
)

py.install_sources(
    python_sources,
    pure: false,
    subdir: 'viqa/fr_metrics/stat_utils',
)
