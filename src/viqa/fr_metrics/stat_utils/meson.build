incdir_numpy = run_command('src/viqa/_build_utils/get_numpy_path.py', check: true).stdout().strip()

py.extension_module(
    'statistics',
    'statisticsmodule.c',
    install: true,
    include_directories: incdir_numpy,
    subdir: 'viqa/fr_metrics',
)