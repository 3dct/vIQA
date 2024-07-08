incdir_numpy = run_command('python', '../../_build_utils/get_numpy_path.py', check: true).stdout().strip()

py.extension_module(
    'statisticscalc',
    'statisticsmodule.c',
    install: true,
    include_directories: incdir_numpy,
    subdir: 'viqa/fr_metrics/stat_utils',
)