incdir_numpy = run_command('../../_build_utils/get_numpy_path.py', check: true).stdout().strip()
# incdir_numpy = '/home/lukas/.virtualenvs/vIQA_Package/lib/python3.12/site-packages/numpy/_core/include'
# incdir_numpy = '/opt/python/cp312-cp312/lib/python3.12/site-packages/numpy/_core/include'

py.extension_module(
    'statisticscalc',
    'statisticsmodule.c',
    install: true,
    include_directories: incdir_numpy,
    subdir: 'viqa/fr_metrics/stat_utils',
)