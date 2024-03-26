incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

py.extension_module(
    'statistics',
    'statisticsmodule.c',
    install: true,
    include_directories: incdir_numpy,
    subdir: 'viqa/fr_metrics',
)