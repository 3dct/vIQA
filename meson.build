project('viqa', 'c',
    version: '0.4.0',
    license: 'BSD-3',
    default_options: [
        'c_std=c99',
    ],
)

cc = meson.get_compiler('c')

_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-function',
)
add_project_arguments(_global_c_args, language : 'c')

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for vIQA). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it. For Fortran code, Meson already adds `-lm`.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
# NOTE: with Meson >=0.64.0 we can add `pure: false` here and remove that line
# everywhere else, see https://github.com/mesonbuild/meson/pull/10783.
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

subdir('src/viqa/metrics/stat_utils')