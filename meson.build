project(
    'viqa',
    'c', 'cpp',
    version: run_command('python', 'src/viqa/utils/_build/version.py', check: true).stdout().strip(),
    license: 'BSD-3',
    default_options: [
        'buildtype=release',
        'c_std=c11',
        'cpp_std=c++20',
    ],
    meson_version: '>=1.3.0',
)

cc = meson.get_compiler('c')
ccpp = meson.get_compiler('cpp')

if ccpp.get_id() == 'clang'
    libs_compiler = ['-stdlib=libc++', '-mmacosx-version-min=13.3']
    libs_linker   = ['-stdlib=libc++', '-mmacosx-version-min=13.3']

    add_project_arguments(libs_compiler, language: 'cpp')
    add_project_link_arguments(libs_linker, language: 'cpp')
endif

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
# NOTE: with Meson >=0.64.0 we can add `pure: false` here and remove that line
# everywhere else, see https://github.com/mesonbuild/meson/pull/10783.
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

subdir('src/viqa')
