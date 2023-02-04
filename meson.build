project('RLC_to_SS',
    'cpp',
    version : '0.8.0',
    default_options : [
        'cpp_std=c++20',
        'b_vscrt=md',
    ])
add_project_arguments('-DNOMINMAX', language : 'cpp')

configure_file(
    input : 'Directory.Build.Props.in',
    output : 'Directory.Build.Props',
    configuration :
    {
        'conan_bin': meson.current_build_dir()/'conan/bin',
        'build_root' : meson.build_root()
    }
)

deps = [dependency('eigen3')]

diode_bridge = executable('diode_bridge',
    sources : 'src/diode_bridge.cpp',
    dependencies : deps
)

lcl_with_faults = executable('lcl_with_faults',
    sources : 'src/lcl_with_faults.cpp',
    dependencies : deps
)

lcl = executable('lcl',
    sources : 'src/lcl.cpp',
    dependencies : deps
)

transformer = executable('transformer',
    sources : 'src/transformer.cpp',
    dependencies : deps
)

gui = subproject('gui').get_variable('lib_gui_dep')
inverter_3L = executable('inverter_3L',
    sources : ['src/inverter_3L.cpp'],
    dependencies : deps + [gui]
)
