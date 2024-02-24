project('RLC_to_SS',
    'cpp',
    version : '0.8.0',
    default_options : [
        'cpp_std=c++20',
        'b_vscrt=md',
    ])
add_project_arguments('-DNOMINMAX', language : 'cpp')
windows = import('windows')

deps = [dependency('eigen3'), dependency('nlohmann_json')]
gui = subproject('gui').get_variable('lib_gui_dep')

diode_bridge_resource = windows.compile_resources('schematics/diode_bridge_matrices.rc', depend_files : 'schematics/diode_bridge_matrices.json')
diode_bridge = executable('diode_bridge',
    sources : [
        'src/diode_bridge.cpp',
        'src/rlc2ss.cpp',
        'schematics/diode_bridge_matrices.cpp',
        diode_bridge_resource
    ],
    include_directories: ['src'],
    dependencies : deps + [gui]
)

#transformer = executable('transformer',
#    sources : 'src/transformer.cpp',
#    dependencies : deps
#)

executable('mutual_inductance',
    sources : [
        'src/mutual_inductance.cpp',
        'src/rlc2ss.cpp',
        'schematics/RL3_matrices.cpp',
    ],
    include_directories: [
        'src'
    ],
    dependencies : deps
)

diode_bridge_3L_resource = windows.compile_resources('qucs/diode_bridge_3l_matrices.rc', depend_files : 'qucs/diode_bridge_3l_matrices.json')
diode_bridge_3L = executable('diode_bridge_3L',
    sources : [
        'src/diode_bridge_3L.cpp',
        'src/rlc2ss.cpp',
        'qucs/diode_bridge_3l_matrices.cpp',
        diode_bridge_3L_resource,
    ],
    include_directories: [
        'src'
    ],
    dependencies : deps + [gui]
)

executable('tests',
    sources : [
        'src/rlc2ss.cpp',
        'tests/expression_eval_test.cpp',
    ],
    include_directories: [
        'src'
    ],
    dependencies: dependency('catch2-with-main')
)
