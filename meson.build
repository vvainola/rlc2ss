project('rlc2ss',
    'cpp',
    version : '0.8.0',
    default_options : [
        'cpp_std=c++20',
        'b_vscrt=md',
        'warning_level=3',
    ])
WINDOWS = (host_machine.system() == 'windows')
LINUX = (host_machine.system() == 'linux')
deps = [dependency('eigen3'), dependency('nlohmann_json')]
add_project_arguments('-DNOMINMAX', language : 'cpp')
lib_rlc2ss = static_library('rlc2ss',
    sources : [
        'src/rlc2ss.cpp',
    ],
    include_directories : [
        'include',
        'src'
    ],
)
if meson.is_subproject()
    lib_rlc2ss_dep = declare_dependency(
        link_with : lib_rlc2ss,
        include_directories : 'include',
        dependencies : deps
    )
    subdir_done()
endif

windows = import('windows')

if WINDOWS
    gui = [subproject('gui').get_variable('lib_gui_dep')]
else
    # GUI is not available on non-Windows platforms
    gui = []
endif

if WINDOWS
    diode_bridge_resource = windows.compile_resources('qucs/diode_bridge_matrices.rc', depend_files : 'qucs/diode_bridge_matrices.json')
else
    # Embed the JSON file as a C array
    diode_bridge_json_h = custom_target(
        'diode_bridge_json_h',
        input : 'qucs/diode_bridge_matrices.json',
        output : 'diode_bridge_matrices_json.h',
        command : [
            'xxd', '-n', 'diode_bridge_matrices_json', '-i', '@INPUT@', '@OUTPUT@'
        ]
    )
    diode_bridge_resource = diode_bridge_json_h
endif

diode_bridge = executable('diode_bridge',
    sources : [
        'src/diode_bridge.cpp',
        'qucs/diode_bridge_matrices.cpp',
        diode_bridge_resource
    ],
    link_with : [lib_rlc2ss],
    include_directories: [
        'src',
        'include'
    ],
    dependencies : deps + [gui]
)

if LINUX
    subdir_done()
endif

shared_library('circuit_tester',
    sources : [
        'qucs/controlled_sources_matrices.cpp',
        'qucs/diode_matrices.cpp',
        'qucs/mutual_inductor_matrices.cpp',
        'qucs/saturating_inductor_matrices.cpp',
        'qucs/converter_matrices.cpp',
        'src/circuit_tester.cpp',
    ],
    include_directories: [
        'src',
        'include'
    ],
    link_with : [lib_rlc2ss],
    dependencies : deps + gui
)

diode_bridge_3L_resource = windows.compile_resources('qucs/diode_bridge_3l_matrices.rc', depend_files : 'qucs/diode_bridge_3l_matrices.json')
diode_bridge_3L = executable('diode_bridge_3L',
    sources : [
        'src/diode_bridge_3L.cpp',
        'qucs/diode_bridge_3l_matrices.cpp',
        diode_bridge_3L_resource,
    ],
    include_directories: [
        'src',
        'include'
    ],
    link_with : [lib_rlc2ss],
    dependencies : deps + gui
)

executable('tests',
    sources : [
        'tests/expression_eval_test.cpp',
    ],
    include_directories: [
        'src',
        'include'
    ],
    link_with : [lib_rlc2ss],
    dependencies: dependency('catch2-with-main')
)
