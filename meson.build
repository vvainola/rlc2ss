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
deps = [
    dependency('eigen3'),
    dependency('nlohmann_json'),
    dependency('symengine')
]
add_project_arguments('-DNOMINMAX', language : 'cpp')
lib_rlc2ss = static_library('rlc2ss',
    sources : [
        'src/netlist/graph.cpp',
        'src/netlist/netlist.cpp',
        'src/netlist/state_space.cpp',
        'src/netlist/str_helpers.cpp',
        'src/rlc2ss.cpp',
    ],
    include_directories : [
        'include',
        'src'
    ],
    dependencies : deps
)
generate_hexdump_headers = find_program('generate_hexdump_headers.py', dirs : [meson.current_source_dir() / 'scripts'])
# Expose the program to the parent project
meson.override_find_program('generate_hexdump_headers', generate_hexdump_headers)
if meson.is_subproject()
    lib_rlc2ss_dep = declare_dependency(
        link_with : lib_rlc2ss,
        include_directories : 'include',
        dependencies : deps
    )
    subdir_done()
endif

if WINDOWS
    gui = [subproject('gui').get_variable('lib_gui_dep')]
else
    # GUI is not available on non-Windows platforms
    gui = []
endif


# Embed the JSON file as a C array
circuit_json_headers  = custom_target('circuit_json_headers',
    input : [
        'qucs/diode_bridge_matrices.json',
        'qucs/diode_bridge_3l_matrices.json',
        'qucs/diode_matrices.json',
        'qucs/controlled_sources_matrices.json',
        'qucs/mutual_inductor_matrices.json',
    ],
    output : [
        'diode_bridge_matrices_json.h',
        'diode_bridge_3l_matrices_json.h',
        'diode_matrices_json.h',
        'controlled_sources_matrices_json.h',
        'mutual_inductor_matrices_json.h',
    ],
    command : [generate_hexdump_headers,
        '--input_files', '@INPUT0@', '@INPUT1@', '@INPUT2@', '@INPUT3@', '@INPUT4@',
        '--output_files', '@OUTPUT0@', '@OUTPUT1@', '@OUTPUT2@', '@OUTPUT3@', '@OUTPUT4@'
    ]
)

diode_bridge = executable('diode_bridge',
    sources : [
        'src/diode_bridge.cpp',
        'qucs/diode_bridge_matrices.cpp',
        circuit_json_headers
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
        circuit_json_headers
    ],
    include_directories: [
        'src',
        'include'
    ],
    link_with : [lib_rlc2ss],
    dependencies : deps + gui
)

diode_bridge_3L = executable('diode_bridge_3L',
    sources : [
        'src/diode_bridge_3L.cpp',
        'qucs/diode_bridge_3l_matrices.cpp',
        circuit_json_headers,
    ],
    include_directories: [
        'src',
        'include'
    ],
    link_with : [lib_rlc2ss],
    dependencies : deps + gui
)

executable('rlc2ss',
    sources : [
        'src/main.cpp',
    ],
    include_directories: [
        'src',
        'include'
    ],
    link_with : [lib_rlc2ss],
    dependencies : deps + gui + [
        dependency('cxxopts'),
    ]
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
