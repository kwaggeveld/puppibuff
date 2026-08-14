 # Adapted from:
#################
#    HLS4ML
#################
set tcldir [file dirname [info script]]
source [file join $tcldir hls_parameters.tcl]

array set opt {
    reset      0
    synth      1
    cosim      0
    export     0
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o=+(\\w+)" $arg unused opt($o)
  }
}

if {$opt(reset)} {
    open_project -reset ${prj_name}
} else {
    open_project ${prj_name}
}

set_top ${top}
add_files ../../firmware/${prj_name}.cpp -cflags "-std=c++0x"

if {$opt(reset)} {
    open_solution -reset "solution1" -flow_target ${flow_target}
} else {
    open_solution "solution1" -flow_target ${flow_target}
}

set_part ${part}
create_clock -period ${clock_period} -name default

config_interface -m_axi_addr64=${m_axi_addr64}

if {$opt(synth)} {
    csynth_design
}

if {$opt(cosim)} {
    cosim_design -trace_level all
}

if {$opt(export)} {
    export_design -vendor cern.ch -library conifer -ipname ${top} -version ${version} -format ${export_format}
}
exit
