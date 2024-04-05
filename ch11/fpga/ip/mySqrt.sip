set_global_assignment -entity "mySqrt" -library "lib_mySqrt" -name IP_TOOL_NAME "altera_fp_functions"
set_global_assignment -entity "mySqrt" -library "lib_mySqrt" -name IP_TOOL_VERSION "19.1"
set_global_assignment -entity "mySqrt" -library "lib_mySqrt" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_mySqrt" -name SPD_FILE [file join $::quartus(sip_path) "mySqrt.spd"]

set_global_assignment -library "lib_mySqrt" -name MISC_FILE [file join $::quartus(sip_path) "mySqrt_sim/mySqrt_memoryC1_uid65_sqrtTables_lutmem.hex"]
set_global_assignment -library "lib_mySqrt" -name MISC_FILE [file join $::quartus(sip_path) "mySqrt_sim/mySqrt_memoryC2_uid68_sqrtTables_lutmem.hex"]
set_global_assignment -library "lib_mySqrt" -name MISC_FILE [file join $::quartus(sip_path) "mySqrt_sim/mySqrt_memoryC0_uid62_sqrtTables_lutmem.hex"]
set_global_assignment -library "lib_mySqrt" -name MISC_FILE [file join $::quartus(sip_path) "mySqrt_sim/mySqrt.vo"]
