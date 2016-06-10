# My tcl script 
# Project automation script for fe65p2 
# 
# Created for ISE version 14.7

proc main {} {

   puts "I'm alive"
   set index(0) 160
   set index(1) 150
   set index(2) 140
   set index(3) 130
   set index(4) 120
   set index(5) 110
   set index(6) 100
   set index(7) 90
   set index(8) 80
   set index(9) 70
   set index(10) 60
   set index(11) 50
   set index(12) 40
   set index(13) 30
   set index(14) 20
   set index(15) 10

   set dfx2(160) 2
   set dfx2(150) 16
   set dfx2(140) 8
   set dfx2(130) 16
   set dfx2(120) 4
   set dfx2(110) 16
   set dfx2(100) 8
   set dfx2(90) 16
   set dfx2(80) 4
   set dfx2(70) 16
   set dfx2(60) 8
   set dfx2(50) 16
   set dfx2(40) 8
   set dfx2(30) 16
   set dfx2(20) 16
   set dfx2(10) 32
      
   set mfx2(160) 2
   set mfx2(150) 15
   set mfx2(140) 7
   set mfx2(130) 13
   set mfx2(120) 3
   set mfx2(110) 11
   set mfx2(100) 5
   set mfx2(90) 9 
   set mfx2(80) 2
   set mfx2(70) 7
   set mfx2(60) 3
   set mfx2(50) 5
   set mfx2(40) 2
   set mfx2(30) 3
   set mfx2(20) 2
   set mfx2(10) 2

   set myProject "fe65p2"
   set myScript "fe65p2.tcl"

      if { ! [ file exists ${myProject}.xise ] } { 
      ## project file isn't there, rebuild it.
      puts "Project $myProject not found. Use project_rebuild to recreate it."
      return false
   }

   project open $myProject
   puts "Project opened."


   # Set the Top Module as well...
   project set top "fe65p2_mio"

   puts "$myScript: project sources reloaded."

   for {set i 5} {$i<6} {incr i} {
#		set DCM2FX_div $dfx($index($i))
#		set DCM2FX_mult $mfx($index($i))
#		set DCM2DV_div $ddv($index($i))

		set FX2_div $dfx2($index($i))
		set FX2_mult $mfx2($index($i))
	
   puts "Generating $index($i) with d2 $FX2_div, m2 $FX2_mult"
   project set "Compiled Library Directory" "\$XILINX/<language>/<simulator>"
   project set "Multiplier Style" "Auto" -process "Synthesize - XST"
   project set "DCI Update Mode" "As Required" -process "Generate Programming File"
   project set "Configuration Rate" "Default (6)" -process "Generate Programming File"
   project set "Number of Clock Buffers" "8" -process "Synthesize - XST"
   project set "Max Fanout" "100000" -process "Synthesize - XST"
   project set "Regenerate Core" "Under Current Project Setting" -process "Regenerate Core"
   project set "Filter Files From Compile Order" "true"
   project set "Last Applied Goal" "Timing Performance"
   project set "Last Applied Strategy" "Performance with IOB Packing;/cadence/xilinx/14.7/ISE_DS/ISE/spartan3/data/spartan3_performance_with_iobpacking.xds"
   project set "Last Unlock Status" "false"
   project set "Manual Compile Order" "false"
   project set "Report Fastest Path(s) in Each Constraint" "true" -process "Generate Post-Place & Route Static Timing"
   project set "Generate Datasheet Section" "true" -process "Generate Post-Place & Route Static Timing"
   project set "Generate Timegroups Section" "false" -process "Generate Post-Place & Route Static Timing"
   project set "Report Fastest Path(s) in Each Constraint" "true" -process "Generate Post-Map Static Timing"
   project set "Generate Datasheet Section" "true" -process "Generate Post-Map Static Timing"
   project set "Generate Timegroups Section" "false" -process "Generate Post-Map Static Timing"
   project set "Project Description" ""
   project set "Property Specification in Project File" "Store all values"
   project set "Case Implementation Style" "None" -process "Synthesize - XST"
   project set "Decoder Extraction" "true" -process "Synthesize - XST"
   project set "Priority Encoder Extraction" "Yes" -process "Synthesize - XST"
   project set "Mux Extraction" "Yes" -process "Synthesize - XST"
   project set "RAM Extraction" "true" -process "Synthesize - XST"
   project set "ROM Extraction" "true" -process "Synthesize - XST"
   project set "FSM Encoding Algorithm" "Auto" -process "Synthesize - XST"
   project set "Logical Shifter Extraction" "true" -process "Synthesize - XST"
   project set "Optimization Goal" "Speed" -process "Synthesize - XST"
   project set "Optimization Effort" "High" -process "Synthesize - XST"
   project set "Resource Sharing" "true" -process "Synthesize - XST"
   project set "Shift Register Extraction" "true" -process "Synthesize - XST"
   project set "XOR Collapsing" "true" -process "Synthesize - XST"
   project set "User Browsed Strategy Files" "/cadence/xilinx/14.7/ISE_DS/ISE/data/default.xds"
   project set "VHDL Source Analysis Standard" "VHDL-93"
   project set "Input TCL Command Script" "" -process "Generate Text Power Report"
   project set "Load Physical Constraints File" "Default" -process "Analyze Power Distribution (XPower Analyzer)"
   project set "Load Physical Constraints File" "Default" -process "Generate Text Power Report"
   project set "Load Simulation File" "Default" -process "Analyze Power Distribution (XPower Analyzer)"
   project set "Load Simulation File" "Default" -process "Generate Text Power Report"
   project set "Load Setting File" "" -process "Analyze Power Distribution (XPower Analyzer)"
   project set "Load Setting File" "" -process "Generate Text Power Report"
   project set "Setting Output File" "" -process "Generate Text Power Report"
   project set "Produce Verbose Report" "false" -process "Generate Text Power Report"
   project set "Other XPWR Command Line Options" "" -process "Generate Text Power Report"
   project set "Other Bitgen Command Line Options" "" -process "Generate Programming File"
   project set "Maximum Signal Name Length" "20" -process "Generate IBIS Model"
   project set "Show All Models" "false" -process "Generate IBIS Model"
   project set "Launch SDK after Export" "true" -process "Export Hardware Design To SDK with Bitstream"
   project set "Launch SDK after Export" "true" -process "Export Hardware Design To SDK without Bitstream"
   project set "Target UCF File Name" "" -process "Back-annotate Pin Locations"
   project set "Ignore User Timing Constraints" "false" -process "Map"
   project set "Use RLOC Constraints" "Yes" -process "Map"
   project set "Other Map Command Line Options" "" -process "Map"
   project set "Use LOC Constraints" "true" -process "Translate"
   project set "Other Ngdbuild Command Line Options" "" -process "Translate"
   project set "Use 64-bit PlanAhead on 64-bit Systems" "true" -process "Floorplan Area/IO/Logic (PlanAhead)"
   project set "Use 64-bit PlanAhead on 64-bit Systems" "true" -process "I/O Pin Planning (PlanAhead) - Pre-Synthesis"
   project set "Use 64-bit PlanAhead on 64-bit Systems" "true" -process "I/O Pin Planning (PlanAhead) - Post-Synthesis"
   project set "Ignore User Timing Constraints" "false" -process "Place & Route"
   project set "Other Place & Route Command Line Options" "" -process "Place & Route"
   project set "Configuration Clk (Configuration Pins)" "Pull Up" -process "Generate Programming File"
   project set "UserID Code (8 Digit Hexadecimal)" "0xFFFFFFFF" -process "Generate Programming File"
   project set "Reset DCM if SHUTDOWN & AGHIGH performed" "false" -process "Generate Programming File"
   project set "Configuration Pin Done" "Pull Up" -process "Generate Programming File"
   project set "Create ASCII Configuration File" "false" -process "Generate Programming File"
   project set "Create Bit File" "true" -process "Generate Programming File"
   project set "Enable BitStream Compression" "false" -process "Generate Programming File"
   project set "Run Design Rules Checker (DRC)" "true" -process "Generate Programming File"
   project set "Enable Cyclic Redundancy Checking (CRC)" "true" -process "Generate Programming File"
   project set "Create IEEE 1532 Configuration File" "false" -process "Generate Programming File"
   project set "Configuration Pin HSWAPEN" "Pull Up" -process "Generate Programming File"
   project set "Configuration Pin M0" "Pull Up" -process "Generate Programming File"
   project set "Configuration Pin M1" "Pull Up" -process "Generate Programming File"
   project set "Configuration Pin M2" "Pull Up" -process "Generate Programming File"
   project set "Configuration Pin Program" "Pull Up" -process "Generate Programming File"
   project set "JTAG Pin TCK" "Pull Up" -process "Generate Programming File"
   project set "JTAG Pin TDI" "Pull Up" -process "Generate Programming File"
   project set "JTAG Pin TDO" "Pull Up" -process "Generate Programming File"
   project set "JTAG Pin TMS" "Pull Up" -process "Generate Programming File"
   project set "Unused IOB Pins" "Pull Down" -process "Generate Programming File"
   project set "Security" "Enable Readback and Reconfiguration" -process "Generate Programming File"
   project set "FPGA Start-Up Clock" "CCLK" -process "Generate Programming File"
   project set "Done (Output Events)" "Default (4)" -process "Generate Programming File"
   project set "Drive Done Pin High" "false" -process "Generate Programming File"
   project set "Enable Outputs (Output Events)" "Default (5)" -process "Generate Programming File"
   project set "Wait for DCI Match (Output Events)" "Auto" -process "Generate Programming File"
   project set "Wait for DLL Lock (Output Events)" "Default (NoWait)" -process "Generate Programming File"
   project set "Release Write Enable (Output Events)" "Default (6)" -process "Generate Programming File"
   project set "Enable Internal Done Pipe" "true" -process "Generate Programming File"
   project set "Allow Logic Optimization Across Hierarchy" "false" -process "Map"
   project set "Optimization Strategy (Cover Mode)" "Area" -process "Map"
   project set "Pack I/O Registers/Latches into IOBs" "For Inputs and Outputs" -process "Map"
   project set "Generate Detailed MAP Report" "false" -process "Map"
   project set "Map Slice Logic into Unused Block RAMs" "false" -process "Map"
   project set "Perform Timing-Driven Packing and Placement" "true" -process "Map"
   project set "Trim Unconnected Signals" "true" -process "Map"
   project set "Create I/O Pads from Ports" "false" -process "Translate"
   project set "Macro Search Path" "" -process "Translate"
   project set "Netlist Translation Type" "Timestamp" -process "Translate"
   project set "User Rules File for Netlister Launcher" "" -process "Translate"
   project set "Allow Unexpanded Blocks" "false" -process "Translate"
   project set "Allow Unmatched LOC Constraints" "false" -process "Translate"
   project set "Allow Unmatched Timing Group Constraints" "false" -process "Translate"
   project set "Placer Effort Level (Overrides Overall Level)" "High" -process "Place & Route"
   project set "Router Effort Level (Overrides Overall Level)" "High" -process "Place & Route"
   project set "Place And Route Mode" "Normal Place and Route" -process "Place & Route"
   project set "Perform Advanced Analysis" "false" -process "Generate Post-Place & Route Static Timing"
   project set "Report Paths by Endpoint" "3" -process "Generate Post-Place & Route Static Timing"
   project set "Report Type" "Verbose Report" -process "Generate Post-Place & Route Static Timing"
   project set "Number of Paths in Error/Verbose Report" "3" -process "Generate Post-Place & Route Static Timing"
   project set "Stamp Timing Model Filename" "" -process "Generate Post-Place & Route Static Timing"
   project set "Report Unconstrained Paths" "" -process "Generate Post-Place & Route Static Timing"
   project set "Perform Advanced Analysis" "false" -process "Generate Post-Map Static Timing"
   project set "Report Paths by Endpoint" "3" -process "Generate Post-Map Static Timing"
   project set "Report Type" "Verbose Report" -process "Generate Post-Map Static Timing"
   project set "Number of Paths in Error/Verbose Report" "3" -process "Generate Post-Map Static Timing"
   project set "Report Unconstrained Paths" "" -process "Generate Post-Map Static Timing"
   project set "Add I/O Buffers" "true" -process "Synthesize - XST"
   project set "Global Optimization Goal" "AllClockNets" -process "Synthesize - XST"
   project set "Keep Hierarchy" "No" -process "Synthesize - XST"
   project set "Register Balancing" "Yes" -process "Synthesize - XST"
   project set "Register Duplication" "true" -process "Synthesize - XST"
   project set "Asynchronous To Synchronous" "false" -process "Synthesize - XST"
   project set "Automatic BRAM Packing" "false" -process "Synthesize - XST"
   project set "BRAM Utilization Ratio" "100" -process "Synthesize - XST"
   project set "Bus Delimiter" "<>" -process "Synthesize - XST"
   project set "Case" "Maintain" -process "Synthesize - XST"
   project set "Cores Search Directories" "" -process "Synthesize - XST"
   project set "Cross Clock Analysis" "false" -process "Synthesize - XST"
   project set "Equivalent Register Removal" "true" -process "Synthesize - XST"
   project set "FSM Style" "LUT" -process "Synthesize - XST"
   project set "Generate RTL Schematic" "Yes" -process "Synthesize - XST"
   project set "Generics, Parameters" "" -process "Synthesize - XST"
   project set "Hierarchy Separator" "/" -process "Synthesize - XST"
   project set "HDL INI File" "" -process "Synthesize - XST"
   project set "Library Search Order" "" -process "Synthesize - XST"
   project set "Netlist Hierarchy" "As Optimized" -process "Synthesize - XST"
   project set "Optimize Instantiated Primitives" "false" -process "Synthesize - XST"
   project set "Pack I/O Registers into IOBs" "Yes" -process "Synthesize - XST"
   project set "Read Cores" "true" -process "Synthesize - XST"
   project set "Slice Packing" "true" -process "Synthesize - XST"
   project set "Slice Utilization Ratio" "100" -process "Synthesize - XST"
   project set "Use Clock Enable" "Yes" -process "Synthesize - XST"
   project set "Use Synchronous Reset" "Yes" -process "Synthesize - XST"
   project set "Use Synchronous Set" "Yes" -process "Synthesize - XST"
   project set "Use Synthesis Constraints File" "true" -process "Synthesize - XST"
   project set "Verilog Include Directories" "/home/carlo/basil/firmware/modules|/home/carlo/basil/firmware/modules/utils" -process "Synthesize - XST"
   project set "Verilog 2001" "true" -process "Synthesize - XST"
   project set "Verilog Macros" "DV1=6 | FX1_d=3 | FX1_m=10 | FX2_d=$FX2_div | FX2_m=$FX2_mult | prd=6.25" -process "Synthesize - XST"
   project set "Work Directory" "/home/carlo/fe65_p2/firmware/ise/xst" -process "Synthesize - XST"
   project set "Write Timing Constraints" "false" -process "Synthesize - XST"
   project set "Other XST Command Line Options" "" -process "Synthesize - XST"
   project set "Auto Implementation Compile Order" "true"
   project set "Map Effort Level" "High" -process "Map"
   project set "Combinatorial Logic Optimization" "false" -process "Map"
   project set "Starting Placer Cost Table (1-100)" "1" -process "Map"
   project set "Power Reduction" "false" -process "Map"
   project set "Register Duplication" "Off" -process "Map"
   project set "Generate Constraints Interaction Report" "false" -process "Generate Post-Map Static Timing"
   project set "Synthesis Constraints File" "" -process "Synthesize - XST"
   project set "Mux Style" "Auto" -process "Synthesize - XST"
   project set "RAM Style" "Auto" -process "Synthesize - XST"
   project set "Maximum Number of Lines in Report" "1000" -process "Generate Text Power Report"
   project set "Output File Name" "pixel" -process "Generate IBIS Model"
   project set "Timing Mode" "Non Timing Driven" -process "Map"
   project set "Generate Asynchronous Delay Report" "false" -process "Place & Route"
   project set "Generate Clock Region Report" "false" -process "Place & Route"
   project set "Generate Post-Place & Route Power Report" "false" -process "Place & Route"
   project set "Generate Post-Place & Route Simulation Model" "false" -process "Place & Route"
   project set "Power Reduction" "false" -process "Place & Route"
   project set "Timing Mode" "Performance Evaluation" -process "Place & Route"
   project set "Create Binary Configuration File" "false" -process "Generate Programming File"
   project set "Enable Debugging of Serial Mode BitStream" "false" -process "Generate Programming File"
   project set "CLB Pack Factor Percentage" "100" -process "Map"
   project set "Place & Route Effort Level (Overall)" "High" -process "Place & Route"
   project set "Generate Constraints Interaction Report" "false" -process "Generate Post-Place & Route Static Timing"
   project set "Move First Flip-Flop Stage" "true" -process "Synthesize - XST"
   project set "Move Last Flip-Flop Stage" "true" -process "Synthesize - XST"
   project set "ROM Style" "Auto" -process "Synthesize - XST"
   project set "Safe Implementation" "No" -process "Synthesize - XST"
   project set "Extra Effort" "Normal" -process "Map"
   project set "Power Activity File" "" -process "Map"
   project set "Power Activity File" "" -process "Place & Route"
   project set "Extra Effort (Highest PAR level only)" "Normal" -process "Place & Route"
   project set "Starting Placer Cost Table (1-100)" "1" -process "Place & Route"
   project set "Functional Model Target Language" "Verilog" -process "View HDL Source"
   project set "Change Device Speed To" "-5" -process "Generate Post-Place & Route Static Timing"
   project set "Change Device Speed To" "-5" -process "Generate Post-Map Static Timing"
		puts "Synthetizing"
		process run "Synthesize - XST"
		puts "Implementing"
		process run "Implement Design"
		puts "Generating bitfile"
		process run "Generate Programming File"
		puts "Done"
	
		set oldname "fe65p2_mio.bit"
		set newname "fe65p2_mio_CMD$index($i).bit"
		file rename -force $oldname $newname
   }

#   project close
   return true
}

if {[catch {main} result]} {
      puts "$myScript failed: $result."
   }
}
