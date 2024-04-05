`timescale 1ns / 10ps   // set time unit size and smallest time tolerance
module blink (input wire clock, input wire disp, output wire[1:0] LED);
  reg[27:0] counter;
  initial 
    counter = {28{1'b0}};   // clear all bits

  always @(posedge clock) begin
    counter <= counter + 1;   // increment on tick
  end	

  always @(disp) begin      // display counter on disp signal change
   $display("time:%6d, counter:%b",$time,counter);
  end

  assign LED[0] = counter[18];	
  assign LED[1] = counter[22];	
endmodule

// test bench
module blink_tb();
  reg clock;
  reg disp;
  wire [1:0] LEDs;
  parameter MAX_TICKS = 1000000;

  //instantiate the module
  blink blink_inst(.clock(clock), .disp(disp),.LED(LEDs));

  initial begin
    clock = 1'b0;
    disp = 1'b0;
    forever #1 clock = ~clock;  //tick the clock every 1 time unit
  end
  
  initial begin
    #200 disp = ~disp;     // flip disp after 200 time units
    #800 disp = ~disp;
    #1000 disp = ~disp;
  end
  
  initial begin
    $monitor("time: %3d, LED0:%b, LED1:%b",$time,LEDs[0],LEDs[1]);
  end

  initial begin
    $dumpfile("blink_tb.vcd");   // open output file
    $dumpvars(0, blink_tb);      // save all data from blink_tb module
    #(MAX_TICKS)                 // wait MAX_TICKS
    $display("Done!");           // show message and terminate simulation
    $finish;
  end
endmodule
