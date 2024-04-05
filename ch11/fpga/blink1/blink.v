module blink (input wire button1, 
              input wire button2, 
              output wire LED);
// connect the buttons to the LED via an OR operator
assign LED = button1 | button2;
endmodule