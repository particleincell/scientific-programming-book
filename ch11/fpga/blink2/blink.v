module blink (input wire clock,
				  output wire[1:0] LED);

reg[27:0] counter;
initial 
 counter[27:0] <= 'b1;   // set all bits to 1 (turns off LED)	
		
always @(posedge clock) begin
  counter <= counter + 1;   // increment on tick
end	

//LEDs are pins L14, K15, J14, and J13
assign LED[0] = counter[23];		
assign LED[1] = counter[26];	  
endmodule