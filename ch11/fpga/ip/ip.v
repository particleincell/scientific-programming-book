module ip( input clock,
           input reset,
           input wire[31:0] in, 
           output wire[31:0] out);
	
reg [31:0] calc_out;
mySqrt mySqrt_inst(clock,reset,in,calc_out);


reg [7:0] buffer;  // 8-bit register
reg [8*20:0] str;   // 21-character string
integer my_int;

always @(posedge clock) begin
buffer = 8'hFF; // some 8-bit binary value
buffer[3:0] = 4'b1100; // overwrite the bits 3 through 0
str <= "Hello World!"; // set string
my_int <= 123;       // some integer value
end

assign out = (in[31]==0) ? calc_out : 32'b0;	
			  		  
endmodule