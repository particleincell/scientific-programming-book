
// Source adapted from https://github.com/wd5gnr/VidorFPGA

#include <wiring_private.h>
#include "jtag.h"
#include "defines.h"

int blink = LOW;
int bit = LOW;

#define SET_PIN 5
#define LED_PIN 6
#define BIT_PIN 4

void setup()
{
  setup_fpga();

  Serial.begin(9600);

  pinMode(SET_PIN, OUTPUT);
  pinMode(BIT_PIN, OUTPUT);
  pinMode(LED_PIN, INPUT);

  digitalWrite(SET_PIN, blink);
  digitalWrite(BIT_PIN, bit);

  Serial.println("Select No Line Ending, 'go', 'stop', or 'bit' to control LED");
}

void loop()
{
  static int oldstate = -1;
  static int num = 0;
  static int oldbit=0;
  int state;
  
  // check if we have any input from the serial port
  if (Serial.available() > 0)  {
    String msg = Serial.readString();
    if (msg.equals("go")) blink=HIGH;  else if (msg.equals("stop")) blink=LOW;
    digitalWrite(SET_PIN, blink);
    if (msg.equals("bit")) {
      digitalWrite(BIT_PIN,LOW);
      delay(1);
      digitalWrite(BIT_PIN,HIGH);      
    } 
  }

  // read the LED pin value that is set by the FPGA
  state = digitalRead(LED_PIN);
  if (state != oldstate)  {
    Serial.print(state);
    if (++num == 40)  {
      Serial.println();
      num = 0;
    }
    
    oldstate = state;
  }
}
