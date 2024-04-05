#include "Blink.h"

#define PIN_LED 9
Blink blink(PIN_LED,5);  // instantiate as a global variable

void setup() {
  while (!blink) { /* hang */ }
}

void loop() {
  blink.run();
}
