#include "Arduino.h"
#include "Blink.h"

void Blink::run() {
   digitalWrite(pin,HIGH);
   delay(duration*1000);
   digitalWrite(pin,LOW);
}

