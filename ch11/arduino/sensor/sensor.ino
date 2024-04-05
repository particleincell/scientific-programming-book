#include "Adafruit_Si7021.h"

Adafruit_Si7021 sensor = Adafruit_Si7021();

void setup() {
  Serial.begin(115200);

  // wait for serial port to open
  while (!Serial) {
    delay(10);
  }

  if (!sensor.begin()) {
    Serial.println("Did not find Si7021 sensor!");
    while (true) {}  //hang
  }  
  Serial.println("humidity,temperature");
  
  pinMode(9,OUTPUT);  
}

void loop() {
  float humidity = sensor.readHumidity();
  float temp = sensor.readTemperature();
  Serial.print(humidity,2);
  Serial.print(",");
  Serial.println(temp,2);

  // turn on the LED if humidity is over a threshold
  if (humidity>40) digitalWrite(9,HIGH);
  else  digitalWrite(9,LOW);
  
  delay(1000);
}
