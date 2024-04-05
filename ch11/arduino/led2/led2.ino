#define PIN_LED 9
#define PIN_BUTTON 12
#define PIN_PHOTORES 1

void setup() {
  pinMode(PIN_LED,OUTPUT);
  pinMode(PIN_BUTTON,INPUT);
  
  Serial.begin(9600);
  if (!Serial) {}  // wait until serial port is ready
}
int counter = 0;
void loop() {
  
  int button = digitalRead(PIN_BUTTON);  // get button state
  int val = analogRead(PIN_PHOTORES);    // get analog value from photoresistor
  counter++;
  Serial.println(val);      // write text to the serial port

  // turn on the LED if the button is pressed or the light is dim
  if (button || val<50) digitalWrite(PIN_LED,HIGH);
  else digitalWrite(PIN_LED,LOW);

  delay(500);   // wait 0.5 seconds
}
