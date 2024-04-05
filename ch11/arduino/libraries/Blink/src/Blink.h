/*
  Blink.h - Blink example
*/
#ifndef Debye_h
#define Debye_h

class Blink {
  public:
  // pin specifies pin connected to the LED, duration is in seconds
  Blink(int pin, int duration):pin{pin},duration{duration} { all_ok = true; }
  ~Blink() {}
 
  operator bool() {return all_ok;} 
  bool operator !() {return !all_ok;}
  
  void run();
  
  protected:
    int pin;
    int duration;  
    bool all_ok = false;
};
#endif
