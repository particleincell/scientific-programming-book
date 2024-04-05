#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import animation
import serial

# try to find a serial port, this may need to be changed on Windows
# make sure to close Serial Monitor/Plotter in Arduino
port_ok = False
for i in range(0,5):
    try:
        port_name = "/dev/ttyACM%d"%i
        print(port_name)
        ser = serial.Serial(port_name,9600)
        print("Opened port "+port_name)
        port_ok = True
        break
    except serial.SerialException:
        pass

if (not(port_ok)):
    raise Exception("Failed to open port")
     
plt.close('all')
fig = plt.figure(1)  # use figure 
temp_plot, = plt.plot([],[],color='black',label='Temp (C)')
rh_plot, = plt.plot([],[],color='gray',linestyle='-.',linewidth=3,label='RH%')

plt.xlim([0,200])
plt.ylim([0,100])
plt.legend()
plt.grid()

T = []
RH = []
samples = []


def read(n):
    global rowj
    global phi,ndi
    line = str(ser.readline()); # read line from serial port
    line = line[2:-5]  #eliminate trailing b' and \r\n
    
    pieces = line.split(','); # split by comma
    RH.append(float(pieces[0]))
    T.append(float(pieces[1]))
    samples.append(n)
    temp_plot.set_data(samples,T)
    rh_plot.set_data(samples,RH)
#    plt.pause(0.01)
    
anim=animation.FuncAnimation(fig, read,frames=200, interval=100, repeat=False)
plt.show()  
    