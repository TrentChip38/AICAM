from gpiozero import Motor
from time import sleep

# Define motor pins (using BCM GPIO numbers)
# Motor 1: Forward=GPIO4, Backward=GPIO14 (Physical Pins 7 & 8)
# Motor 2: Forward=GPIO17, Backward=GPIO27 (Physical Pins 11 & 13)
#motor1 = Motor(forward=4, backward=14)
#motor2 = Motor(forward=17, backward=27)
motor1 = Motor(forward=5, backward=6)
motor2 = Motor(forward=13, backward=19)

print("Motors moving forward")
motor1.forward(speed=0.5) # 50% speed
motor2.forward(speed=0.5)
sleep(2)

print("Motors reversing")
motor1.backward(speed=0.5)
motor2.backward(speed=0.5)
sleep(2)

print("Stopping")
motor1.stop()
motor2.stop()
