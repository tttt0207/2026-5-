import serial
import time

ser = serial.Serial('/dev/ttyS2', 115200, timeout=1)
#固定发送dx = 0.1cm dy = 0.2cm

while True:
    # dx = 0.1cm -> 10 -> 00 0A
    # dy = 0.2cm -> 20 -> 00 14
    data = bytes([0x01, 0x00, 0x0A, 0x00, 0x14, 0x05])

    ser.write(data)
    print("send:", data.hex(" "))

    time.sleep(0.5)
