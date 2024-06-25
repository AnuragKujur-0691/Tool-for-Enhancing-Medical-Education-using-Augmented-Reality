#pipe server
import self

from body import BodyThread, GestureThread,OrganDetectionThread

import time
import struct
import global_vars
from sys import exit

bodyThread = BodyThread()
bodyThread.start()

gestureThread = GestureThread()
gestureThread.start()

#organThread= OrganDetectionThread()
#organThread.start()

i = input()
print("Exiting…")
global_vars.KILL_THREADS = True
time.sleep(0.5)
exit()