#!/usr/bin/env python

import signal
import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Signal handler for stopping pipeline before destruction.
def signal_handler(sig, frame):
    p.set_state(Gst.State.NULL)
    sys.exit(0)

# Initialize gstreamer
GObject.threads_init()
Gst.init(None)

# Defining the pipeline 
gst_str = "nvcompositor name=mix background-w=1280 background-h=180 \
    sink_0::xpos=0    sink_0::ypos=0   sink_0::width=320 sink_0::height=180 \
    sink_1::xpos=320  sink_1::ypos=0   sink_1::width=320 sink_1::height=180 \
    sink_2::xpos=640  sink_2::ypos=0   sink_2::width=320 sink_2::height=180 \
    sink_3::xpos=960 sink_3::ypos=0   sink_3::width=320 sink_3::height=180 \
    ! video/x-raw(memory:NVMM),format=RGBA, width=1280,height=180 ! nvvidconv flip_method=2 ! video/x-raw(memory:NVMM),format=I420 ! nvv4l2h264enc insert-vui=1 insert-sps-pps=1 ! h264parse ! rtph264pay ! udpsink host=<IP-address-of-receiving-computer> port=5000  \
    nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),format=NV12,width=320,height=180,framerate=(fraction)30/1 ! queue ! mix.sink_0    \
    nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM),format=NV12,width=320,height=180,framerate=(fraction)30/1 ! queue ! mix.sink_1    \
    nvarguscamerasrc sensor-id=2 ! video/x-raw(memory:NVMM),format=NV12,width=320,height=180,framerate=(fraction)30/1 ! queue ! mix.sink_2    \
    nvarguscamerasrc sensor-id=3 ! video/x-raw(memory:NVMM),format=NV12,width=320,height=180,framerate=(fraction)30/1 ! queue ! mix.sink_3 "

# Creating the pipeline
p = Gst.parse_launch (gst_str)

# Register signal handler for proper termination if receiving SIGINT, for instance Ctrl-C.
signal.signal(signal.SIGINT, signal_handler)

# Start the pipeline
p.set_state(Gst.State.READY)
p.set_state(Gst.State.PAUSED)
p.set_state(Gst.State.PLAYING)

# Run for 1000s 
time.sleep(1000)

# Done. Stop the pipeline before clean up on exit.
p.set_state(Gst.State.NULL)
exit(0)
