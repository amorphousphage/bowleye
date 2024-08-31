# BowlEye - Bowling Ball Tracker
BowlEye is a camera based bowling ball tracking software which allows you to visualize the path of the bowling ball going down the lane and record a close-up video of the pin action

## Current Features
This software automatically records a video when it detects a bowling ball going down the lane and analyses the video to show the following things:
- The path the bowling ball took
- The position (of the ball) at the foul line
- The position (of the ball) at the arrows
- The position (of the ball) at the breakpoint
- The ball speed

This software operates in two modes:
- Single Shot mode: Every shot is tracked and displayed, but no values are saved
- Record Multiple Shots Mode: Every Shot is tracked and assigned to a selected player. All data points of all shots in the current game of a player a displayed inn a table and the mean and standard deviation values are calculated

Videos can also be saved if wanted.
Additionally, the software uses an additional camera to show a close-up of the pin action.

## Features and improvements to come in the future
- Revisiting recording of pin camera as it is too fast
- Revisit ball speed calculation as it is inaccurate currently
- Revisit the display of the debugging image as it is currently not working
- Optimize data storage of tracking data
- Program the overlay of different shots in Record Multiple Shots Mode
- Rename certain variables to make the code more user friendly
- Program Entry Point detection for the pins camera
- Program Reading the Score (pins left standing) from the pins camera
- Re-structure the ball tracker to not analyse the saved video but each frame live as it is recorded
- Program an Oil Pattern overlay
- Program a web-service so data can be made accessible to the user via a web portal with a login
- Translate the program to different languages
- Compile the program to an executable file
- Write a complete installation guide and documentation

## Hardware requirements
To be able to run the software you will need:
- 1 Raspberry Pi 5 (8GB)
- 2 USB Cameras per Lane (Full HD, 30/60 fps is enough)
- some USB cables (incl. repeaters for long distance) to connect all cameras to the Raspberry Pi
- A Screen to connect the Raspberry Pi to or alternatively a Tablet/Notebook to connect to the Raspberry Pi with VNC (Raspberry Pi Remote Connect is not correctly working with the software)

The software is in very early stages. So far it has been tested on one single lane, but in theory supportes multiple lanes at the same time

  
