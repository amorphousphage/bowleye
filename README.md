# BowlEye - Bowling Ball Tracker
BowlEye is a camera based bowling ball tracking software which allows you to visualize the path of the bowling ball going down the lane and record a close-up video of the pin action. It is also able to track your score and show you statistical evaluation of your accuracy.

## Current and planned Features
This software automatically records a video when it detects a bowling ball going down the lane and analyses the video to show the following things:
- The path the bowling ball took
- The position (of the ball) at the foul line
- The position (of the ball) at the arrows
- The position (of the ball) at the breakpoint
- The ball speed
- The pins that fell (webapp is then calculating the score from it)
- The entry point into the pins (coming soon)

The software will run as a webapp for the players to connect to via their smart phones. There will be player profiles, stats and game overview.

This software operates in two modes:
- Training mode: Every shot is tracked and displayed, but no values are saved
- Game Mode: Every Shot is tracked and assigned to a selected player. All data points of all shots in the current game of a player a displayed and the mean and standard deviation values are calculated (coming soon)

Additionally, the software uses an additional camera to show a close-up of the pin action, which is then used to determine the pins left standing (reading the score)
Pin action videos can also be saved if wanted. (coming soon)

## Hardware requirements
To be able to run the software you will need at least:
- 1 Orange Pi 5 (8GB/16GB RAM) (not sure yet if powerful enough in the end)
- 2 USB Cameras per Lane (Full HD, 30/60 fps is enough)
- some USB cables (incl. repeaters for long distance) to connect all cameras to the Pi (might become wireless before final release)
- A Laptop running the BowlEye software (will change before final release)

The software is in very early stages. So far it has been tested on one single lane, but in theory supportes multiple lanes at the same time

  
