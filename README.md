# Klipper Web Control

This is the *klippy* module for serving and controlling [Klipper](https://github.com/KevinOConnor/klipper) through [Klipper Web Control](https://github.com/knoopx/KlipperWebControl).

Although this module was initially based on the work by [Stephan3](https://github.com/Stephan3/dwc2-for-klipper), this is a complete, clean re-implementation that communicates over WebSockets.

This module also super-seeds the `virtual_sdcard` and `pause_resume` modules. **Please disable them before using it**

## Installation

```
git clone https://github.com/knoopx/klippy-web-control ~/klippy-web-control
ln -s ~/klippy-web-control/kwc.py ~/klipper/klippy/extras
mkdir -p ~/sdcard/www ~/sdcard/sys ~/sdcard/gcodes
cd ~/sdcard/www && wget https://github.com/knoopx/KlipperWebControl/releases/latest/download/KlipperWebControl.zip && unzip KlipperWebControl.zip && rm KlipperWebControl.zip
~/klippy-env/bin/pip install tornado
```

## Usage

Add a new section to your `printer.cfg`:

```
[kwc]
path: ~/sdcard
port: 4444
abort_gcode:
  G91; relative
  {% if printer.extruder.temperature >= 170 %}
  G1 E-1 F300; retract the filament a bit before lifting the nozzle
  {% endif %}
  G0 Z15; move z axis up 15
  G90; absolute
  G0 X0 Y210 F5000; move part out for inspection
  M104 S0 ; turn off extruder heat
  M140 S0 ; turn off heated bed
  M106 S0 ; Turn off fan
  M18; Disable steppers
```

## Additional Notes

You can move `printer.cfg` to `~/sdcard/sys` and update the `init.d` script to be able to edit it from the web interface.
You can also use `[include another_config.cfg]` to split config into separate files.

The macros created using the web interface are not equivalent to Klipper gcode macros. You need to use `RUN_MACRO` command below to execute them. However these support Jinja2 templates and printer variable resolution (see [Command Templates](https://github.com/KevinOConnor/klipper/blob/master/docs/Command_Templates.md)).

## G-Codes

`RUN_MACRO FILE="0:/macros/macro.g"`: Runs the macro at the specified path

`PRINT_FILE FILE="0:/gcodes/file.g"`: Starts printing the file at the specified path

`SELECT_FILE FILE="0:/gcodes/file.g"`: Selects the file for printing, does not start it

`PAUSE`: Pauses the print job allowing you to change filament, clean nozzle or whatever.

`RESUME`: Resumes the print job.

`ABORT`: Aborts printing the current job and runs the specified `abort_gcode` if set.
