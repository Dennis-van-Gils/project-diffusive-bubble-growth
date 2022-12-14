/**
 * @file    main.cpp
 * @author  Dennis van Gils (vangils.dennis@gmail.com)
 * @version https://github.com/Dennis-van-Gils/project-diffusive-bubble-growth
 * @date    11-10-2022
 *
 * @brief   A pressure logger for the Diffusive Bubble Growth setup.
 *
 * @copyright MIT License. See the LICENSE file for details.
 */

/*------------------------------------------------------------------------------
  Description
--------------------------------------------------------------------------------

  Hardware:
  - Adafruit #3857: Adafruit Feather M4 Express - Featuring ATSAMD51 Cortex M4
  - MIKROE 4-20 mA R Click (MIKROE-1387): 4-20 mA current loop receiver
  - RS PRO #797-5018: Pressure Sensor, 0-10 bar, current output

  Reports the readings of the pressure sensor over the serial port as follows:
  >> [averaged bit value] \t [averaged mA] \t [averaged bar] \n

  The NeoPixel RGB LED of the Feather M4 will indicate its status:
  - Blue : We're setting up
  - Green: Running okay
  - Every read out, the LED will flash brightly in green
------------------------------------------------------------------------------*/

#include "Adafruit_NeoPixel.h"
#include "DvG_StreamCommand.h"
#include "MIKROE_4_20mA_RT_Click.h"

#include <Arduino.h>

// Instantiate serial port listener for receiving ASCII commands
const uint8_t CMD_BUF_LEN = 16;  // Length of the ASCII command buffer
char cmd_buf[CMD_BUF_LEN]{'\0'}; // The ASCII command buffer
DvG_StreamCommand sc(Serial, cmd_buf, CMD_BUF_LEN);

// Onboard NeoPixel
Adafruit_NeoPixel neo(1, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);
const uint8_t NEO_DIM = 2;    // Brightness level for dim intensity [0 -255]
const uint8_t NEO_BRIGHT = 6; // Brightness level for bright intensity [0 - 255]
const uint32_t FLASH_LENGTH = 100; // [us]

/*------------------------------------------------------------------------------
  MIKROE 4-20 mA R click board for reading out the pressure sensor
------------------------------------------------------------------------------*/

// Cable select pin
const uint8_t PIN_R_CLICK = 5;

// Calibrated against a multimeter @ 11-10-2022 by DPM van Gils
const RT_Click_Calibration R_CLICK_CALIB{3.99, 20.15, 796, 4020};

// Single R click readings fluctuate a lot and so we will employ an exponential
// moving average (EMA) by using oversampling and subsequent low-pass filtering
const uint32_t EMA_DT = 5000; // Desired oversampling interval [??s]
const float EMA_LP = 1.;      // Low-pass filter cut-off frequency [Hz]

R_Click R_click(PIN_R_CLICK, R_CLICK_CALIB, EMA_DT, EMA_LP);

/*------------------------------------------------------------------------------
  PR PRO pressure sensor, type 797-5018
------------------------------------------------------------------------------*/

// Structure to hold the RS PRO pressure sensor calibration parameters.
// The parameters are found on the calibration sheet supplied with the sensor.
struct Pressure_Calibration {
  float zero_mA;
  float span_mA;
  float full_range_bar;
};

// Calibration parameters supplied with the pressure sensor: serial 1037812
const Pressure_Calibration PRESSURE_CALIB{4.01, 15.99, 10.0};

inline float mA2bar(float mA, const Pressure_Calibration calib) {
  return (mA - calib.zero_mA) / calib.span_mA * calib.full_range_bar;
}

/*------------------------------------------------------------------------------
  Readings
------------------------------------------------------------------------------*/

struct Readings {
  float pres_bitval = NAN; // Pressure sensor [EMA bitval]
  float pres_mA = NAN;     // Pressure sensor [mA]
  float pres_bar = NAN;    // Pressure sensor [bar]
};
Readings readings; // Structure holding the sensor readings

/*------------------------------------------------------------------------------
  setup
------------------------------------------------------------------------------*/

void setup() {
  neo.begin();
  neo.setPixelColor(0, neo.Color(0, 0, NEO_BRIGHT)); // Blue: We're in setup()
  neo.show();

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  Serial.begin(9600);
  R_click.begin();

  neo.setPixelColor(0, neo.Color(0, NEO_DIM, 0)); // Green: All set up
  neo.show();
}

/*------------------------------------------------------------------------------
  loop
------------------------------------------------------------------------------*/

void loop() {
  uint32_t now = millis();
  static bool flash = false;
  static uint32_t flash_tick = now;

  // Update R-click readings
  R_click.poll_EMA();

  // Listen for incoming commands over serial
  if (sc.available()) {
    char *str_cmd = sc.getCommand();

    // Flash the NeoPixel bright green to indicate we received a command
    neo.setPixelColor(0, neo.Color(0, NEO_BRIGHT, 0));
    neo.show();
    flash = true;
    flash_tick = now;

    if (strcmp(str_cmd, "id?") == 0) {
      // Report identity string
      Serial.println("Arduino, Diffusive Bubble Growth logger");

    } else if (strcmp(str_cmd, "?") == 0) {
      // Report pressure readings
      readings.pres_bitval = R_click.get_EMA_bitval();
      readings.pres_mA = R_click.get_EMA_mA();
      readings.pres_bar = mA2bar(readings.pres_mA, PRESSURE_CALIB);

      Serial.print(readings.pres_bitval, 0);
      Serial.write('\t');
      Serial.print(readings.pres_mA, 2);
      Serial.write('\t');
      Serial.println(readings.pres_bar, 3);
    }
  }

  if (flash && (now - flash_tick >= FLASH_LENGTH)) {
    // Set the NeoPixel back to dim green to indicate we're in an idle state
    flash = false;
    neo.setPixelColor(0, neo.Color(0, NEO_DIM, 0));
    neo.show();
  }
}
