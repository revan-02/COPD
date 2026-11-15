/*
 * ======================================================================================
 * AI-POWERED COPD SEVERITY PREDICTION AND RESPIRATORY MONITORING SYSTEM
 * Embedded ESP32 Firmware for Digital Spirometry & Acoustic Auscultation
 * ======================================================================================
 *
 * Hardware Configuration:
 *   - Microcontroller: ESP32 Dual-Core Tensilica Xtensa LX6 (240 MHz)
 *   - Pressure Sensor: MPX10DP (0 - 10 kPa / 0 - 1.45 psi Differential Pressure)
 *   - Instrumentation Amplifier: AD620 (Gain Rg = 620 Ohm, G ≈ 80.68)
 *   - Ambient Environmental Sensor: DHT22 / AM2302 (Temperature & Humidity for BTPS)
 *   - Audio Acoustic Sensor: Electret Condenser + MAX4466 Preamplifier
 *
 * Pin Mapping:
 *   - GPIO 34 (ADC1_CH6): MPX10DP + AD620 Pressure Sensor Analog Input
 *   - GPIO 35 (ADC1_CH7): Microphone Analog Audio Waveform Input
 *   - GPIO 27 (Digital):  Microphone Digital Sound Activity Threshold Comparator
 *   - GPIO 4  (Digital):  DHT22 Single-Bus Data Line
 *   - VCC: 3.3V / 5.0V (LDO Regulated)
 *   - GND: Common System Ground
 * ======================================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <DHT.h>
#include <math.h>

// ---------------- PIN DEFINITIONS ----------------
#define PIN_PRESSURE_ANALOG   34  // AD620 amplified differential pressure (0 - 3.3V)
#define PIN_MIC_ANALOG        35  // Microphone analog acoustic signal
#define PIN_MIC_DIGITAL       27  // Microphone digital sound threshold
#define PIN_DHT22_DATA        4   // DHT22 environmental sensor

#define DHTTYPE               DHT22

// ---------------- CALIBRATION & SAMPLING CONSTANTS ----------------
const float ADC_VREF          = 3.30f;       // ESP32 ADC Reference Voltage
const int   ADC_RESOLUTION    = 4095;        // 12-bit SAR ADC (0 - 4095)
const float SAMPLING_FREQ_HZ  = 100.0f;      // 100 Hz Spirometry Sampling Rate
const float DT_SEC            = 1.0f / SAMPLING_FREQ_HZ; // 10 ms integration step

// AD620 & MPX10DP Calibration Parameters
// MPX10DP: 3.5 mV / kPa. AD620 Gain: G = 1 + (49.4k / 620) = 80.68
// Full scale ~35 mV differential * 80.68 = 2.82 V at 10 kPa
const float AD620_GAIN        = 80.68f;
const float MPX10DP_SENS_V_KPA = 0.0035f;    // 3.5 mV/kPa
const float FLOW_CALIBRATION_K = 1.425f;     // Empirical flow orifice constant (L/s per sqrt(kPa))

// Flow Detection Thresholds (ATS/ERS Standardization)
const float FLOW_START_THRESH = 0.10f;       // Exhalation start threshold: 0.10 L/s
const float FLOW_END_THRESH   = 0.025f;      // Exhalation end plateau threshold: 0.025 L/s
const float MAX_TEST_DURATION = 8.0f;        // Maximum maneuver duration: 8.0 seconds

// ---------------- INSTANCES & GLOBAL STATE ----------------
DHT dht(PIN_DHT22_DATA, DHTTYPE);

// Environmental BTPS variables
float ambient_temp_c   = 24.0f;
float ambient_humidity = 50.0f;
float btps_factor      = 1.05f;

// Baseline pressure offset (Zero Tare Calibration)
float pressure_baseline_voltage = 0.0f;

// Spirometric Maneuver Variables
bool  is_exhaling         = false;
float current_flow_lps    = 0.0f;
float cumulative_vol_l    = 0.0f;
float fev1_liters         = 0.0f;
float fvc_liters          = 0.0f;
float pef_lps             = 0.0f;
float maneuver_time_sec   = 0.0f;
int   end_plateau_samples = 0;

// ---------------- BTPS FACTOR CALCULATION ----------------
float compute_btps_factor(float temp_c, float rh_pct) {
    // Saturated water vapor pressure at ambient temperature: Antoine / Magnus-Tetens Equation
    float p_sat_ambient = (rh_pct / 100.0f) * 4.584f * exp((17.27f * temp_c) / (237.3f + temp_c));
    float p_barometric  = 760.0f;  // Standard atmospheric pressure in mmHg
    float p_vapor_body  = 47.07f;  // Water vapor pressure at 37°C saturated

    float temp_ratio     = (273.15f + 37.0f) / (273.15f + temp_c);
    float pressure_ratio = (p_barometric - p_sat_ambient) / (p_barometric - p_vapor_body);

    float factor = temp_ratio * pressure_ratio;
    // Constrain factor to physiologically plausible range (1.00 - 1.25)
    if (factor < 1.0f) factor = 1.0f;
    if (factor > 1.25f) factor = 1.25f;
    return factor;
}

// ---------------- SENSOR ZERO-OFFSET TARE ----------------
void calibrate_zero_offset() {
    Serial.println(F("[SYSTEM] Initiating zero-offset tare calibration (Keep mouthpiece still)..."));
    long sum_adc = 0;
    const int CAL_SAMPLES = 200;
    
    for (int i = 0; i < CAL_SAMPLES; i++) {
        sum_adc += analogRead(PIN_PRESSURE_ANALOG);
        delay(5);
    }
    
    float avg_adc = (float)sum_adc / (float)CAL_SAMPLES;
    pressure_baseline_voltage = (avg_adc / (float)ADC_RESOLUTION) * ADC_VREF;
    
    Serial.print(F("[SYSTEM] Baseline zero offset voltage: "));
    Serial.print(pressure_baseline_voltage, 4);
    Serial.println(F(" V. Calibration complete."));
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println(F("\n========================================================"));
    Serial.println(F("ESP32 SMART SPIROMETER & RESPIRATORY MONITORING SYSTEM"));
    Serial.println(F("========================================================"));

    // Configure ADC resolution and attenuation
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db); // Full-scale input voltage ~0 - 3.3V

    pinMode(PIN_PRESSURE_ANALOG, INPUT);
    pinMode(PIN_MIC_ANALOG, INPUT);
    pinMode(PIN_MIC_DIGITAL, INPUT);

    // Initialize DHT22 environmental sensor
    dht.begin();
    delay(500);

    // Read ambient conditions for BTPS correction
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) {
        ambient_temp_c   = t;
        ambient_humidity = h;
    }
    btps_factor = compute_btps_factor(ambient_temp_c, ambient_humidity);

    Serial.print(F("[ENV] Ambient Temp: "));
    Serial.print(ambient_temp_c, 1);
    Serial.print(F(" C, Humidity: "));
    Serial.print(ambient_humidity, 1);
    Serial.print(F(" %, Calculated BTPS Factor: "));
    Serial.println(btps_factor, 4);

    // Perform baseline tare
    calibrate_zero_offset();

    Serial.println(F("[READY] Waiting for patient exhalation maneuver..."));
}

void loop() {
    static unsigned long last_sample_time = 0;
    unsigned long current_time = millis();

    // Deterministic 100 Hz sampling loop (10 ms period)
    if (current_time - last_sample_time >= 10) {
        last_sample_time = current_time;

        // 1. Read Pressure Sensor
        int raw_adc = analogRead(PIN_PRESSURE_ANALOG);
        float v_sensor = ((float)raw_adc / (float)ADC_RESOLUTION) * ADC_VREF;
        float v_diff   = v_sensor - pressure_baseline_voltage;

        if (v_diff < 0.0f) v_diff = 0.0f; // Ignore negative flow artifacts

        // Calculate Differential Pressure in kPa
        // v_diff = ΔP_kPa * Sens * Gain
        float delta_p_kpa = v_diff / (MPX10DP_SENS_V_KPA * AD620_GAIN);

        // Bernoulli flow rate Q(t) = k * sqrt(ΔP) in L/s
        float raw_flow_lps = FLOW_CALIBRATION_K * sqrt(delta_p_kpa);

        // Apply BTPS correction
        current_flow_lps = raw_flow_lps * btps_factor;

        // 2. Read Acoustic Microphone
        int raw_mic_analog  = analogRead(PIN_MIC_ANALOG);
        int mic_activity    = digitalRead(PIN_MIC_DIGITAL);

        // 3. Exhalation Maneuver State Machine
        if (!is_exhaling) {
            // Trigger start of forced exhalation
            if (current_flow_lps >= FLOW_START_THRESH) {
                is_exhaling         = true;
                cumulative_vol_l    = 0.0f;
                fev1_liters         = 0.0f;
                pef_lps             = 0.0f;
                maneuver_time_sec   = 0.0f;
                end_plateau_samples = 0;
                Serial.println(F("[MANEUVER] Exhalation started!"));
            }
        } else {
            // Active forced exhalation
            maneuver_time_sec += DT_SEC;

            // Numerical trapezoidal integration: V = ∫ Q dt
            cumulative_vol_l += current_flow_lps * DT_SEC;

            // Peak Expiratory Flow (PEF) tracking
            if (current_flow_lps > pef_lps) {
                pef_lps = current_flow_lps;
            }

            // Capture FEV1 at exactly 1.0 second
            if (maneuver_time_sec >= 1.0f && fev1_liters == 0.0f) {
                fev1_liters = cumulative_vol_l;
            }

            // Check for end of exhalation (plateau < 0.025 L/s for 0.5s or max time reached)
            if (current_flow_lps < FLOW_END_THRESH) {
                end_plateau_samples++;
            } else {
                end_plateau_samples = 0;
            }

            if (end_plateau_samples >= 50 || maneuver_time_sec >= MAX_TEST_DURATION) {
                // Maneuver completed
                is_exhaling = false;
                fvc_liters  = cumulative_vol_l;
                if (fev1_liters == 0.0f) fev1_liters = fvc_liters; // Fallback if < 1s

                float ratio = (fvc_liters > 0.0f) ? (fev1_liters / fvc_liters) : 0.0f;

                // Transmit JSON Diagnostic Packet over Serial / Wi-Fi
                Serial.println(F("\n--- [MANEUVER COMPLETE] ---"));
                Serial.print(F("{\"type\":\"spirometry_result\","));
                Serial.print(F("\"fev1_ml\":"));    Serial.print(fev1_liters * 1000.0f, 0);
                Serial.print(F(",\"fvc_ml\":"));    Serial.print(fvc_liters * 1000.0f, 0);
                Serial.print(F(",\"pef_lps\":"));   Serial.print(pef_lps, 2);
                Serial.print(F(",\"ratio\":"));     Serial.print(ratio, 3);
                Serial.print(F(",\"temp_c\":"));    Serial.print(ambient_temp_c, 1);
                Serial.print(F(",\"humidity\":"));  Serial.print(ambient_humidity, 1);
                Serial.println(F("}"));
                Serial.println(F("---------------------------\n"));
                Serial.println(F("[READY] Waiting for next exhalation maneuver..."));
            }
        }

        // Live Real-Time Telemetry Stream (30 Hz downsampled)
        static int telemetry_divider = 0;
        if (++telemetry_divider >= 3) {
            telemetry_divider = 0;
            if (is_exhaling) {
                Serial.print(F("T:"));     Serial.print(maneuver_time_sec, 2);
                Serial.print(F("s | Q:")); Serial.print(current_flow_lps, 2);
                Serial.print(F(" L/s | V:")); Serial.print(cumulative_vol_l, 3);
                Serial.print(F(" L | Mic:")); Serial.println(raw_mic_analog);
            }
        }
    }
}
