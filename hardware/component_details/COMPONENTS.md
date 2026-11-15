# Component Specifications & Bill of Materials (BOM)

## 1. Primary Hardware Components

### 1.1 ESP32 Development Board (Main MCU)
* **Processor:** Dual-core 32-bit Xtensa LX6 microprocessor operating up to $240\,\text{MHz}$.
* **Internal Memory:** 520 KB SRAM, 4 MB Flash.
* **ADC Resolution:** 12-bit SAR ADC (up to 18 channels), calibrated with 11 dB attenuation for $0\text{--}3.3\,\text{V}$ range.
* **Connectivity:** Integrated Wi-Fi ($802.11\,\text{b/g/n}$) and Bluetooth $4.2/\text{BLE}$.
* **Role:** High-speed real-time data acquisition, digital signal filtering, trapezoidal numerical integration, and WebSocket telemetry.

### 1.2 MPX10DP Differential Pressure Sensor
* **Pressure Range:** $0\text{--}10\,\text{kPa}$ ($0\text{--}1.45\,\text{psi}$).
* **Sensitivity:** $3.5\,\text{mV/kPa}$ nominal.
* **Full-Scale Differential Output:** $\approx 35\,\text{mV}$ at $10\,\text{kPa}$.
* **Operating Temperature:** $-40^\circ\text{C}$ to $+125^\circ\text{C}$.
* **Role:** Senses differential pressure drop across the internal flow restriction of the spirometer mouthpiece during patient exhalation.

### 1.3 AD620 Precision Instrumentation Amplifier
* **Gain Range:** $1\text{ to }10,000$ programmed by a single external resistor $R_G$.
* **Gain Formula:** $G = 1 + \frac{49.4\,\text{k}\Omega}{R_G}$.
* **Configured Gain:** $R_G = 620\,\Omega \implies G \approx 80.68$.
* **Output Voltage Span:** $35\,\text{mV} \times 80.68 \approx 2.82\,\text{V}$ (perfectly matches ESP32 3.3V ADC range).
* **CMRR:** High Common-Mode Rejection Ratio ($> 100\,\text{dB}$ at $G=10$).
* **Role:** Amplifies weak microvolt differential signals with minimal noise and temperature drift.

### 1.4 DHT22 (AM2302) Environmental Sensor
* **Temperature Range:** $-40^\circ\text{C}$ to $+80^\circ\text{C}$ ($\pm 0.5^\circ\text{C}$ accuracy).
* **Relative Humidity Range:** $0\text{ to }100\%\,\text{RH}$ ($\pm 2\%\,\text{RH}$ accuracy).
* **Sampling Rate:** $0.5\,\text{Hz}$ (every 2 seconds).
* **Interface:** Single-bus digital interface on **GPIO 4**.
* **Role:** Provides dynamic ambient temperature and humidity measurements for real-time Body Temperature and Pressure Saturated (BTPS) physiological volume compensation.

### 1.5 Electret Microphone + MAX4466 Preamplifier
* **Frequency Response:** $20\,\text{Hz}$ to $20\,\text{kHz}$.
* **Supply Voltage:** $2.4\,\text{V}$ to $5.5\,\text{V}$.
* **Output:** Low-noise analog output biased at $V_{CC} / 2$.
* **Role:** Transduces patient respiratory breath sounds for acoustic deep learning (detection of wheezes and crackles).

---

## 2. Estimated Bill of Materials (BOM Cost)

| Component Description | Part Model | Unit Qty | Est. Unit Cost (USD) |
| :--- | :--- | :---: | :---: |
| ESP32 Development Board | ESP32-WROOM-32D | 1 | $\$4.50$ |
| Differential Pressure Sensor | NXP / Freescale MPX10DP | 1 | $\$11.20$ |
| Instrumentation Amplifier IC | Analog Devices AD620AN | 1 | $\$2.80$ |
| Environmental Sensor | DHT22 / AM2302 Module | 1 | $\$3.50$ |
| Audio Microphone Module | MAX4466 Electret Preamp | 1 | $\$1.80$ |
| Passive Components (Resistors, Caps) | $620\,\Omega$, $1.5\,\text{k}\Omega$, $1.0\,\mu\text{F}$ | 1 kit | $\$0.70$ |
| 3D-Printed Spirometer Mouthpiece | Custom PLA/PETG Orifice | 1 | $\$1.50$ |
| Breadboard / PCB & Jumper Wires | Prototyping Hardware | 1 | $\$2.50$ |
| **Total Estimated Hardware Prototype Cost** | — | — | **$\approx \$28.50\,\text{USD}$** |

*(Compare against commercial hospital diagnostic spirometers: $\$2,000\text{--}\$5,000\,\text{USD}$)*
