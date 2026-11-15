# Hardware Circuit Schematics & Electrical Interfacing

## 1. System Pin Mapping & Wiring Guide

| Subsystem | Sensor / Module | Signal Type | ESP32 GPIO Pin | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Differential Pressure** | MPX10DP + AD620 | Analog ($0\text{--}3.3\,\text{V}$) | **GPIO 34** (`ADC1_CH6`) | Amplified differential exhalation pressure |
| **Respiratory Acoustics** | Electret + MAX4466 | Analog ($0\text{--}3.3\,\text{V}$) | **GPIO 35** (`ADC1_CH7`) | Continuous lung/breath audio signal |
| **Audio Threshold** | Sound Comparator Module | Digital Input | **GPIO 27** | High-level sound activity logic flag |
| **Environmental (BTPS)** | DHT22 / AM2302 | Bidirectional Single-Bus | **GPIO 4** | Ambient temperature ($^\circ\text{C}$) and relative humidity ($\%$) |
| **Power Supply Rail** | AMS1117-3.3 LDO | Power ($+3.3\,\text{V}$) | **3V3** | Clean regulated analog sensor rail |
| **System Ground** | Common Star Ground | Ground ($0\,\text{V}$) | **GND** | Shared low-noise analog/digital ground |

---

## 2. AD620 Instrumentation Amplifier Circuit

```text
               +5V / +3.3V
                   │
                [Pin 7]
                   │
  MPX10DP (+) ──► [Pin 3] (+IN)
                           AD620
  MPX10DP (-) ──► [Pin 2] (-IN)
                           [Pin 6] (V_OUT) ──► Low-Pass RC Filter ──► ESP32 GPIO 34
                   │
                [Pin 4]
                   │
                  GND

      Gain Resistor R_G (Pins 1 & 8):
         [Pin 1] ─── [ R_G = 620 Ω ] ─── [Pin 8]
         Gain G = 1 + (49,400 / 620) ≈ 80.68
```

### Low-Pass Anti-Aliasing Filter:
* **Series Resistor:** $R = 1.5\,\text{k}\Omega$
* **Shunt Capacitor:** $C = 1.0\,\mu\text{F}$
* **Cutoff Frequency:** $f_c = \frac{1}{2\pi R C} \approx 106\,\text{Hz}$ (filters high-frequency mechanical and EMI noise before the 100 Hz ADC).

---

## 3. High-Resolution Vector Circuit Diagrams
* Vector PDF Circuit Diagram: [`COPD_IEEE_RESEARCH_PAPER/figures/fig6_circuit_diagram.pdf`](../../COPD_IEEE_RESEARCH_PAPER/figures/fig6_circuit_diagram.pdf)
* High-Resolution PNG Circuit Diagram: [`COPD_IEEE_RESEARCH_PAPER/figures/fig6_circuit_diagram.png`](../../COPD_IEEE_RESEARCH_PAPER/figures/fig6_circuit_diagram.png)
