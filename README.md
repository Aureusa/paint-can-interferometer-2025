# 🎯 Paint Can L-Band Interferometer 2025

**Leiden Radio Astronomy Class – Fall 2025**

> *Building imaging radio interferometer from scratch.*

---

## 📡 Project Overview

This repository contains all the code, configurations, and documentation for the **Paint Can L-band Interferometer**, a collaborative project by students at **Leiden Observatory**.  
Our mission: design, construct, and operate a **L-band interferometer** capable of imaging bright radio sources.

---

## 🧩 Project Structure

Each sub-team is responsible for a critical component of the full interferometric system:

| Subsystem | Leads | Responsibilities |
|------------|--------|------------------|
| **Astronomical Requirements** | Joshua, Huson, Boyd | Source selection, flux & noise simulations, sky map generation |
| **Technical & Operational Requirements** | Leonor, Moritz | Power, bandwidth, hardware specs |
| **Array Configuration** | Sam, Adriaan, Ella | Antenna geometry, UV-coverage, PSF simulations |
| **Physical Setup** | Ethan, Konstantinos, Antonios | Site layout, antenna alignment, setup plan |
| **Data Acquisition** | Eirini, Petar, Carlotta | Signal capture, file formats, data I/O Python tools |
| **Synchronization** | Niels, Tristan | GPS clock configuration, timing & delay calibration |
| **Mission Assurance** | Patrick, Maurien, Britt | Observation checklists, in-field validation, data QA |

---
## 📚 Getting Started
To get started with the Paint Can L-band Interferometer project, follow these steps:
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Aureusa/paint-can-interferometer-2025
2. **Create a Virtual Environment**  
   It's best practice to use a virtual environment to manage dependencies. Make sure you have Python 3.9 installed. Then, create a virtual environment:
   ```bash
   python3.9 -m venv MyVenv
   ```
3. **Activate the Virtual Environment**  
   ```bash
   source MyVenv/bin/activate  # If for some reason you use Windows use `MyVenv\Scripts\activate`
   ```
4. **Install Dependencies**  
   Then, install the required packages:
   ```bash
   cd paint-can-interferometer # Make sure you are in the project directory
   pip install -r requirements.txt
   ```
---

---

## 🛠️ Pushing Code

---
When pushing code to this repository, please ensure:
- Document your code so others can understand it.
- Don't break the existing codebase.
- If you use exotic external libraries, list them in `requirements.txt` so that others can install them easily.
- If you want to ignore certain files or directories, add them to the `.gitignore` file.
- If your functions or classes assume certain preconditions, a certain type, or a certain shape of data, please add some sort of validation to them and document them using docstrings.
- If you want to declare project-wide dependencies in a form of global constants, please use the `.env` file and load them using the `dotenv` library - so that we don't have "magic numbers" in the code. For more info read: https://pypi.org/project/python-dotenv/.
- It would be nice if you conform to PEP 8 style guidelines.

---

## 🧠 Repository Contents