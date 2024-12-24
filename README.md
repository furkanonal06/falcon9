# SpaceX Falcon 9 Launch Analysis and Dashboard

This repository contains an end-to-end project focused on analyzing SpaceX Falcon 9 rocket launches. The project explores key factors affecting mission success, employs machine learning models for predictions, and integrates findings into an interactive dashboard built with Dash Plotly.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview
This project analyzes 375 Falcon 9 launches and builds a predictive model for key engineering features like `GridFins` and `Legs`. It also includes a dashboard for interactive exploration and model predictions.

### Key Objectives:
1. Data preprocessing and feature engineering.
2. Predictive modeling for categorical variables.
3. Visualization and exploration using an interactive dashboard.

---

## Features
- **Data Analysis**: Exploratory data analysis and visualizations.
- **Feature Engineering**: Encoding and standardizing categorical and numerical features.
- **Machine Learning**: Predictive modeling for key engineering decisions.
- **Interactive Dashboard**:
  - Dropdown menus for selecting launch parameters.
  - Real-time predictions for `GridFins` and `Legs` based on user input.
  - Clear, user-friendly visualizations.

---

## Technologies Used
- **Python**: Data analysis, modeling, and dashboard implementation.
- **Dash Plotly**: For creating the interactive dashboard.
- **Pandas/NumPy**: For data preprocessing and wrangling.
- **Scikit-learn**: For building and evaluating predictive models.

---

## Dataset Description
The dataset combines REST API and web scraping data. It includes:

| **Feature**             | **Description**                                |
|-------------------------|-----------------------------------------------|
| `Flight_No`            | Flight identifier.                           |
| `Date`                 | Launch date.                                 |
| `Payload_Mass`         | Payload mass (kg).                           |
| `Launchsite`           | Launch site location.                        |
| `GridFins`, `Legs`     | Binary engineering features.                 |
| `Orbit`                | Orbit type (e.g., LEO, GTO).                 |
| `Block`                | Falcon 9 block version.                      |
| `Reused`, `FlightCount`| Reuse data and flight counts.                |
| `Longitude`, `Latitude`| Geospatial data for launch sites.            |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/falcon9-analysis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd falcon9-analysis
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Preprocess the dataset (if required):
   ```bash
   python preprocess.py
   ```

2. Train the machine learning models:
   ```bash
   python train_model.py
   ```

3. Run the dashboard application:
   ```bash
   python app.py
   ```

4. Open the dashboard in your browser at:
   ```
   http://127.0.0.1:8050/
   ```

---

## Project Structure
```
falcon9-analysis/
├── data/                   # Dataset files.
├── notebooks/              # Jupyter notebooks for exploration.
├── app.py                  # Dashboard application.
├── preprocess.py           # Data preprocessing script.
├── train_model.py          # Model training script.
├── requirements.txt        # Python dependencies.
├── README.md               # Project documentation.
└── models/                 # Trained model files.
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
