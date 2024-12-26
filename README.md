<div align="center">
  <a href="https://falcon9-webapp.onrender.com">
    <img src="assets/SpaceX-black.svg" width="350px" alt="SpaceX Falcon 9">
  </a>
  
  ## Falcon 9 Launch and Landing Analysis Web Application
<br/>

  [**Click here**](https://falcon9-webapp.onrender.com) to access the web app.
  
</div>


<br/>

This web-app is an end-to-end project focused on analyzing SpaceX Falcon 9 rocket launches and landings. The project explores key factors affecting mission success, employs machine learning models for predictions, and integrates findings into an interactive dashboard built with Dash Plotly. It also includes a notebook to explain processes like data gathering, web scrapping, explaratory analyses and building a predictive model for missing `GridFins` and `Legs` features in addition to the project's landing success predictive model.

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
The project focuses on creating a fully functional web app to explore and analyze SpaceX Falcon 9 rocket launches. With a combination of data preprocessing, feature engineering, and machine learning, the application predicts critical engineering features such as `GridFins` and `Legs`. The app also emphasizes responsive design and user experience, leveraging custom CSS to ensure accessibility across devices.

### Key Objectives:
1. Build an interactive and visually appealing web app for SpaceX Falcon 9 data.
2. Predictive modeling for landing outcome.
3. Enhance user experience through responsive design with custom CSS.
4. Implement data preprocessing and feature engineering techniques.
5. Feature engineering with predictive modeling for categorical variables.

---

## Features
- **Interactive Dashboard**:
  - Dropdown menus for selecting launch and landing parameters.
  - Intuitive visualizations for exploring launch data and trends.
  - Clear, user-friendly visualizations.
- **Machine Learning**:
  - Real-time predictions for landing outcome based on user input.
  - Predictive modeling to infer missing features like `GridFins` and `Legs`.
- **Data Analysis**: Exploratory data analysis with visual insights into trends and patterns.
- **Feature Engineering**: Encoding and standardizing categorical and numerical features.

---

## Technologies Used
- **Python**: Data analysis, modeling, and dashboard implementation.
- **Scikit-learn**: For building and evaluating predictive models.
- **CSS**: Custom styling for responsiveness and improved design.
- **Dash Plotly**: For creating the interactive dashboard.
- **Pandas/NumPy**: For data preprocessing and wrangling.
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
falcon9-webapp/
├── assets/                 # Project assets including CSS codes and images.
├── notebooks/              # Jupyter notebooks for exploration, data preprocessing and model training.
├── app.py                  # Dashboard application.
├── requirements.txt        # Python dependencies.
├── README.md               # Project documentation.
├── spacexcomplete.csv      # Data with encoded features for model training.
└── spacexvisual.csv        # Data for creating graphs.
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
