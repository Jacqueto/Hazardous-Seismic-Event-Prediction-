
# Seismic GA Predictor

**Project Title:** Hazardous Seismic Event Prediction Using Genetic Algorithm  
**Author:** Jacqueline Chiazor  
**Course:** CS 548 – Advanced Artificial Intelligence  
**Technology:** Python, Streamlit, scikit-learn

---

## Files Included

- `Genetic_Algorithm.py` – Main Streamlit web app
- `seismic_bumps.csv` – Cleaned dataset (from UCI repository)

---

##  Setup Instructions (Local Installation)

### 1. Clone or Download the Project

Make sure both the `.py` file and `.csv` are in the same folder.

### 2. Install Python Libraries

Open your terminal (Command Prompt, PowerShell, or Terminal), then install dependencies:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### 3. Run the Streamlit App

Navigate to the folder where your files are saved:

```bash
cd path/to/your/folder
```

Then run the app:

```bash
streamlit run Genetic_Algorithm.py
```

---

## What the App Does

- Loads and previews seismic dataset
- Displays class imbalance using a bar chart
- Runs Genetic Algorithm to select features
- Uses Random Forest to evaluate fitness
- Visualizes accuracy over generations
- Allows you to adjust mutation rate, population size, and generation count interactively

---

##  Dataset Source

**Seismic Bumps Dataset**  
From the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/seismic+bumps)

---

## Notes

- Built for CS 548 Final Project (Spring 2025)
- Feel free to modify `mutation_rate`, `pop_size`, or `n_generations` from the sidebar
- Supports experimentation and analysis for safety-critical prediction problems

---

## Contact

Jacqueline Chiazor  
Graduate Student, MS Computer Science  
Western Illinois University
