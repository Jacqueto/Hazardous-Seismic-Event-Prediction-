import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



df = pd.read_csv("seismic_bumps.csv")
st.title("Hazardous Seismic Event Prediction Using Genetic Algorithm")
st.markdown("""This app uses a genetic algorithm to select the most relevant features for predicting dangerous seismic events in coal mines.  
Designed by Jacqueline Chiazor for CS 548 Project""")

st.subheader(" Dataset Preview")
st.dataframe(df.head())

st.subheader("Class Distribution")
st.markdown("""
This chart shows the number of seismic records in each category:
- **0 = Non-hazardous**: No dangerous seismic activity detected
- **1 = Hazardous**: A hazardous seismic bump is expected

Understanding class distribution helps identify if the dataset is imbalanced.
If very few samples belong to class 1, accuracy alone can be misleading.
""")
class_counts = df["class"].value_counts().sort_index()
fig1, ax1 = plt.subplots(figsize=(5, 3))
bars = ax1.bar(["Non-Hazardous (0)", "Hazardous (1)"], class_counts, color=["seagreen", "crimson"])
ax1.set_title("Class Distribution of Seismic Events", fontsize=12)
ax1.set_xlabel("Event Class")
ax1.set_ylabel("Number of Records")


for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', fontsize=10)

st.pyplot(fig1)



df = pd.get_dummies(df, drop_first=True)
X = df.drop("class", axis=1).values
y = df["class"].astype(int).values


with st.sidebar:
    st.sidebar.title("Genetic Algorithm Settings")
    st.markdown("Choose the Genetic Algorithm parameters:")
    pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
    n_generations = st.sidebar.slider("Number of Generations", 5, 50, 20)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.1)


def initialize_population(pop_size, n_features):
    return np.random.randint(2, size=(pop_size, n_features))

def fitness(individual, X, y):
    if np.count_nonzero(individual) == 0:
        return 0
    X_selected = X[:, individual == 1]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def selection(population, scores):
    idx = np.argsort(scores)[-2:]
    return population[idx]

def crossover(parents):
    point = random.randint(1, len(parents[0]) - 1)
    child1 = np.concatenate([parents[0][:point], parents[1][point:]])
    child2 = np.concatenate([parents[1][:point], parents[0][point:]])
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


st.markdown("### Run Genetic Algorithm")
st.markdown("Click the button below to start feature selection using a genetic algorithm. "
            "The GA will evolve feature subsets over generations to improve model accuracy.")

if st.button("Run Genetic Algorithm"):
    n_features = X.shape[1]
    population = initialize_population(pop_size, n_features)
    best_individual = None
    best_score = 0
    history = []

    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for generation in range(n_generations):
        scores = [fitness(ind, X, y) for ind in population]
        best_gen_score = max(scores)
        history.append(best_gen_score)
        status_text.text(f"Running Generation {generation + 1} - Best Accuracy: {best_gen_score:.4f}")
        progress_bar.progress((generation + 1) / n_generations)

        if best_gen_score > best_score:
            best_score = best_gen_score
            best_individual = population[np.argmax(scores)]

        parents = selection(population, scores)
        children = []

        for _ in range(pop_size // 2):
            c1, c2 = crossover(parents)
            children.append(mutate(c1, mutation_rate))
            children.append(mutate(c2, mutation_rate))

        population = np.array(children)


    progress_bar.empty()
    status_text.text("Genetic Algorithm completed!")

    # Final model evaluation on best individual
    X_selected = X[:, best_individual == 1]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show detailed performance
    report = classification_report(y_test, y_pred, target_names=["Non-Hazardous", "Hazardous"], output_dict=True)
    hazardous_metrics = report["Hazardous"]

    st.markdown("### Final Classification Metrics")
    st.write(f"**Precision (Hazardous):** {hazardous_metrics['precision']:.2f}")
    st.write(f"**Recall (Hazardous):** {hazardous_metrics['recall']:.2f}")
    st.write(f"**F1 Score (Hazardous):** {hazardous_metrics['f1-score']:.2f}")

    st.success(f"Best Accuracy Achieved: **{best_score:.4f}**")
    selected_features = np.where(best_individual == 1)[0]
    st.markdown("### Selected Feature Indices:")
    st.code(selected_features.tolist())

    # Accuracy plot
    st.subheader("Accuracy Over Generations")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, n_generations + 1), history, marker='o')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Accuracy")
    ax2.set_title("Fitness Progression")
    st.pyplot(fig2)


st.markdown("<hr style='margin-top: 40px;'><center><small>CS 548 Project | Jacqueline Chiazor | Spring 2025</small></center>", unsafe_allow_html=True)