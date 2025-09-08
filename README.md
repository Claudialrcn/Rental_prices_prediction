# Rental Prices Prediction

**Author:** Claudia Alarcón Montesinos
**Date:** September 2025

## Project Overview

The rental housing market is highly dynamic, making accurate price prediction essential for property owners, prospective tenants, and real estate analysts.  
This project leverages real data from the province of Málaga to build predictive models for both rental and sales properties using machine learning techniques.  

The solution includes data extraction from Idealista's API, feature selection and preprocessing, model training with Linear Regression, SVM, Random Forest, and XGBoost, and an interactive dashboard built with Dash to explore predictions and visual trends.

This project was developed as part of a Master's Thesis for the program *Big Data, Artificial Intelligence, and Data Engineering*.

## Project Structure

- `dash_app/` – contains the Dash application and related modules  
- `data/` – datasets used for training and visualization  
- `models/` – trained model files  
- `utils/` – utility scripts for data extraction, cleaning, and training  
- `notebooks/` – Jupyter notebooks for analysis and experimentation  
- `main.py` – optional entry point for scripts  
- `pyproject.toml` – project dependencies and metadata  
- `uv.lock` – locked versions of all dependencies  

---

## Prerequisites

- Python ≥ 3.13  
- `uv` package manager (for local execution)  
- Docker (optional, for containerized execution)  

### About `uv`

The project is structured and executed using [`uv`](https://github.com/astral-sh/uv), a modern Python package manager and environment tool. Please consult the official documentation for detailed guidance.

## Local Execution

1. **Install `uv`**  

   - On Windows (PowerShell):

     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

   - On macOS/Linux:

     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   - Alternatively, `uv` can be installed via `pip`:

     ```bash
     pip install uv
     ```

2. **Synchronize project dependencies**  

   Navigate to the project root (where `pyproject.toml` is located) and run:

   ```bash
   uv sync
   ```

    This command will create and synchronize the virtual environment with all required dependencies specified in ``pyproject.toml``.

3. **Run the application locally**
    Once dependencies are synchronized, start the application:
    ```
    uv run dash_app/app.py
    ```
    Open your browser at `http://localhost:8050` to access the dashboard.

## Docker Execution
For portability and reproducibility, the project can also be executed using Docker.

1. **Build the Docker image**

    From the project root:

    ```
    docker build -t rental-prices-prediction .
    ```

2. **Run the Docker container**
    ```
    docker run -p 8050:8050 rental-prices-prediction
    ```

## Usage

Once the application is running locally or in Docker:

1. Use the interactive dashboard to request predictions for rental or sales prices.  
2. Explore visualizations and trends of the data through built-in graphs and maps.  
3. Inspect feature importance and model behavior directly within the dashboard interface.

---

## References

- [`uv` package manager](https://github.com/astral-sh/uv)  
- [Dash documentation](https://dash.plotly.com/)  
- [Scikit-learn documentation](https://scikit-learn.org/stable/)  

---

This README ensures reproducibility, allowing users to run the project either locally using `uv` or in a containerized environment with Docker. Both options support seamless execution of the interactive dashboard and access to predictive models.

