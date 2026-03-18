# Telco Churn Assistant
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/ML%20Model-Logistic%20Regression-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20GPT--5-black?logo=openai&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Telco%20Customer%20Churn-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

# 🚀 Overview

A demo LLM application that combines the **OpenAI Agents SDK**, **Model Context Protocol (MCP)**, and a **logistic regression churn model** to help telecom retention teams analyze customer churn risk and receive AI-generated retention recommendations.

The assistant uses a multi-agent architecture: a **Triage Agent** that routes incoming requests and a **Retention Strategist** that performs the data retrieval, model inference and strategy generation, all coordinated through MCP tools.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [MCP Tools Reference](#mcp-tools-reference)
- [Sample Examples](#sample-examples)
- [Risk Segments & Retention Offers](#risk-segments--retention-offers)

---

## Architecture Overview

```
User Prompt
    │
    ▼
Triage Agent  ──(handoff)──▶  Retention Strategist
    │                                │
    └── MCP Tools ◀──────────────────┘
           │
    ┌──────┼──────────────────────┐
    │      │                      │
dataset  predict_churn    get_retention_offers
overview  (LogReg model)    (rule-based)
    │
get_customer_profile / list_customer_ids
```

- **MCP Server** (`telco_churn_server.py`): A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes five tools the agents can call.
- **Retention Strategist**: A specialist agent that retrieves a customer profile, runs churn prediction, and maps the result to a retention strategy.
- **Triage Agent**: The front-door agent. It routes churn-related requests to the Retention Strategist and politely refuses out-of-scope questions.

---

## Data Source

The dataset used is the **[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** dataset, originally published by IBM as part of the Watson Analytics sample datasets.

| Property | Value |
|---|---|
| Rows | ~7,043 customers |
| Columns | 21 (demographics, services, account info, churn label) |
| Target column | `Churn` (Yes / No) |
| Churn rate | ~26.5% |

**Key features include:**

- *Demographics*: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- *Services*: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `StreamingTV`, …
- *Account*: `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges`, `tenure`

The CSV file is already included at `data/telco-customer-churn.csv`. No Kaggle download is required to run the demo.

---

## Project Structure

```
Telco-Churn-Assistant/
├── telco_churn_server.py        # FastMCP server — exposes tools to the agents
├── telco-churn-assistant.ipynb  # Jupyter notebook demo
├── requirements.txt
├── data/
│   └── telco-customer-churn.csv # Source dataset
└── artifacts/
    ├── churn_model.joblib        # Pre-trained logistic regression model
    └── training_columns.json     # Feature columns used during training
```

> The model is trained automatically on first use if the artifact files are missing.

---

## Installation

### Prerequisites

- Python 3.10 or newer
- An **OpenAI API key** with access to `gpt-5` or later models

### 1. Clone the repository

```bash
git clone https://github.com/pradeepchan/Telco-Churn-Assistant.git
cd Telco-Churn-Assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."   # macOS / Linux
# set OPENAI_API_KEY=sk-...      # Windows CMD
```

---

## Running the Application

Launch the notebook and run cells top-to-bottom:

```bash
jupyter notebook telco-churn-assistant.ipynb
```

The notebook:
1. Validates the `OPENAI_API_KEY` environment variable.
2. Defines the two agents (Triage Agent and Retention Strategist).
3. Starts the MCP server as a subprocess via `MCPServerStdio`.
4. Runs a series of demo prompts and prints the assistant's responses.

---

## MCP Tools Reference

The server exposes five tools that the agents can call:

| Tool | Parameters | Description |
|---|---|---|
| `dataset_overview` | — | Returns dataset shape, churn rate, and sample column names |
| `list_customer_ids` | `limit` (default 10) | Returns a list of customer IDs for testing |
| `get_customer_profile` | `customer_id` | Returns the full raw CSV row for a customer |
| `predict_churn` | `customer_id` | Runs logistic regression and returns probability, label, and risk segment |
| `get_retention_offers` | `risk_segment` (`Low`/`Medium`/`High`) | Returns a list of retention offers for the given risk level |

---

## Sample Examples

All examples below are taken directly from the notebook. Pass any string as `prompt` to `churn_assistant()`.

### 1. Summarize the dataset

```python
response = await churn_assistant("Summarize the dataset that powers this churn demo.")
print(response)
```

**What happens**: The assistant calls `dataset_overview`, retrieves the row count, column count, churn rate and example feature names, then produces a plain English summary.

**Example output** (abbreviated):
```
The dataset contains 7,032 telecom customers across 21 attributes.
Approximately 26.5% of customers have churned.
Key features include tenure, MonthlyCharges, Contract type, and InternetService.
```

---

### 2. List valid customer IDs

```python
response = await churn_assistant("List 5 valid customer IDs from the dataset.")
print(response)
```

**What happens**: The assistant calls `list_customer_ids(limit=5)` and formats the result as a readable list.

**Example output**:
```
Here are 5 valid customer IDs from the dataset:
1. 7590-VHVEG
2. 5575-GNVDE
3. 3668-QPYBK
4. 7795-CFOCW
5. 9237-HQITU
```

---

### 3. Analyze a single customer's churn risk

```python
response = await churn_assistant(
    "Analyze customer 7590-VHVEG using the real dataset and prediction tool. "
    "Provide a risk level, main churn drivers when churn level is high or medium, "
    "and recommended retention offers."
)
print(response)
```

**What happens**:
1. `get_customer_profile("7590-VHVEG")` fetches the customer's attributes.
2. `predict_churn("7590-VHVEG")` returns the churn probability and risk segment.
3. `get_retention_offers(risk_segment)` fetches the matching retention offers.
4. The Retention Strategist synthesizes all three results into a natural-language recommendation.

**Example output** (abbreviated):
```
Customer 7590-VHVEG — Risk Level: High

Churn Probability: 78%

Main Churn Drivers:
- Month-to-month contract with no long-term commitment
- Fiber optic internet service (correlated with higher churn)
- No online security or tech support add-ons
- Relatively low tenure (1 month)

Recommended Retention Offers:
- 20% discount for 6 months
- Free tech support for 3 months
- Annual contract upgrade incentive
```

---

### 4. Compare two customers

```python
response = await churn_assistant(
    "Explain how the retention offer recommendations differ "
    "for customers 7590-VHVEG and 4445-ZJNMU."
)
print(response)
```

**What happens**: The assistant retrieves profiles and predictions for both customers, then performs comparative reasoning to explain why their risk levels and recommended offers differ.

**Example output** (abbreviated):
```
Customer 7590-VHVEG
- Risk level: Medium (predicted churn probability ~0.64)
- Recommended offers: 10% discount for 3 months, Bundle upgrade offer, Loyalty points campaign

Customer 4445-ZJNMU
- Risk level: High (predicted churn probability ~0.79)
- Recommended offers: 20% discount for 6 months, Free tech support for 3 months, Annual contract upgrade incentive

Summary:
The second customer receives deeper, longer incentives because churn risk is higher.
```

---

### 5. Explain the model's reasoning

```python
response = await churn_assistant(
    "Explain how the model is predicting churn risk for customer 7590-VHVEG."
)
print(response)
```

**What happens**: The Retention Strategist uses the customer profile and churn probability returned by the model to explain, in plain English which customer attributes are most likely driving the prediction.

**Example output** (abbreviated):
```
Customer 7590-VHVEG is Medium risk with predicted churn probability ~0.64.

Likely churn drivers:
- Very short tenure (1 month)
- Month-to-month contract
- Electronic check payment and paperless billing pattern
- No OnlineSecurity and no TechSupport

Recommended offers:
- 10% discount for 3 months
- Bundle upgrade offer
- Loyalty points campaign
```

---

### 6. Out-of-scope question (guardrail demo)

```python
response = await churn_assistant(
    "What were the main reasons for the fall of the Roman Empire?"
)
print(response)
```

**What happens**: The Triage Agent detects that the question is unrelated to telecom churn analysis and refuses to answer, demonstrating how domain restrictions are enforced.

**Example output**:
```
I'm designed to assist with telecom customer churn analysis and questions
related to this demo system. I cannot answer unrelated questions.
```

---

## Risk Segments & Retention Offers

The model's output probability is mapped to one of three risk segments:

| Segment | Probability Range | Retention Offers |
|---|---|---|
| **High** | ≥ 0.75 | 20% discount for 6 months, Free tech support for 3 months, Annual contract upgrade incentive |
| **Medium** | 0.40 – 0.74 | 10% discount for 3 months, Bundle upgrade offer, Loyalty points campaign |
| **Low** | < 0.40 | Standard loyalty communication, Periodic engagement reminder |

---

# 📜 License

This project is open source and available under the [MIT License](LICENSE).