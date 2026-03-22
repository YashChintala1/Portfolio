# 📊 Empirical Bayes Analysis of NFL Catch Rates

## 📌 Overview

This project was completed as part of coursework in an Introduction to Machine Learning class. The goal is to estimate the true catch probability (catch rate) of NFL players using statistical modeling techniques.

The project compares three different modeling approaches:

* Unpooled (individual estimates)
* Fully pooled (global estimate)
* Empirical Bayes (partially pooled model)

The Empirical Bayes approach is the primary focus, demonstrating how information can be shared across groups (players) to produce more stable and realistic estimates.

---

## 🎯 Objectives

* Model player catch rates using probabilistic methods
* Understand the limitations of MLE (Maximum Likelihood Estimation)
* Apply Bayesian reasoning using conjugate priors
* Implement Empirical Bayes to improve estimation for sparse data
* Compare pooled vs unpooled vs partially pooled approaches

---

## 🧠 Methodology

### 1. Unpooled Model (MLE)

* Each player is modeled independently
* Catch probability estimated using Maximum Likelihood Estimation
* Limitation: Highly variable for players with few observations

### 2. Fully Pooled Model

* Assumes all players share the same catch probability
* Produces a single global estimate
* Limitation: Ignores individual differences between players

### 3. Empirical Bayes Model (Primary Model)

* Likelihood: Binomial distribution
* Prior: Beta distribution
* Posterior: Beta distribution (conjugate prior)

Steps:

1. Estimate prior parameters (α, β) from all players
2. Use this prior for each individual player
3. Compute posterior distributions and posterior means

Key Feature:

* Produces "shrinkage estimates" where extreme values are pulled toward the population mean

---

## 🛠️ Tools & Technologies

* **Language:** R
* **Environment:** RMarkdown / Quarto
* **Libraries:**

  * tidyverse (dplyr, ggplot2, tidyr, readr, purrr)

---

## 📈 Key Insights

* MLE estimates are unreliable for small sample sizes
* Fully pooled estimates are overly simplistic
* Empirical Bayes provides a balance between individual and group-level information
* Extreme player estimates are "shrunk" toward the overall average

---

## 📂 Project Structure

```
.
├── completed_midterm.html   # Rendered report
├── README.md                # Project documentation
```

---

## ⚠️ Disclaimer

This project is part of academic coursework and is shared solely for the purpose of demonstrating my understanding of statistical modeling and machine learning concepts.

* This work is **not intended for reproduction, submission, or academic reuse**.
* Any use of this material for coursework or academic credit by others would constitute **academic misconduct**.
* The code, analysis, and written content reflect my individual learning and should be treated as such.

If you are viewing this as a recruiter or collaborator, this project is intended to showcase:

* My ability to apply statistical theory to real-world data
* My understanding of Bayesian and Empirical Bayes methods
* My proficiency in R and data analysis workflows

---

## 🙌 Author

Yashasvi Chintala

---

## 📬 Notes

If you have questions about this project or would like a walkthrough of the methodology, feel free to reach out.
