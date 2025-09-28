# Step 9: Fairness Analysis

## Objective

The primary goal of this step is to rigorously assess the fairness of our trained machine learning model. It is not enough for a model to be accurate; it must also be equitable and not perpetuate or amplify existing societal biases. This analysis specifically investigates whether the model's predictions are biased with respect to the borrower's **ethnicity**, which is a legally protected attribute.

We aim to answer two key questions:
1.  **Is the model unfair?** We measure if there are statistically significant disparities in model outcomes between different ethnic groups.
2.  **If so, why is it unfair?** We seek to identify the specific input features (e.g., income, location) that are the likely drivers of the detected bias.

## Methodology

To achieve this, we employ a two-part methodology based on established academic research in algorithmic fairness.

### 1. Fairness Metrics & Statistical Significance

First, we quantify the model's fairness using a set of standard metrics that evaluate different definitions of fairness:

-   **Statistical Parity (and Disparate Impact)**: These metrics assess whether all groups receive positive outcomes at similar rates. A significant difference may suggest that the model's benefit is not equally distributed.
-   **Equal Opportunity & Equalized Odds**: These metrics evaluate if the model's *accuracy* is consistent across groups. For example, is the model equally good at correctly identifying qualified applicants (true positives) regardless of their ethnicity?

Crucially, we don't just look at the raw differences. We use a **Chi-Squared statistical test** (`χ²`) to determine if the observed disparities are statistically significant or if they could have occurred simply by random chance.

### 2. Candidate Variable Analysis

If the statistical test reveals significant bias, we proceed to the diagnostic step. The "candidate variable" analysis is a technique to find the *root cause* of the unfairness.

The logic is as follows:
- An overall fairness test (`χ²_SP > d₁-α`) tells us the model is biased.
- We then hypothesize that another feature, say `zip_code`, might be the cause.
- We test this by analyzing the model's fairness *within a single zip code*. If, for that subgroup, the model is no longer biased against the protected group (`χ²_SP(zip_code_A) < d₁-α`), it implies that `zip_code` is a **candidate variable**. It is a feature that, through its correlation with ethnicity, is likely driving the model's biased behavior.

This analysis is performed automatically by the script for all relevant features, providing a list of variables that are the most likely sources of the model's unfairness.

### 3. Advanced Analyses

To provide a more nuanced view of fairness, two additional analyses are performed:

-   **Fairness Equivalence Testing (TOST)**: Standard statistical tests are designed to find *any* difference, even if it's tiny and practically meaningless. A TOST flips the question: it tests if the difference between groups is *small enough* to be considered practically equivalent. We define a tolerance threshold (`δ`), and the test determines if the disparity in average scores between groups falls within this "zone of fairness."

-   **Fairness Partial Dependence Plots (FPDP)**: This visualization shows the average predicted score for each protected group (e.g., for each ethnicity quartile). It provides a direct, easy-to-interpret view of how the model's output varies across groups, complementing the more abstract fairness metrics.

## How to Run the Analysis

The analysis can be executed via the command line.

```bash
# Make sure your dataset is in the ../data/ directory
# The script will auto-detect the protected attribute if a common name is found.
# For this project, specify 'Pct_afro_american' to ensure correctness.
# You can also specify the tolerance for the equivalence test with --delta.

python step_9_fairness/step_9.py --data "data/dataproject2025.csv" --protected Pct_afro_american --delta 0.05
```

## Outputs

The primary output is a self-contained **HTML report** (`step_9_fairness/result_html/fairness_report.html`) that consolidates all analyses and visualizations. The script also prints a summary to the console and saves individual graphs in the `step_9_fairness/graphs/` directory.

The report includes the following sections:

1.  **Setup Information**: Details on the dataset and the protected attribute used.
2.  **Group Distribution**: A table and a bar chart showing the size of each protected group (e.g., the number of data points in each ethnicity quartile).
3.  **Standard Fairness Metrics**: A table summarizing key metrics like Statistical Parity Difference (SPD), Disparate Impact (DI), and Equal Opportunity Difference.
4.  **Fairness Equivalence Test (TOST)**: The results of the equivalence test, indicating whether the difference in scores between groups is small enough to be considered practically insignificant.
5.  **Fairness Partial Dependence Plot (FPDP)**: A bar chart showing the average model score for each group, visualizing overall disparities.
6.  **Candidate Variable Analysis**: A summary of the diagnostic analysis, accompanied by a bar chart that highlights which other features are most likely contributing to the model's bias.
7.  **Score Distributions**: Histograms and boxplots showing if the model assigns systematically different scores to different groups.
8.  **ROC Curves per Group**: To check if the model's predictive power (AUC) is consistent across groups.
9.  **Confusion Matrices per Group**: To visualize the types of errors (false positives/negatives) the model makes for each group.
