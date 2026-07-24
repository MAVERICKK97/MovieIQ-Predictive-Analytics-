"""
Stage 3 — Statistical Testing
T-test on a numeric feature, Chi-square test on genre association with success.
"""

import pandas as pd
from scipy import stats


def run_ttest(df: pd.DataFrame, feature: str = "popularity", alpha: float = 0.05) -> dict:
    """
    Null hypothesis (H0): the mean of `feature` is the same for successful
    and unsuccessful movies.
    """
    group_success = df.loc[df["success"] == 1, feature].dropna()
    group_fail = df.loc[df["success"] == 0, feature].dropna()

    t_stat, p_value = stats.ttest_ind(group_success, group_fail, equal_var=False)

    result = {
        "feature": feature,
        "t_statistic": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "mean_success": group_success.mean(),
        "mean_fail": group_fail.mean(),
        "conclusion": (
            f"Reject H0: {feature} differs significantly between successful and "
            f"unsuccessful movies (p={p_value:.4f})."
            if p_value < alpha else
            f"Fail to reject H0: no significant difference in {feature} "
            f"between the two groups (p={p_value:.4f})."
        ),
    }
    return result


def run_chi_square(df: pd.DataFrame, feature: str = "genre_primary", alpha: float = 0.05) -> dict:
    """
    Null hypothesis (H0): `feature` (e.g. genre) and success are independent.
    """
    contingency = pd.crosstab(df[feature], df["success"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    result = {
        "feature": feature,
        "chi2_statistic": chi2,
        "p_value": p_value,
        "dof": dof,
        "alpha": alpha,
        "significant": p_value < alpha,
        "conclusion": (
            f"Reject H0: {feature} is significantly associated with success "
            f"(p={p_value:.4f})."
            if p_value < alpha else
            f"Fail to reject H0: no significant association between {feature} "
            f"and success (p={p_value:.4f})."
        ),
    }
    return result


if __name__ == "__main__":
    from data_prep import full_pipeline

    df = full_pipeline("movies.csv")

    ttest_result = run_ttest(df, "popularity")
    print("T-TEST RESULT:")
    for k, v in ttest_result.items():
        print(f"  {k}: {v}")

    chi_result = run_chi_square(df, "genre_primary")
    print("\nCHI-SQUARE RESULT:")
    for k, v in chi_result.items():
        print(f"  {k}: {v}")
