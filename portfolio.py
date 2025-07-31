import streamlit as st
import plotly as plotly
import pandas as pd
import numpy as np
import re
import os
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- Configuration ---
WEEKLY_DATA_PATH = "weekly_comparison.csv"
WEEKS_PER_YEAR = 52

def calculate_portfolio_performance(weights, mu, cov_mat, weekly_rf_series, n_obs):
    """
    Calculates annualised performance metrics for a specified portfolio.

    Args:
        weights (numpy.ndarray): The portfolio weights allocated to each asset.
        mu (pandas.Series): Mean weekly logarithmic returns of the assets.
        cov_mat (pandas.DataFrame): Covariance matrix of weekly logarithmic returns.
        weekly_rf_series (pandas.Series): Time series of weekly risk-free rates (e.g., DGS10 in percent).
                                            Note: DGS10 values are annualized yields, observed weekly.
        n_obs (int): Number of observations used in return calculations.

    Returns:
        dict: A dictionary containing annualised Return, Volatility, Sharpe Ratio,
              and a 95% confidence interval for the return.
    """
    weights = np.array(weights)

    # Calculate portfolio mean weekly log return
    portfolio_mean_weekly_log_return = np.sum(mu * weights)

    # Annualise portfolio log return
    annualized_log_return = portfolio_mean_weekly_log_return * WEEKS_PER_YEAR

    # Convert annualised log return to annualised simple (geometric) return
    annualized_simple_return = np.exp(annualized_log_return) - 1

    # Annualise portfolio volatility from log returns
    annualized_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(WEEKS_PER_YEAR)

    # Compute the 95% confidence interval for the annualised simple return
    return_ci_lower, return_ci_upper = np.nan, np.nan
    if n_obs > 1 and annualized_volatility > 0:
        z_score = 1.96  # For 95% confidence interval (two-tailed)

        # Standard error of the annualised log return
        # This scales the weekly volatility by sqrt(WEEKS_PER_YEAR / n_obs) for the annualised log return
        se_annualized_log_return = annualized_volatility * np.sqrt(WEEKS_PER_YEAR / n_obs)

        # Confidence Interval for the annualised log return
        ci_log_lower = annualized_log_return - z_score * se_annualized_log_return
        ci_log_upper = annualized_log_return + z_score * se_annualized_log_return

        # Transform log return CI to simple return CI
        return_ci_lower = np.exp(ci_log_lower) - 1
        return_ci_upper = np.exp(ci_log_upper) - 1

    # Calculate the mean weekly risk-free rate from the series
    # DGS10 is in percent and represents an annualized yield.
    # We take the mean of these annualized yields, convert to decimal.
    mean_annual_rf_from_series = weekly_rf_series.mean() / 100
    # Convert this average annual yield to an equivalent average weekly simple rate.
    mean_weekly_rf_for_sharpe = (1 + mean_annual_rf_from_series)**(1/WEEKS_PER_YEAR) - 1

    # Annualise the mean weekly risk-free rate (from the average weekly simple rate)
    annualized_rf = (1 + mean_weekly_rf_for_sharpe)**WEEKS_PER_YEAR - 1

    # Compute Sharpe Ratio
    sharpe_ratio = (annualized_simple_return - annualized_rf) / annualized_volatility if annualized_volatility != 0 else 0

    return {
        "Return": annualized_simple_return * 100,
        "Return_95_CI_Lower": return_ci_lower * 100,
        "Return_95_CI_Upper": return_ci_upper * 100,
        "Volatility": annualized_volatility * 100,
        "SharpeRatio": sharpe_ratio
    }

def optimize_portfolio(mu, cov_mat, weekly_rf_series, n_obs):
    """
    Identifies the portfolio exhibiting the maximum Sharpe ratio through optimization.

    Args:
        mu (pandas.Series): Mean weekly logarithmic returns for each asset.
        cov_mat (pandas.DataFrame): Covariance matrix of weekly logarithmic returns.
        weekly_rf_series (pandas.Series): Time series of weekly risk-free rates (e.g., DGS10 in percent).
                                            Note: DGS10 values are annualized yields, observed weekly.
        n_obs (int): Number of observations used for performance computation.

    Returns:
        tuple: A tuple comprising:
                - dict: Performance metrics of the optimized portfolio.
                - numpy.ndarray: Optimal weights for the assets.
    """
    num_assets = len(mu)

    # Calculate the mean weekly risk-free rate from the series for optimization
    # DGS10 is in percent and represents an annualized yield.
    # We take the mean of these annualized yields, convert to decimal.
    mean_annual_rf_from_series = weekly_rf_series.mean() / 100
    # Convert this average annual yield to an equivalent average weekly simple rate.
    mean_weekly_rf_for_opt = (1 + mean_annual_rf_from_series)**(1/WEEKS_PER_YEAR) - 1

    def neg_sharpe_ratio(weights):
        """Helper function to compute the negative Sharpe ratio for minimization."""
        portfolio_mean_weekly_log_return = np.sum(mu * weights)
        annualized_log_return = portfolio_mean_weekly_log_return * WEEKS_PER_YEAR
        annualized_simple_return = np.exp(annualized_log_return) - 1
        annualized_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(WEEKS_PER_YEAR)
        # Annualize the mean weekly risk-free rate for use in Sharpe calculation
        annualized_rf = (1 + mean_weekly_rf_for_opt)**WEEKS_PER_YEAR - 1

        if annualized_volatility == 0:
            return 0
        return -(annualized_simple_return - annualized_rf) / annualized_volatility

    # Constraints: sum of weights must be 1.
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    # Bounds for each weight (long-only portfolio).
    bounds = tuple((0, 1) for _ in range(num_assets))
    # Initial guess: an equally weighted portfolio.
    initial_weights = np.array(num_assets * [1. / num_assets])

    # Optimize to minimize the negative Sharpe ratio.
    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    # Calculate performance of the optimal portfolio
    optimal_performance = calculate_portfolio_performance(optimal_weights, mu, cov_mat, weekly_rf_series, n_obs)

    return optimal_performance, optimal_weights


def plot_weights_comparison(user_weights, optimal_weights, asset_names):
    """
    Generates a Plotly bar chart comparing user-defined and optimal portfolio weights.

    Args:
        user_weights (numpy.ndarray): User-specified weights (normalised).
        optimal_weights (numpy.ndarray): Optimal weights from portfolio optimization.
        asset_names (list): Names of the assets corresponding to the weights.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the bar chart.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=asset_names,
        y=user_weights * 100,
        name='User Weights',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=asset_names,
        y=optimal_weights * 100,
        name='Max Sharpe Weights',
        marker_color='orange'
    ))
    fig.update_layout(
        barmode='group',
        title_text='<b>Portfolio Allocation: User vs. Max Sharpe Ratio</b>',
        xaxis_title="Assets",
        yaxis_title="Weight (%)",
        legend_title="Portfolio Type",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"
        )
    )
    return fig


@st.cache_data
def load_data(data_path):
    """
    Loads weekly data from a CSV file.

    Args:
        data_path (str): The file path to the CSV data.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded data.
    """
    if not os.path.exists(data_path):
        st.error(f"FATAL: The data file '{data_path}' was not found.")
        st.stop()
    return pd.read_csv(data_path, parse_dates=["Date"])

def get_currencies(columns):
    """
    Identifies unique currency codes present in the DataFrame columns.

    Args:
        columns (list): A list of column names.

    Returns:
        list: A sorted list of unique, uppercase currency codes.
    """
    currencies = set()
    for col in columns:
        match = re.search(r'_([A-Z]{3})$', col)
        if match:
            currencies.add(match.group(1))
    return sorted(list(currencies))

def main():
    """
    The primary function to execute the Streamlit Portfolio Analysis Application.
    """
    st.set_page_config(layout="wide")
    st.title("Interactive Portfolio Performance Analysis")

    # --- Information & Methodology Section ---
    st.header("Information & Methodology")
    with st.expander("Click to view Methodology Details"):
        st.markdown(r"""
        This interactive application allows users to analyse and compare the performance of user-defined portfolios against an optimised portfolio (Maximum Sharpe Ratio). The analysis is based on historical weekly price data for Gold, Nifty 50, and S&P 500, denominated in USD.

        ### Data Sources:
        * **Gold Spot Price Data:** Sourced from gold.org (via ICE Data), representing price per gram.
        * **Nifty 50 Index Data:** Sourced from TSDB (`in.index.nifty.closing`).
        * **S&P 500 Index Data:** Sourced from Yahoo Finance (`^GSPC`).
        * **DAX Index Data:** Sourced from Yahoo Finance (`^GDAXI`).
        * **FTSE Index Data:** Sourced from Yahoo Finance (`^FTSE`).
        * **Nikkei 225 Index Data:** Sourced from Yahoo Finance (`^N225`).
        * **Exchange Rate Data:** Sourced from TSDB (`in.curr`,`de.curr`,`jp.curr`,`gb.curr`), utilised for currency conversions in the underlying data preparation.
        * **Risk-Free Rate:** The 10-Year US Treasury Yield (DGS10) is employed as the proxy for the USD risk-free rate. **It is provided as an annualized yield, observed weekly, and aligned with asset return dates.**

        ### Data Processing and Preparation:
        1.  **Data Collection:** Daily time-series data were procured for all assets and exchange rates.
        2.  **Temporal Alignment:** All datasets were synchronised to a common date range (from 3rd January 1995 to 12th June 2025) to ensure consistency.
        3.  **Frequency Conversion:** Daily data were resampled to a weekly frequency, typically using the last available observation (e.g., Friday's closing price).
        4.  **Missing Data Imputation:** Gaps in the weekly time series were addressed using a forward-fill imputation method.
        5.  **Currency Normalisation:** All series were converted to USD denominations using corresponding weekly exchange rates. For this application, only non USD-denominated assets are considered.
            **Note:** The conversion of non-USD denominated assets to USD using exchange rates can introduce additional **volatility** and impact **returns** due to fluctuations in currency exchange rates.

        ### Calculation of Performance Metrics:
        * **Weekly Logarithmic Returns:** Computed using the formula:
            $$\text{log\_return}_t = \ln(\text{Price}_t) - \ln(\text{Price}_{t-1})$$
        * **Annualised Geometric Return (%):** Derived from the mean of the weekly logarithmic returns. This represents the compound annual growth rate.
            $$\left( e^{\overline{\text{log\_return}_{\text{weekly}}} \times 52} \right) - 1$$
        * **Annualised Volatility (%):** Calculated by scaling the standard deviation of weekly logarithmic returns. This is a measure of the dispersion of returns.
            $$\sigma_{\text{weekly\_log\_return}} \times \sqrt{52}$$
        * **Sharpe Ratio:** Computed as (Portfolio Annualised Geometric Return - Annualised Risk-Free Rate) / Portfolio Annualised Volatility. This metric quantifies the risk-adjusted return of a portfolio.
        * **95% Confidence Interval for Annualised Return:** This interval furnishes a range within which the true annualised geometric return of the portfolio is likely to reside 95% of the time. It is derived by first calculating the confidence interval for the annualised logarithmic return and subsequently transforming these bounds using the exponential function to obtain the confidence interval for the annualised simple (geometric) return.

        ### Portfolio Optimisation:
        * **Input Data:** The analysis utilises the previously computed weekly logarithmic returns for Gold, Nifty 50, and S&P 500.
        * **Risk-Free Rate:** The 10-Year US Treasury Yield (DGS10) time series is used. **For Sharpe Ratio calculation and optimization, the mean of this time-varying annualized risk-free rate over the observation period is used.**
        * **Parameter Estimation:** The expected returns vector (mean of weekly log returns) and the covariance matrix are estimated empirically from the historical weekly return data.
        * **Maximum Sharpe Ratio (MSR) Portfolio:** This optimisation technique aims to maximise the risk-adjusted return, identifying the "Tangency Portfolio" on the efficient frontier. The weights for this portfolio are determined by solving an optimisation problem.
        """)

    weekly_data = load_data(WEEKLY_DATA_PATH)
    all_cols = weekly_data.columns.tolist()

    # Ensure 'DGS10' column exists for risk-free rate calculations
    if 'DGS10' not in weekly_data.columns:
        st.error("FATAL: 'DGS10' (10-Year US Treasury Yield) column not found in the data. This is required for risk-free rate calculations.")
        st.stop()

    currencies = get_currencies(all_cols)

    selected_currency = 'USD'

    if not currencies and selected_currency == 'USD':
        st.warning("No currency suffixes (e.g., '_USD') found in data columns. Proceeding with USD as the assumed currency for all assets.")

    # Asset columns are identified by the selected currency suffix.
    asset_cols = sorted([col for col in all_cols if col.endswith(f'_{selected_currency}')])
    if not asset_cols:
        st.error(f"No asset columns found for currency '{selected_currency}'. Please ensure your data has columns ending with '_{selected_currency}'.")
        st.stop()

    # Initialize session state for weights and results display flag
    if 'weights' not in st.session_state:
        st.session_state['weights'] = {asset: 0.0 for asset in asset_cols}
    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = False
    if 'last_performance' not in st.session_state:
        st.session_state['last_performance'] = None
    if 'last_optimal_performance' not in st.session_state:
        st.session_state['last_optimal_performance'] = None
    if 'last_user_weights' not in st.session_state:
        st.session_state['last_user_weights'] = None
    if 'last_optimal_weights' not in st.session_state:
        st.session_state['last_optimal_weights'] = None
    if 'last_asset_names' not in st.session_state:
        st.session_state['last_asset_names'] = None


    st.sidebar.header("Portfolio Weights (%)")
    with st.sidebar.form(key='weights_form'):
        weights_input = {}
        for asset in asset_cols:
            display_asset_name = asset.replace(f'_{selected_currency}', '')
            # Rename GSPC to S&P 500 for display
            if display_asset_name == 'GSPC':
                display_asset_name = 'S&P 500'
            # Use session state to manage the value of the number input
            weights_input[asset] = st.number_input(
                f"Weight for {display_asset_name}",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['weights'].get(asset, 0.0), # Get current value from session state
                step=1.0,
                format="%.1f",
                key=f"weight_{asset}" # Unique key for each input
            )

        col_form1, col_form2 = st.columns(2)

        with col_form1:
            submit_button = st.form_submit_button(label='Calculate')
        with col_form2:
            # Add a reset button
            reset_button = st.form_submit_button(label='Reset')

    # Handle reset action
    if reset_button:
        for asset in asset_cols:
            st.session_state['weights'][asset] = 0.0 # Reset session state values
            # Also clear the specific key used by the number_input
            if f"weight_{asset}" in st.session_state:
                del st.session_state[f"weight_{asset}"]
        st.session_state['show_results'] = False # Hide results on reset
        st.session_state['last_performance'] = None # Clear previous results
        st.session_state['last_optimal_performance'] = None
        st.session_state['last_user_weights'] = None
        st.session_state['last_optimal_weights'] = None
        st.session_state['last_asset_names'] = None
        st.rerun() # Force a rerun to re-initialize the widgets

    total_weight = sum(weights_input.values()) # Use weights_input for the sum check

    # --- New: Check weights sum before proceeding with calculations and trigger display ---
    if submit_button:
        if not np.isclose(total_weight, 100.0):
            st.error(f"Error: Your portfolio weights sum to {total_weight:.1f}%, but they must sum to exactly 100% to proceed with the analysis. Please adjust the weights in the sidebar.(You DUMMY!)")
            st.session_state['show_results'] = False # Hide results if validation fails
            # Removed st.stop() to allow user to correct weights without full app restart.
        else:
            # Update session state with current weights for persistence
            for asset in asset_cols:
                st.session_state['weights'][asset] = weights_input[asset]

            # Prepare data for calculation, including DGS10 for alignment
            required_cols = asset_cols + ["Date", "DGS10"]
            currency_specific_data = weekly_data[required_cols].copy().dropna().sort_values(by="Date")

            if len(currency_specific_data) < 2:
                st.error("Not enough historical data points for the selected assets or risk-free rate to perform analysis. Please ensure at least two valid weekly price entries and DGS10 values.")
                st.session_state['show_results'] = False
                # Removed st.stop()
            else:
                log_returns = np.log(currency_specific_data[asset_cols] / currency_specific_data[asset_cols].shift(1)).dropna()

                # Align the risk-free rate series with the log returns for consistent periods
                weekly_rf_series_aligned = currency_specific_data.loc[log_returns.index, 'DGS10']

                if weekly_rf_series_aligned.empty:
                    st.error("Not enough valid 'DGS10' risk-free rate data points aligned with asset returns to perform analysis.")
                    st.session_state['show_results'] = False
                    # Removed st.stop()
                elif len(log_returns) < 2:
                    st.error("Not enough return data to perform analysis (requires at least 2 data points after calculating log returns). This might happen if your data has many missing values or only one valid price entry.")
                    st.session_state['show_results'] = False
                    # Removed st.stop()
                else:
                    n_obs = len(log_returns)
                    mu = log_returns.mean()
                    cov_mat = log_returns.cov()

                    # Normalize weights to sum to 1 for calculation
                    user_weights = np.array([weights_input[asset] / 100.0 for asset in asset_cols])

                    # --- Calculate and Display Performance ---
                    performance = calculate_portfolio_performance(user_weights, mu, cov_mat, weekly_rf_series_aligned, n_obs)
                    optimal_performance, optimal_weights = optimize_portfolio(mu, cov_mat, weekly_rf_series_aligned, n_obs)

                    clean_asset_names = []
                    for asset in asset_cols:
                        display_name = asset.replace(f'_{selected_currency}', '')
                        if display_name == 'GSPC':
                            display_name = 'S&P 500'
                        clean_asset_names.append(display_name)

                    # Store results in session state
                    st.session_state['last_performance'] = performance
                    st.session_state['last_optimal_performance'] = optimal_performance
                    st.session_state['last_user_weights'] = user_weights
                    st.session_state['last_optimal_weights'] = optimal_weights
                    st.session_state['last_asset_names'] = clean_asset_names
                    st.session_state['show_results'] = True # Set flag to display results

    # Display results if the flag is set (i.e., after a successful 'Calculate' click)
    if st.session_state['show_results']:
        performance = st.session_state['last_performance']
        optimal_performance = st.session_state['last_optimal_performance']
        user_weights = st.session_state['last_user_weights']
        optimal_weights = st.session_state['last_optimal_weights']
        clean_asset_names = st.session_state['last_asset_names']

        st.header("Portfolio Performance Metrics")
        st.divider()

        # --- User's Portfolio Performance ---
        st.subheader("User-Defined Portfolio")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annualised Return", f"{performance['Return']:.2f}%")
        col2.metric("Annualised Volatility", f"{performance['Volatility']:.2f}%")
        col3.metric("Sharpe Ratio", f"{performance['SharpeRatio']:.2f}",
                                     help="Measures the risk-adjusted return of a portfolio. Higher is better.")
        lower_ci = performance['Return_95_CI_Lower']
        upper_ci = performance['Return_95_CI_Upper']
        col4.metric("Return 95% CI", f"{lower_ci:.2f}% - {upper_ci:.2f}%" if not np.isnan(lower_ci) else "N/A (Insufficient data or zero volatility)")

        st.divider()

        # --- Max Sharpe Ratio Portfolio ---
        st.subheader("Max Sharpe Ratio Portfolio (Optimal)")
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        opt_col1.metric("Optimal Annualised Return", f"{optimal_performance['Return']:.2f}%")
        opt_col2.metric("Optimal Annualised Volatility", f"{optimal_performance['Volatility']:.2f}%")
        opt_col3.metric("Optimal Sharpe Ratio", f"{optimal_performance['SharpeRatio']:.2f}",
                                         help="The portfolio with the highest risk-adjusted return.")
        opt_lower_ci = optimal_performance['Return_95_CI_Lower']
        opt_upper_ci = optimal_performance['Return_95_CI_Upper']
        opt_col4.metric("Optimal Return 95% CI", f"{opt_lower_ci:.2f}% - {opt_upper_ci:.2f}%" if not np.isnan(opt_lower_ci) else "N/A (Insufficient data or zero volatility)")

        st.divider()

        # --- Weights Comparison Chart ---
        st.header("Asset Weight Comparison")
        fig = plot_weights_comparison(user_weights, optimal_weights, clean_asset_names)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
