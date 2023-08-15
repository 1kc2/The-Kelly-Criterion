import pandas as pd
import numpy as np
from numba import njit, prange
from typing import Union, List


@njit(cache=True)
def get_kelly_numba(
    cov_arr: np.ndarray[float], returns: np.ndarray[float]
) -> np.ndarray[float]:
    """Compute kelly values using numba for speed

    Args:
        cov_arr (np.ndarray[float]): array of covariances without NaNs
        returns (np.ndarray[float]): array of returns without NaNs

    Returns:
        np.ndarray[float]: array of kelly values
    """
    ret_covars = np.empty_like(cov_arr)
    kelly = np.empty_like(returns)
    for i in prange(cov_arr.shape[0]):
        # Use copies to avoid issues with non-contiguous arrays
        inv_cov = np.linalg.inv(cov_arr[i])
        ret_day = returns[i].copy() 
        ret_covars[i] = inv_cov 
        kelly[i] = np.dot(inv_cov, ret_day)
    return kelly


def get_kelly_wrap(cov_arr: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Wrapper around numba function to build kelly DataFrame

    Args:
        cov_arr (pd.DataFrame): DataFrame of covariances
        returns (pd.DataFrame): DataFrame of excess returns

    Returns:
        pd.DataFrame: Kelly values
    """
   
    # Get dimensions to reshape the array
    dim0 = len(cov_arr.index.get_level_values(0).unique())
    dim1 = len(cov_arr.index.get_level_values(1).unique())
    dim3 = cov_arr.shape[1]
    
    # Get array of kelly values
    kelly = get_kelly_numba(
        cov_arr.values.reshape(dim0, dim1, dim3), returns.values
    )
    
    # Return DataFrame
    return pd.DataFrame(kelly, columns=returns.columns, index=returns.index)


def get_kelly(
    returns: Union[pd.DataFrame, pd.Series],
    window: int = 400,
    r: float = 0.02,
    correlation=False,
    days: int = 250,
) -> pd.DataFrame:
    """Gets Kelly optimal investment fraction

    Parameters
    ----------
    returns : pd.DataFrame or pd.Series
        Series containing the daily returns of a security
    window : int, optional
        Minimum periods to calculate the parameters. Default 400.
    r : int, optional
        Risk-free yearly returns. Example: Treasury bills. Default 0.02.
    correlation : bool, optional
        If a portfolio of securities is given, indicate whether the
        securities are correlationated or not.
    days : int, optional
        Number of days to use. Default 250

    Returns
    -------
    kelly : pd.DataFrame
        Frame containing the corresponding kelly values for each security
    """
    r_adjusted = (1 + r) ** (1 / days) - 1
    mean = returns.expanding(window).mean().dropna()
    return_exces = mean - r_adjusted

    if correlation:
        roll_cov = returns.expanding(window).cov().dropna()
        kelly = get_kelly_wrap(roll_cov, return_exces)
    else:
        var = returns.expanding(window).var().dropna()
        kelly = return_exces / var

    return kelly


def filter_leverage(serie: pd.Series, leverage: int) -> pd.Series:
    """filters leverage

    Parameters
    ----------
    serie : pd.Series
    leverage : int

    Returns
    -------
    filtered_serie : pd.Series
    """

    filtered_serie = serie.copy()
    filtered_serie[filtered_serie > leverage] = leverage

    return filtered_serie


def get_cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """Gets cumulative returns

    Parameters
    ----------
    returns : pd.Series, pd.DataFrame

    Returns
    -------
    cum_returns : pd.Series, pd.DataFrame
    """

    cum_returns = (1 + returns).cumprod()
    cum_returns = cum_returns.dropna()

    return cum_returns


def backtest(
    kelly_df: pd.DataFrame, returns_df: pd.DataFrame, leverages: List[int]
) -> pd.DataFrame:
    """Backtests Kelly strategy

    Parameters
    ----------
    kelly_df : pd.DataFrame
        kelly optimal allocations for the securities
    returns_df : pd.DataFrame
        daily returns of the securities
    leverages : list
        list containing the number of leverages to study

    Returns
    -------
    total_returns : pd.DataFrame
    """

    total_returns = pd.DataFrame()

    for leverage in leverages:
        kelly_weights = kelly_df.copy()

        # restrict shortselling
        kelly_weights[kelly_weights < 0] = 0

        daily_weights_sum = kelly_weights.sum(axis=1)
        leverage_cond = daily_weights_sum > leverage

        kelly_weights[leverage_cond] = leverage * kelly_weights[leverage_cond].div(
            daily_weights_sum[leverage_cond], axis=0
        )

        name = "max_leverage_" + str(leverage)
        total_returns[name] = (returns_df * kelly_weights).sum(axis=1)

    return total_returns
