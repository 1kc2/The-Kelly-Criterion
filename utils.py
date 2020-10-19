import pandas as pd
import numpy as np


def get_kelly(returns, window=400, r=0.02, correlation=False):
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
        
    Returns
    -------
    kelly : pd.DataFrame
        Frame containing the corresponding kelly values for each security
    """
    days = 250
    r_adjusted = (1 + r) ** (1 / days) - 1
    mean = returns.expanding(window).mean()
    return_exces = mean - r_adjusted

    if correlation:
        roll_cov = returns.expanding(400).cov()
        roll_inv_cov = roll_cov.copy()
        kelly = pd.DataFrame(columns=mean.columns) 

        for day in returns.index:
            roll_inv_cov.loc[day] = np.linalg.inv(roll_cov.loc[day])
            kelly.loc[day] = np.dot(roll_inv_cov.loc[day], return_exces.loc[day])
            
    else:
        var = returns.expanding(window).var()
        kelly = return_exces / var
    
    kelly = kelly.dropna()
    
    return kelly


def filter_leverage(serie, leverage):
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
    filtered_serie[filtered_serie>leverage] = leverage
    
    return filtered_serie


def get_cumulative_returns(returns):
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

def backtest(kelly_df, returns_df, leverages):
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
        kelly_weights[kelly_weights<0] = 0

        daily_weights_sum = kelly_weights.sum(axis=1)
        leverage_cond = daily_weights_sum > leverage

        kelly_weights[leverage_cond] = leverage * kelly_weights[leverage_cond].div(daily_weights_sum[leverage_cond], axis=0)

        name = 'max_leverage_' + str(leverage) 
        total_returns[name] = (returns_df * kelly_weights).sum(axis=1)
        
    return total_returns