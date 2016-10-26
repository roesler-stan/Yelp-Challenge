import pandas as pd
import statsmodels.iolib.summary2 as s2

""" Change the stars definition to be 0.05, 0.01, 0.001 """
def _col_params(result, float_format='%.4f', stars=True):
    '''Stack coefficients and standard errors in single column'''
    # Extract parameters
    res = s2.summary_params(result)
    # Format float
    for col in res.columns[:2]:
        res[col] = res[col].apply(lambda x: float_format % x)
    # Std.Errors in parentheses
    res.ix[:, 1] = '(' + res.ix[:, 1] + ')'
    # Significance stars
    if stars:
        idx = res.ix[:, 3] < .05
        res.ix[:, 0][idx] = res.ix[:, 0][idx] + '*'
        idx = res.ix[:, 3] < .01
        res.ix[:, 0][idx] = res.ix[:, 0][idx] + '*'
        idx = res.ix[:, 3] < .001
        res.ix[:, 0][idx] = res.ix[:, 0][idx] + '*'
    # Stack Coefs and Std.Errors
    res = res.ix[:, :2]
    res = res.stack()
    res = pd.DataFrame(res)
    res.columns = [str(result.model.endog_names)]
    return res
 
s2._col_params = _col_params