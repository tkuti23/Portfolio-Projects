

```python
import numpy as np, pandas as pd
import statistics as st
import statsmodels.api as sm
import RamiFunctions as RF, statsmodels.formula.api as smf
# from yahoofinancials import YahooFinancials
import statsmodels.tsa.api as smt

import scipy.stats as ss
import scipy.stats.mstats as st1
from scipy import stats  
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import statsmodels.stats._adnorm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white

from statsmodels.stats.diagnostic import acorr_breusch_godfrey

import warnings
warnings.filterwarnings('ignore')



# Applying Chow test for Market vs Stock
def get_rss(y, x):
    '''
    Inputs:
    formula_model: Is the regression equation that you are running
    data: The DataFrame of the independent and dependent variable

    Outputs: 
    rss: the sum of residuals
    N: the observations of inputs
    K: total number of parameters
    '''
    x = sm.add_constant(x)
    results=sm.OLS(y, x).fit()
    rss= (results.resid**2).sum()
    N=results.nobs
    K=results.df_model
    return rss, N, K, results


def Chow_Test(df, y, x, special_date, level_of_sig=0.05):
    
    from scipy.stats import f
    date=special_date
    x1=df[x][:date]
    y1=df[y][:date]
    x2=df[x][date:]
    y2=df[y][date:]

    RSS_total, N_total, K_total, results=get_rss(df[y], df[x])
    RSS_1, N_1, K_1, results1=get_rss(y1, x1)
    RSS_2, N_2, K_2, results2=get_rss(y2, x2)
    num=(RSS_total-RSS_1-RSS_2)/K_total
    den=(RSS_1+RSS_2)/(N_1+N_2-2*K_total)

    p_val = f.sf(num/den, 2, N_1+N_2-2*K_total)
    

    df['Before_Special'] = np.where(df.index<special_date , 'Before', 'After')
    g = sns.lmplot(x=x, y=y, hue="Before_Special", truncate=True, height=5, markers=["o", "x"], data=df)
    # return df

    if p_val<level_of_sig:
        print('The P vale {:3.5f} is lower than the level of significance {}. Therefore, reject the null that the coefficients are the same in the two periods are equal'.format(p_val,level_of_sig))
    else:
        print('The P vale {:3.5f} is higher than the level of significance {}. Therefore, accept the null that the coefficients are the same in the two periods are equal'.format(p_val,level_of_sig))

    return num/den, p_val

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        ss.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


def Naive_Forecast(df_in_sample, df_out_sample, column):
    a=df_in_sample.tail(1).values
    df_naive_method_forecast=pd.DataFrame(np.repeat(a, df_out_sample.shape[0], axis=0))
    df_naive_method_forecast.columns=df_out_sample.columns
    # df_naive_method_forecast.head(3)

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_naive_method_forecast[column], label='Naive Forecast Data')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.show()
    df_naive_method_forecast.index=df_out_sample.index

    return df_naive_method_forecast


def Average_Forecast(df_in_sample, df_out_sample, column):
    a=df_in_sample.mean().values
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Average_Forecast=pd.DataFrame(a)
    df_Average_Forecast.columns=df_out_sample.columns
    df_Average_Forecast.index=df_out_sample.index

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Average_Forecast[column], label='Simple Average Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Average Forecast")
    plt.show()
    df_Average_Forecast.index=df_out_sample.index

    return df_Average_Forecast


def Moving_Average_Forecast(df_in_sample, df_out_sample, column, window_leng):
    a=df_in_sample.rolling(window_leng).mean().iloc[-1].values
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Moving_Average_Forecast=pd.DataFrame(a)
    df_Moving_Average_Forecast.columns=df_out_sample.columns
    df_Moving_Average_Forecast.index=df_out_sample.index

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Moving_Average_Forecast[column], label='Moving Average Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Moving Average Forecast")
    plt.show()

    df_Moving_Average_Forecast.index=df_out_sample.index

    return df_Moving_Average_Forecast


def Simple_Exponential_Smoothing_Forecast(df_in_sample, df_out_sample, column, level):
    a=[]
    for col in df_in_sample.columns:
        fit2 =smt.SimpleExpSmoothing(np.asarray(df_in_sample[col])).fit(smoothing_level=level, optimized=False)
        a.append(fit2.forecast())
    a=np.array(a)
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Simple_Exponential_Smoothing_Forecast=pd.DataFrame(a)
    df_Simple_Exponential_Smoothing_Forecast.columns=df_out_sample.columns
    df_Simple_Exponential_Smoothing_Forecast.index=df_out_sample.index
    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Simple_Exponential_Smoothing_Forecast[column], label='Simple Exponential Smoothing Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Exponential Smoothing Forecast")
    plt.show()
    df_Simple_Exponential_Smoothing_Forecast.index=df_out_sample.index

    return df_Simple_Exponential_Smoothing_Forecast


# Here we are conducting a one tail test by speecifing if the alternative is "two-sided", "larger", or "smaller"

# def ttest_OLS(res, numberofbeta, X, value=0, alternative='two-sided', level_of_sig = 0.05):
#     results=np.zeros([2])
#     # numberofbeta represent the coeffiecent you would like to test 0 standts for interecept
#     results[0]=res.tvalues[numberofbeta]
#     results[1]=res.pvalues[numberofbeta]
#     if isinstance(X, pd.DataFrame):
#         column=X.columns[numberofbeta]
#     else:
#         column=numberofbeta
#     if alternative == 'two-sided':
#         if results[1]<level_of_sig:
#             print("We reject the null hypothesis that the Selected Coefficient: {} is equal to {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is equal to {} with a {} % significance level".format(column, value, level_of_sig*100))
#     elif alternative == 'larger':
#         if (results[0] > 0) & (results[1]/2 < level_of_sig):
#             print("We reject the null hypothesis that the Selected Coefficient: {} is less than {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is less than {} with a {} % significance level".format(column, value, level_of_sig*100))

#     elif alternative == 'smaller':
#         if (results[0] < 0) & (results[1]/2 < level_of_sig):
#             print("We reject the null hypothesis that the Selected Coefficient: {} is more than {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is more than {} with a {} % significance level".format(column, value, level_of_sig*100))

def Simple_ttest_Ols(results, hypothesis, alternative='two-sided', level_of_sig = 0.05):
    results1=np.zeros([2])
    t_test = results.t_test(hypothesis)
    results1[0]=t_test.tvalue
    results1[1]=t_test.pvalue
    if alternative == 'two-sided':
        if results1[1]<level_of_sig:
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
    elif alternative == 'larger':
        if (results1[0] > 0) & (results1[1]/2 < level_of_sig):
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))

    elif alternative == 'smaller':
        if (results1[0] < 0) & (results1[1]/2 < level_of_sig):
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))

    pass



def Joining_columns(df1, x, y=None,Name_of_new_column=None):
    # Find all columns except x
    f=df1.columns.to_list()
    f.remove(x)
    if y!=None:
        df2=pd.melt(df1, id_vars=[x], value_vars=y)  
    else:
        df2=pd.melt(df1, id_vars=[x], value_vars=f)
    if Name_of_new_column!=None:
        df2=df2.rename(columns={"value": Name_of_new_column})
    return df2



def get_betas_SLR(df1, x, column=None):
    if column==None:
    # Choose all the columns except the x
        f=df1.columns.to_list()
        f.remove(x)
        column=f
    A=np.zeros([len(column)])
    j=0
    for i in column:
        formula ='Q("'+ i+'"'+') ~ Q("Excess Market Returns")'
        results = smf.ols(formula, df1).fit()
        A[j]=results.params[1]
        j=j+1
        
    A=pd.DataFrame(data=A,columns=['Beta'], index=column)
    return A


def Get_indicators(BB, indicat):
    # BB represents the BB=yahoo_financials.get_key_statistics_data()
    V=np.zeros([len(BB)])
    j=0
    for i in BB.keys():
        V[j]=BB[i][indicat]
        j=j+1
    return V

# Examples V=Get_indicators(BB, 'priceToBook')


# Get Data from a dictionary downloaded from yahoo finance
def Get_Dataframe_of_tickes(tickers):
    yahoo_financials = YahooFinancials(tickers)
    BB=yahoo_financials.get_key_statistics_data()
    dict_of_df = {k: pd.DataFrame.from_dict(v, orient='index') for k,v in BB.items()}
    df = pd.concat(dict_of_df, axis=1)
    return df

# Examples
# tickers = ['AAPL', 'WFC', 'F', 'FB', 'DELL', 'SNE','NOK', 'MSFT', 'JPM', 'GE', 'BAC']
# Name=['Apple', 'Wells_Fargo_Company', 'Ford Motor Company', 'Facebook', 'Dell Technologies', 'Sony', 'Nokia', 'Microsoft', 'JPMorgan Chase & Co', 'General Electric', 'Bank of America']
# df=RF.Get_Dataframe_of_tickes(tickers)


def Get_Yahoo_stats(tickers):
    yahoo_financials = YahooFinancials(tickers)
    f=['get_interest_expense()', 'get_operating_income()', 'get_total_operating_expense()', 'get_total_revenue()', 'get_cost_of_revenue()', 'get_income_before_tax()', 'get_income_tax_expense()', 'get_gross_profit()', 'get_net_income_from_continuing_ops()', 'get_research_and_development()', 'get_current_price()', 'get_current_change()', 'get_current_percent_change()', 'get_current_volume()', 'get_prev_close_price()', 'get_open_price()', 'get_ten_day_avg_daily_volume()', 'get_three_month_avg_daily_volume()', 'get_stock_exchange()', 'get_market_cap()', 'get_daily_low()', 'get_daily_high()', 'get_currency()', 'get_yearly_high()', 'get_yearly_low()', 'get_dividend_yield()', 'get_annual_avg_div_yield()', 'get_five_yr_avg_div_yield()', 'get_dividend_rate()', 'get_annual_avg_div_rate()', 'get_50day_moving_avg()', 'get_200day_moving_avg()', 'get_beta()', 'get_payout_ratio()', 'get_pe_ratio()', 'get_price_to_sales()', 'get_exdividend_date()', 'get_book_value()', 'get_ebit()', 'get_net_income()', 'get_earnings_per_share()', 'get_key_statistics_data()']
    i=0
    exec('d=yahoo_financials.'+f[i], locals(), globals())
    col=f[i].replace("get_","").replace("()","")
    A=pd.DataFrame.from_dict(d, orient='index', columns=[col])
    for i in range(1,3):
        exec('d=yahoo_financials.'+f[i], locals(), globals())
        col=f[i].replace("get_","").replace("()","").replace("_"," ")
        B=pd.DataFrame.from_dict(d, orient='index', columns=[col])
        A= pd.concat([A, B], axis=1, sort=False)
    return A    

# from yahoofinancials import YahooFinancials
# tickers = ['AAPL', 'WFC', 'F', 'FB', 'DELL', 'SNE']
# Get_Yahoo_stats(tickers)

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

# Examples

# data = sm.datasets.longley.load_pandas()
# df1=data.data
# formula = 'GNP ~ YEAR + UNEMP + POP + GNPDEFL'
# results = smf.ols(formula, df1).fit()
# print(results.summary())
# res = RF.forward_selected(df1, 'GNP')
# print(res.model.formula)
# print(res.rsquared_adj)
# print(res.summary())


def GQTest(lm2, level_of_sig=0.05, sp=None):
    name = ['F statistic', 'p-value']
    test = sms.het_goldfeldquandt(lm2.resid, lm2.model.exog, split=sp)
    R=lzip(name, test)

    if test[1]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are homoscedastic'.format(test[1], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are hetroscedastic'.format(test[1], level_of_sig))
    return R


def WhiteTest(statecrime_model, level_of_sig=0.05):
    white_test = het_white(statecrime_model.resid,  statecrime_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    R=dict(zip(labels, white_test))

    if white_test[3]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are homoscedastic'.format(white_test[3], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are hetroscedastic'.format(white_test[3], level_of_sig))
    return R


def Plot_resi_corr(results):
    res_min_1=results.resid[:-1]
    res_plus_1=results.resid[1:]
    data1=pd.DataFrame(np.column_stack((res_min_1.T,res_plus_1.T)), columns=['u_t-1','u_t'])
    sns.set()
    plt.figure(figsize=(5,5))
    ax = sns.scatterplot(x='u_t-1', y='u_t', data=data1)
    pass

def Plot_resi_corr_time(results,df):
    C=pd.DataFrame(results.resid, index=df.index, columns=['Residuals'])
    C.plot(figsize=(10,5), linewidth=1.5, fontsize=10)
    plt.xlabel('Date', fontsize=10);
    return C


def Breusch_Godfrey(results, level_of_sig=0.05, lags=None):
    A=acorr_breusch_godfrey(results, nlags=lags)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    R=dict(zip(labels, A))

    if A[3]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are not Auto-corrolated'.format(A[3], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are Auto-corrolated'.format(A[3], level_of_sig))
    return R

def Create_lags_of_variable(MainDF_first_period, lags, column):
    # Crete a new dataframe based on the lag variables
    x=column
    if type(lags) == int:
        j=lags
        values=MainDF_first_period[x]
        dataframe = pd.concat([values.shift(j), values], axis=1)
        dataframe.columns = [x+' at time t-'+str(j), x+' at time t']
        dataframe=dataframe.dropna()
    else:
        values=MainDF_first_period[x]
        dataframe=values
        for j in lags:
            dataframe = pd.concat([values.shift(j), dataframe], axis=1)
        c=[x+' for time t-'+str(j) for j in range(len(lags),-1,-1)]
        dataframe.columns=c
        dataframe=dataframe.dropna()
    return dataframe

# # Things students shouldn't know
# MainDF1=MainDF.reset_index(drop=False)
# df2=pd.melt(MainDF1, id_vars=MainDF.index.name)
# palette = dict(zip(df2['variable'].unique(), sns.color_palette("rocket_r", len(df2['variable'].unique()))))
# sns.relplot(x=MainDF.index.name, y="value",
#             hue="variable", palette=palette,
#             height=5, aspect=3, facet_kws=dict(sharex=False), kind="line", data=df2)import numpy as np, pandas as pd
import statistics as st
import statsmodels.api as sm
import RamiFunctions as RF, statsmodels.formula.api as smf
# from yahoofinancials import YahooFinancials
import statsmodels.tsa.api as smt

import scipy.stats as ss
import scipy.stats.mstats as st1
from scipy import stats  
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import statsmodels.stats._adnorm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white

from statsmodels.stats.diagnostic import acorr_breusch_godfrey

import warnings
warnings.filterwarnings('ignore')



# Applying Chow test for Market vs Stock
def get_rss(y, x):
    '''
    Inputs:
    formula_model: Is the regression equation that you are running
    data: The DataFrame of the independent and dependent variable

    Outputs: 
    rss: the sum of residuals
    N: the observations of inputs
    K: total number of parameters
    '''
    x = sm.add_constant(x)
    results=sm.OLS(y, x).fit()
    rss= (results.resid**2).sum()
    N=results.nobs
    K=results.df_model
    return rss, N, K, results


def Chow_Test(df, y, x, special_date, level_of_sig=0.05):
    
    from scipy.stats import f
    date=special_date
    x1=df[x][:date]
    y1=df[y][:date]
    x2=df[x][date:]
    y2=df[y][date:]

    RSS_total, N_total, K_total, results=get_rss(df[y], df[x])
    RSS_1, N_1, K_1, results1=get_rss(y1, x1)
    RSS_2, N_2, K_2, results2=get_rss(y2, x2)
    num=(RSS_total-RSS_1-RSS_2)/K_total
    den=(RSS_1+RSS_2)/(N_1+N_2-2*K_total)

    p_val = f.sf(num/den, 2, N_1+N_2-2*K_total)
    

    df['Before_Special'] = np.where(df.index<special_date , 'Before', 'After')
    g = sns.lmplot(x=x, y=y, hue="Before_Special", truncate=True, height=5, markers=["o", "x"], data=df)
    # return df

    if p_val<level_of_sig:
        print('The P vale {:3.5f} is lower than the level of significance {}. Therefore, reject the null that the coefficients are the same in the two periods are equal'.format(p_val,level_of_sig))
    else:
        print('The P vale {:3.5f} is higher than the level of significance {}. Therefore, accept the null that the coefficients are the same in the two periods are equal'.format(p_val,level_of_sig))

    return num/den, p_val

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        ss.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


def Naive_Forecast(df_in_sample, df_out_sample, column):
    a=df_in_sample.tail(1).values
    df_naive_method_forecast=pd.DataFrame(np.repeat(a, df_out_sample.shape[0], axis=0))
    df_naive_method_forecast.columns=df_out_sample.columns
    # df_naive_method_forecast.head(3)

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_naive_method_forecast[column], label='Naive Forecast Data')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.show()
    df_naive_method_forecast.index=df_out_sample.index

    return df_naive_method_forecast


def Average_Forecast(df_in_sample, df_out_sample, column):
    a=df_in_sample.mean().values
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Average_Forecast=pd.DataFrame(a)
    df_Average_Forecast.columns=df_out_sample.columns
    df_Average_Forecast.index=df_out_sample.index

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Average_Forecast[column], label='Simple Average Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Average Forecast")
    plt.show()
    df_Average_Forecast.index=df_out_sample.index

    return df_Average_Forecast


def Moving_Average_Forecast(df_in_sample, df_out_sample, column, window_leng):
    a=df_in_sample.rolling(window_leng).mean().iloc[-1].values
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Moving_Average_Forecast=pd.DataFrame(a)
    df_Moving_Average_Forecast.columns=df_out_sample.columns
    df_Moving_Average_Forecast.index=df_out_sample.index

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Moving_Average_Forecast[column], label='Moving Average Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Moving Average Forecast")
    plt.show()

    df_Moving_Average_Forecast.index=df_out_sample.index

    return df_Moving_Average_Forecast


def Simple_Exponential_Smoothing_Forecast(df_in_sample, df_out_sample, column, level):
    a=[]
    for col in df_in_sample.columns:
        fit2 =smt.SimpleExpSmoothing(np.asarray(df_in_sample[col])).fit(smoothing_level=level, optimized=False)
        a.append(fit2.forecast())
    a=np.array(a)
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Simple_Exponential_Smoothing_Forecast=pd.DataFrame(a)
    df_Simple_Exponential_Smoothing_Forecast.columns=df_out_sample.columns
    df_Simple_Exponential_Smoothing_Forecast.index=df_out_sample.index
    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Simple_Exponential_Smoothing_Forecast[column], label='Simple Exponential Smoothing Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Exponential Smoothing Forecast")
    plt.show()
    df_Simple_Exponential_Smoothing_Forecast.index=df_out_sample.index

    return df_Simple_Exponential_Smoothing_Forecast


# Here we are conducting a one tail test by speecifing if the alternative is "two-sided", "larger", or "smaller"

# def ttest_OLS(res, numberofbeta, X, value=0, alternative='two-sided', level_of_sig = 0.05):
#     results=np.zeros([2])
#     # numberofbeta represent the coeffiecent you would like to test 0 standts for interecept
#     results[0]=res.tvalues[numberofbeta]
#     results[1]=res.pvalues[numberofbeta]
#     if isinstance(X, pd.DataFrame):
#         column=X.columns[numberofbeta]
#     else:
#         column=numberofbeta
#     if alternative == 'two-sided':
#         if results[1]<level_of_sig:
#             print("We reject the null hypothesis that the Selected Coefficient: {} is equal to {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is equal to {} with a {} % significance level".format(column, value, level_of_sig*100))
#     elif alternative == 'larger':
#         if (results[0] > 0) & (results[1]/2 < level_of_sig):
#             print("We reject the null hypothesis that the Selected Coefficient: {} is less than {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is less than {} with a {} % significance level".format(column, value, level_of_sig*100))

#     elif alternative == 'smaller':
#         if (results[0] < 0) & (results[1]/2 < level_of_sig):
#             print("We reject the null hypothesis that the Selected Coefficient: {} is more than {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is more than {} with a {} % significance level".format(column, value, level_of_sig*100))

def Simple_ttest_Ols(results, hypothesis, alternative='two-sided', level_of_sig = 0.05):
    results1=np.zeros([2])
    t_test = results.t_test(hypothesis)
    results1[0]=t_test.tvalue
    results1[1]=t_test.pvalue
    if alternative == 'two-sided':
        if results1[1]<level_of_sig:
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
    elif alternative == 'larger':
        if (results1[0] > 0) & (results1[1]/2 < level_of_sig):
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))

    elif alternative == 'smaller':
        if (results1[0] < 0) & (results1[1]/2 < level_of_sig):
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))

    pass



def Joining_columns(df1, x, y=None,Name_of_new_column=None):
    # Find all columns except x
    f=df1.columns.to_list()
    f.remove(x)
    if y!=None:
        df2=pd.melt(df1, id_vars=[x], value_vars=y)  
    else:
        df2=pd.melt(df1, id_vars=[x], value_vars=f)
    if Name_of_new_column!=None:
        df2=df2.rename(columns={"value": Name_of_new_column})
    return df2



def get_betas_SLR(df1, x, column=None):
    if column==None:
    # Choose all the columns except the x
        f=df1.columns.to_list()
        f.remove(x)
        column=f
    A=np.zeros([len(column)])
    j=0
    for i in column:
        formula ='Q("'+ i+'"'+') ~ Q("Excess Market Returns")'
        results = smf.ols(formula, df1).fit()
        A[j]=results.params[1]
        j=j+1
        
    A=pd.DataFrame(data=A,columns=['Beta'], index=column)
    return A


def Get_indicators(BB, indicat):
    # BB represents the BB=yahoo_financials.get_key_statistics_data()
    V=np.zeros([len(BB)])
    j=0
    for i in BB.keys():
        V[j]=BB[i][indicat]
        j=j+1
    return V

# Examples V=Get_indicators(BB, 'priceToBook')


# Get Data from a dictionary downloaded from yahoo finance
def Get_Dataframe_of_tickes(tickers):
    yahoo_financials = YahooFinancials(tickers)
    BB=yahoo_financials.get_key_statistics_data()
    dict_of_df = {k: pd.DataFrame.from_dict(v, orient='index') for k,v in BB.items()}
    df = pd.concat(dict_of_df, axis=1)
    return df

# Examples
# tickers = ['AAPL', 'WFC', 'F', 'FB', 'DELL', 'SNE','NOK', 'MSFT', 'JPM', 'GE', 'BAC']
# Name=['Apple', 'Wells_Fargo_Company', 'Ford Motor Company', 'Facebook', 'Dell Technologies', 'Sony', 'Nokia', 'Microsoft', 'JPMorgan Chase & Co', 'General Electric', 'Bank of America']
# df=RF.Get_Dataframe_of_tickes(tickers)


def Get_Yahoo_stats(tickers):
    yahoo_financials = YahooFinancials(tickers)
    f=['get_interest_expense()', 'get_operating_income()', 'get_total_operating_expense()', 'get_total_revenue()', 'get_cost_of_revenue()', 'get_income_before_tax()', 'get_income_tax_expense()', 'get_gross_profit()', 'get_net_income_from_continuing_ops()', 'get_research_and_development()', 'get_current_price()', 'get_current_change()', 'get_current_percent_change()', 'get_current_volume()', 'get_prev_close_price()', 'get_open_price()', 'get_ten_day_avg_daily_volume()', 'get_three_month_avg_daily_volume()', 'get_stock_exchange()', 'get_market_cap()', 'get_daily_low()', 'get_daily_high()', 'get_currency()', 'get_yearly_high()', 'get_yearly_low()', 'get_dividend_yield()', 'get_annual_avg_div_yield()', 'get_five_yr_avg_div_yield()', 'get_dividend_rate()', 'get_annual_avg_div_rate()', 'get_50day_moving_avg()', 'get_200day_moving_avg()', 'get_beta()', 'get_payout_ratio()', 'get_pe_ratio()', 'get_price_to_sales()', 'get_exdividend_date()', 'get_book_value()', 'get_ebit()', 'get_net_income()', 'get_earnings_per_share()', 'get_key_statistics_data()']
    i=0
    exec('d=yahoo_financials.'+f[i], locals(), globals())
    col=f[i].replace("get_","").replace("()","")
    A=pd.DataFrame.from_dict(d, orient='index', columns=[col])
    for i in range(1,3):
        exec('d=yahoo_financials.'+f[i], locals(), globals())
        col=f[i].replace("get_","").replace("()","").replace("_"," ")
        B=pd.DataFrame.from_dict(d, orient='index', columns=[col])
        A= pd.concat([A, B], axis=1, sort=False)
    return A    

# from yahoofinancials import YahooFinancials
# tickers = ['AAPL', 'WFC', 'F', 'FB', 'DELL', 'SNE']
# Get_Yahoo_stats(tickers)

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

# Examples

# data = sm.datasets.longley.load_pandas()
# df1=data.data
# formula = 'GNP ~ YEAR + UNEMP + POP + GNPDEFL'
# results = smf.ols(formula, df1).fit()
# print(results.summary())
# res = RF.forward_selected(df1, 'GNP')
# print(res.model.formula)
# print(res.rsquared_adj)
# print(res.summary())


def GQTest(lm2, level_of_sig=0.05, sp=None):
    name = ['F statistic', 'p-value']
    test = sms.het_goldfeldquandt(lm2.resid, lm2.model.exog, split=sp)
    R=lzip(name, test)

    if test[1]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are homoscedastic'.format(test[1], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are hetroscedastic'.format(test[1], level_of_sig))
    return R


def WhiteTest(statecrime_model, level_of_sig=0.05):
    white_test = het_white(statecrime_model.resid,  statecrime_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    R=dict(zip(labels, white_test))

    if white_test[3]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are homoscedastic'.format(white_test[3], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are hetroscedastic'.format(white_test[3], level_of_sig))
    return R


def Plot_resi_corr(results):
    res_min_1=results.resid[:-1]
    res_plus_1=results.resid[1:]
    data1=pd.DataFrame(np.column_stack((res_min_1.T,res_plus_1.T)), columns=['u_t-1','u_t'])
    sns.set()
    plt.figure(figsize=(5,5))
    ax = sns.scatterplot(x='u_t-1', y='u_t', data=data1)
    pass

def Plot_resi_corr_time(results,df):
    C=pd.DataFrame(results.resid, index=df.index, columns=['Residuals'])
    C.plot(figsize=(10,5), linewidth=1.5, fontsize=10)
    plt.xlabel('Date', fontsize=10);
    return C


def Breusch_Godfrey(results, level_of_sig=0.05, lags=None):
    A=acorr_breusch_godfrey(results, nlags=lags)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    R=dict(zip(labels, A))

    if A[3]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are not Auto-corrolated'.format(A[3], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are Auto-corrolated'.format(A[3], level_of_sig))
    return R

def Create_lags_of_variable(MainDF_first_period, lags, column):
    # Crete a new dataframe based on the lag variables
    x=column
    if type(lags) == int:
        j=lags
        values=MainDF_first_period[x]
        dataframe = pd.concat([values.shift(j), values], axis=1)
        dataframe.columns = [x+' at time t-'+str(j), x+' at time t']
        dataframe=dataframe.dropna()
    else:
        values=MainDF_first_period[x]
        dataframe=values
        for j in lags:
            dataframe = pd.concat([values.shift(j), dataframe], axis=1)
        c=[x+' for time t-'+str(j) for j in range(len(lags),-1,-1)]
        dataframe.columns=c
        dataframe=dataframe.dropna()
    return dataframe

# # Things students shouldn't know
# MainDF1=MainDF.reset_index(drop=False)
# df2=pd.melt(MainDF1, id_vars=MainDF.index.name)
# palette = dict(zip(df2['variable'].unique(), sns.color_palette("rocket_r", len(df2['variable'].unique()))))
# sns.relplot(x=MainDF.index.name, y="value",
#             hue="variable", palette=palette,
#             height=5, aspect=3, facet_kws=dict(sharex=False), kind="line", data=df2)

# Call the important packages I want to use
import numpy as np, pandas as pd
import statistics as st
import statsmodels.api as sm
import RamiFunctions as RF, statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.stats.api as sms
import statsmodels.graphics.tsaplots as smgtsplot

import scipy.stats as ss
import scipy.stats.mstats as st1
from scipy import stats  
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import statsmodels.stats._adnorm

import warnings
warnings.filterwarnings('ignore')
```


```python

```


```python
df =pd.read_excel('Data_For_Analysis.xlsx')
df.head(80)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1976-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1976-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1976-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1976-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1976-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1976-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1976-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1976-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1977-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1977-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1977-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1977-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1977-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1977-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1977-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1977-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1977-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1977-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1977-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1977-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1978-01-01</td>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1978-02-01</td>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1978-03-01</td>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1978-04-01</td>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1978-05-01</td>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1978-06-01</td>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1980-03-01</td>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1980-04-01</td>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1980-05-01</td>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1980-06-01</td>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1980-07-01</td>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1980-08-01</td>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1980-09-01</td>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1980-10-01</td>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1980-11-01</td>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1980-12-01</td>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1981-01-01</td>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1981-02-01</td>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1981-03-01</td>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1981-04-01</td>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1981-05-01</td>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1981-06-01</td>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1981-07-01</td>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>67</th>
      <td>1981-08-01</td>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1981-09-01</td>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1981-10-01</td>
      <td>0.016</td>
      <td>0.052</td>
      <td>0.112</td>
      <td>0.049</td>
      <td>0.094</td>
      <td>0.102</td>
      <td>-0.072</td>
      <td>0.067</td>
      <td>-0.012</td>
      <td>...</td>
      <td>0.008</td>
      <td>-0.020</td>
      <td>31.00</td>
      <td>149.1</td>
      <td>279.9</td>
      <td>0.10256</td>
      <td>0.06701</td>
      <td>0.06230</td>
      <td>-0.01430</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1981-11-01</td>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1981-12-01</td>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1982-01-01</td>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1982-02-01</td>
      <td>-0.159</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.175</td>
      <td>-0.152</td>
      <td>-0.049</td>
      <td>0.145</td>
      <td>0.098</td>
      <td>-0.111</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>29.76</td>
      <td>142.9</td>
      <td>283.4</td>
      <td>-0.17021</td>
      <td>-0.10825</td>
      <td>0.00276</td>
      <td>-0.02619</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1982-03-01</td>
      <td>0.108</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>-0.017</td>
      <td>-0.302</td>
      <td>-0.104</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.136</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.009</td>
      <td>28.31</td>
      <td>141.7</td>
      <td>283.1</td>
      <td>-0.05128</td>
      <td>0.09595</td>
      <td>-0.06643</td>
      <td>-0.11714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1982-04-01</td>
      <td>-0.009</td>
      <td>0.101</td>
      <td>0.050</td>
      <td>-0.013</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>-0.025</td>
      <td>0.076</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.008</td>
      <td>0.059</td>
      <td>27.65</td>
      <td>140.2</td>
      <td>284.3</td>
      <td>0.13514</td>
      <td>-0.02151</td>
      <td>0.05993</td>
      <td>0.06141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1982-05-01</td>
      <td>-0.189</td>
      <td>-0.101</td>
      <td>0.016</td>
      <td>-0.091</td>
      <td>-0.180</td>
      <td>-0.056</td>
      <td>0.042</td>
      <td>-0.027</td>
      <td>0.043</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.086</td>
      <td>27.67</td>
      <td>139.2</td>
      <td>287.1</td>
      <td>-0.04762</td>
      <td>-0.05495</td>
      <td>-0.02898</td>
      <td>-0.04610</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1982-06-01</td>
      <td>-0.044</td>
      <td>-0.003</td>
      <td>-0.024</td>
      <td>-0.096</td>
      <td>-0.060</td>
      <td>-0.073</td>
      <td>0.106</td>
      <td>0.050</td>
      <td>-0.033</td>
      <td>...</td>
      <td>-0.017</td>
      <td>-0.015</td>
      <td>28.11</td>
      <td>138.7</td>
      <td>290.6</td>
      <td>-0.02500</td>
      <td>-0.01395</td>
      <td>-0.02222</td>
      <td>-0.05799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1982-07-01</td>
      <td>0.006</td>
      <td>-0.025</td>
      <td>-0.032</td>
      <td>-0.303</td>
      <td>-0.054</td>
      <td>-0.055</td>
      <td>-0.118</td>
      <td>0.038</td>
      <td>0.019</td>
      <td>...</td>
      <td>-0.060</td>
      <td>-0.012</td>
      <td>28.33</td>
      <td>138.8</td>
      <td>292.6</td>
      <td>0.10256</td>
      <td>-0.02410</td>
      <td>-0.08333</td>
      <td>0.07975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1982-08-01</td>
      <td>0.379</td>
      <td>0.077</td>
      <td>0.133</td>
      <td>0.070</td>
      <td>0.216</td>
      <td>0.273</td>
      <td>0.055</td>
      <td>0.032</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.221</td>
      <td>28.18</td>
      <td>138.4</td>
      <td>292.8</td>
      <td>0.09302</td>
      <td>0.22222</td>
      <td>0.17273</td>
      <td>0.07607</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 28 columns</p>
</div>




```python
df.set_index('Date', inplace=True)
```


```python
df.head(300)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1976-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>1976-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>1976-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>1976-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>1976-06-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>1976-07-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>1976-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>1976-09-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>1976-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>1976-11-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>1976-12-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>1977-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>1977-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>1977-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>1977-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>1977-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>1977-06-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>1977-07-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>1977-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>1977-09-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>1977-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>1977-11-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>1977-12-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.086</td>
      <td>0.043</td>
      <td>0.046</td>
      <td>-0.050</td>
      <td>0.060</td>
      <td>-0.101</td>
      <td>0.021</td>
      <td>0.060</td>
      <td>-0.031</td>
      <td>-0.038</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.004</td>
      <td>24.03</td>
      <td>166.5</td>
      <td>322.3</td>
      <td>0.00893</td>
      <td>0.06471</td>
      <td>-0.03727</td>
      <td>-0.01393</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>-0.026</td>
      <td>-0.030</td>
      <td>-0.084</td>
      <td>0.018</td>
      <td>0.043</td>
      <td>0.080</td>
      <td>0.008</td>
      <td>-0.099</td>
      <td>-0.036</td>
      <td>0.062</td>
      <td>...</td>
      <td>-0.030</td>
      <td>0.020</td>
      <td>24.00</td>
      <td>166.2</td>
      <td>322.8</td>
      <td>-0.06195</td>
      <td>0.03147</td>
      <td>0.03011</td>
      <td>-0.00816</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.011</td>
      <td>-0.063</td>
      <td>0.043</td>
      <td>-0.052</td>
      <td>-0.006</td>
      <td>0.032</td>
      <td>-0.066</td>
      <td>0.002</td>
      <td>0.025</td>
      <td>-0.028</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.013</td>
      <td>23.92</td>
      <td>167.7</td>
      <td>323.5</td>
      <td>0.08491</td>
      <td>-0.02712</td>
      <td>-0.02296</td>
      <td>0.03275</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>-0.095</td>
      <td>-0.085</td>
      <td>-0.032</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.112</td>
      <td>0.081</td>
      <td>-0.048</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.007</td>
      <td>-0.074</td>
      <td>23.93</td>
      <td>167.6</td>
      <td>324.5</td>
      <td>-0.04348</td>
      <td>-0.02927</td>
      <td>-0.00649</td>
      <td>-0.01440</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>-0.035</td>
      <td>0.090</td>
      <td>0.066</td>
      <td>0.105</td>
      <td>0.032</td>
      <td>0.040</td>
      <td>-0.083</td>
      <td>0.013</td>
      <td>0.097</td>
      <td>0.048</td>
      <td>...</td>
      <td>0.099</td>
      <td>0.008</td>
      <td>24.06</td>
      <td>166.6</td>
      <td>325.5</td>
      <td>0.14545</td>
      <td>0.06182</td>
      <td>0.08497</td>
      <td>0.01663</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.088</td>
      <td>0.062</td>
      <td>0.032</td>
      <td>0.048</td>
      <td>0.109</td>
      <td>0.073</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.137</td>
      <td>0.085</td>
      <td>...</td>
      <td>-0.175</td>
      <td>0.171</td>
      <td>24.31</td>
      <td>167.6</td>
      <td>326.6</td>
      <td>0.03175</td>
      <td>0.06507</td>
      <td>0.03012</td>
      <td>-0.00413</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.064</td>
      <td>0.065</td>
      <td>0.082</td>
      <td>0.197</td>
      <td>0.023</td>
      <td>0.095</td>
      <td>0.030</td>
      <td>0.027</td>
      <td>0.063</td>
      <td>0.113</td>
      <td>...</td>
      <td>-0.077</td>
      <td>-0.004</td>
      <td>24.53</td>
      <td>168.8</td>
      <td>327.4</td>
      <td>0.04615</td>
      <td>0.06624</td>
      <td>0.07101</td>
      <td>-0.02318</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.032</td>
      <td>0.005</td>
      <td>0.022</td>
      <td>0.000</td>
      <td>-0.055</td>
      <td>0.162</td>
      <td>0.122</td>
      <td>0.019</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>...</td>
      <td>-0.038</td>
      <td>0.072</td>
      <td>23.12</td>
      <td>169.6</td>
      <td>328.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>0.093</td>
      <td>0.101</td>
      <td>0.048</td>
      <td>-0.051</td>
      <td>-0.044</td>
      <td>0.093</td>
      <td>-0.055</td>
      <td>0.121</td>
      <td>0.034</td>
      <td>0.003</td>
      <td>...</td>
      <td>0.071</td>
      <td>0.123</td>
      <td>17.65</td>
      <td>168.4</td>
      <td>327.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>0.066</td>
      <td>0.153</td>
      <td>0.021</td>
      <td>-0.040</td>
      <td>-0.043</td>
      <td>-0.063</td>
      <td>0.076</td>
      <td>0.072</td>
      <td>0.174</td>
      <td>0.004</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.051</td>
      <td>12.62</td>
      <td>166.1</td>
      <td>326.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.013</td>
      <td>-0.042</td>
      <td>-0.006</td>
      <td>-0.097</td>
      <td>0.061</td>
      <td>0.119</td>
      <td>0.059</td>
      <td>-0.051</td>
      <td>0.113</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.050</td>
      <td>-0.037</td>
      <td>10.68</td>
      <td>167.6</td>
      <td>325.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-05-01</th>
      <td>0.072</td>
      <td>0.038</td>
      <td>0.042</td>
      <td>-0.046</td>
      <td>-0.015</td>
      <td>0.037</td>
      <td>-0.043</td>
      <td>0.109</td>
      <td>-0.040</td>
      <td>-0.018</td>
      <td>...</td>
      <td>0.069</td>
      <td>0.010</td>
      <td>10.75</td>
      <td>166.9</td>
      <td>326.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-06-01</th>
      <td>-0.013</td>
      <td>-0.036</td>
      <td>0.017</td>
      <td>-0.161</td>
      <td>-0.155</td>
      <td>-0.063</td>
      <td>-0.070</td>
      <td>0.071</td>
      <td>-0.038</td>
      <td>-0.039</td>
      <td>...</td>
      <td>-0.042</td>
      <td>-0.061</td>
      <td>10.68</td>
      <td>166.9</td>
      <td>327.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-07-01</th>
      <td>-0.060</td>
      <td>-0.117</td>
      <td>0.125</td>
      <td>-0.038</td>
      <td>-0.072</td>
      <td>0.066</td>
      <td>0.018</td>
      <td>0.049</td>
      <td>-0.105</td>
      <td>-0.096</td>
      <td>...</td>
      <td>-0.036</td>
      <td>-0.048</td>
      <td>9.25</td>
      <td>167.9</td>
      <td>328.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-08-01</th>
      <td>0.115</td>
      <td>0.082</td>
      <td>0.061</td>
      <td>-0.040</td>
      <td>0.167</td>
      <td>0.105</td>
      <td>0.018</td>
      <td>0.003</td>
      <td>0.111</td>
      <td>0.055</td>
      <td>...</td>
      <td>0.135</td>
      <td>0.122</td>
      <td>9.77</td>
      <td>168.1</td>
      <td>328.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-09-01</th>
      <td>-0.052</td>
      <td>-0.111</td>
      <td>-0.139</td>
      <td>0.021</td>
      <td>-0.240</td>
      <td>-0.110</td>
      <td>0.026</td>
      <td>-0.088</td>
      <td>0.037</td>
      <td>-0.031</td>
      <td>...</td>
      <td>0.026</td>
      <td>-0.058</td>
      <td>11.09</td>
      <td>167.9</td>
      <td>330.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-10-01</th>
      <td>0.059</td>
      <td>0.040</td>
      <td>0.045</td>
      <td>0.000</td>
      <td>0.105</td>
      <td>0.103</td>
      <td>0.134</td>
      <td>0.123</td>
      <td>-0.069</td>
      <td>-0.081</td>
      <td>...</td>
      <td>0.043</td>
      <td>0.135</td>
      <td>11.00</td>
      <td>168.4</td>
      <td>330.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-11-01</th>
      <td>0.023</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.143</td>
      <td>0.020</td>
      <td>0.048</td>
      <td>-0.018</td>
      <td>0.011</td>
      <td>-0.020</td>
      <td>0.037</td>
      <td>...</td>
      <td>-0.028</td>
      <td>0.006</td>
      <td>11.05</td>
      <td>169.3</td>
      <td>330.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-12-01</th>
      <td>-0.027</td>
      <td>0.019</td>
      <td>-0.046</td>
      <td>0.028</td>
      <td>-0.078</td>
      <td>0.008</td>
      <td>-0.010</td>
      <td>-0.034</td>
      <td>-0.060</td>
      <td>-0.056</td>
      <td>...</td>
      <td>0.047</td>
      <td>-0.041</td>
      <td>11.75</td>
      <td>170.2</td>
      <td>331.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-01-01</th>
      <td>0.276</td>
      <td>0.087</td>
      <td>0.040</td>
      <td>0.093</td>
      <td>0.135</td>
      <td>0.385</td>
      <td>0.161</td>
      <td>0.123</td>
      <td>0.057</td>
      <td>0.073</td>
      <td>...</td>
      <td>0.049</td>
      <td>0.270</td>
      <td>13.89</td>
      <td>169.6</td>
      <td>333.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-02-01</th>
      <td>-0.008</td>
      <td>-0.066</td>
      <td>-0.067</td>
      <td>-0.064</td>
      <td>0.045</td>
      <td>0.056</td>
      <td>0.133</td>
      <td>0.049</td>
      <td>0.019</td>
      <td>0.092</td>
      <td>...</td>
      <td>-0.080</td>
      <td>0.094</td>
      <td>14.50</td>
      <td>170.8</td>
      <td>334.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-03-01</th>
      <td>0.071</td>
      <td>-0.052</td>
      <td>-0.050</td>
      <td>-0.087</td>
      <td>-0.096</td>
      <td>0.061</td>
      <td>-0.129</td>
      <td>0.010</td>
      <td>0.040</td>
      <td>0.076</td>
      <td>...</td>
      <td>0.103</td>
      <td>0.089</td>
      <td>14.53</td>
      <td>171.2</td>
      <td>335.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-04-01</th>
      <td>-0.037</td>
      <td>0.070</td>
      <td>0.020</td>
      <td>-0.025</td>
      <td>-0.020</td>
      <td>0.055</td>
      <td>-0.121</td>
      <td>-0.104</td>
      <td>-0.063</td>
      <td>0.067</td>
      <td>...</td>
      <td>-0.094</td>
      <td>-0.027</td>
      <td>14.95</td>
      <td>171.2</td>
      <td>337.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-05-01</th>
      <td>-0.111</td>
      <td>0.052</td>
      <td>-0.012</td>
      <td>0.000</td>
      <td>0.161</td>
      <td>-0.082</td>
      <td>0.151</td>
      <td>0.190</td>
      <td>0.138</td>
      <td>0.006</td>
      <td>...</td>
      <td>0.114</td>
      <td>-0.107</td>
      <td>15.29</td>
      <td>172.3</td>
      <td>338.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-06-01</th>
      <td>0.063</td>
      <td>0.051</td>
      <td>0.059</td>
      <td>0.081</td>
      <td>-0.145</td>
      <td>0.041</td>
      <td>0.014</td>
      <td>0.030</td>
      <td>0.005</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.073</td>
      <td>0.026</td>
      <td>15.95</td>
      <td>173.5</td>
      <td>340.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-07-01</th>
      <td>0.064</td>
      <td>0.041</td>
      <td>-0.039</td>
      <td>0.071</td>
      <td>0.057</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.036</td>
      <td>0.232</td>
      <td>-0.009</td>
      <td>...</td>
      <td>0.142</td>
      <td>0.021</td>
      <td>16.88</td>
      <td>175.5</td>
      <td>340.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-08-01</th>
      <td>0.061</td>
      <td>0.033</td>
      <td>0.043</td>
      <td>-0.044</td>
      <td>-0.008</td>
      <td>0.157</td>
      <td>-0.037</td>
      <td>0.022</td>
      <td>-0.113</td>
      <td>0.053</td>
      <td>...</td>
      <td>-0.076</td>
      <td>0.081</td>
      <td>17.06</td>
      <td>176.3</td>
      <td>342.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-09-01</th>
      <td>-0.029</td>
      <td>-0.086</td>
      <td>-0.006</td>
      <td>0.004</td>
      <td>0.015</td>
      <td>0.001</td>
      <td>-0.067</td>
      <td>-0.009</td>
      <td>-0.061</td>
      <td>-0.105</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.054</td>
      <td>16.29</td>
      <td>176.1</td>
      <td>344.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-10-01</th>
      <td>-0.274</td>
      <td>-0.282</td>
      <td>-0.017</td>
      <td>-0.372</td>
      <td>-0.342</td>
      <td>-0.281</td>
      <td>-0.260</td>
      <td>-0.148</td>
      <td>-0.288</td>
      <td>-0.187</td>
      <td>...</td>
      <td>-0.194</td>
      <td>-0.271</td>
      <td>15.95</td>
      <td>178.1</td>
      <td>345.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-11-01</th>
      <td>0.043</td>
      <td>-0.136</td>
      <td>-0.012</td>
      <td>-0.148</td>
      <td>-0.075</td>
      <td>-0.127</td>
      <td>-0.137</td>
      <td>-0.102</td>
      <td>-0.085</td>
      <td>-0.087</td>
      <td>...</td>
      <td>-0.031</td>
      <td>-0.066</td>
      <td>15.46</td>
      <td>179.0</td>
      <td>345.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 27 columns</p>
</div>




```python
df=df.dropna()
```


```python
df[['BOISE','CONTIL','MARKET']].head(200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.034</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>-0.018</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>0.034</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>0.035</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>-0.017</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.103</td>
      <td>-0.056</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>0.040</td>
      <td>-0.014</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>0.069</td>
      <td>-0.009</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.024</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>-0.025</td>
      <td>-0.008</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.117</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>0.077</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.092</td>
      <td>-0.033</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.030</td>
      <td>-0.031</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.003</td>
      <td>-0.164</td>
    </tr>
  </tbody>
</table>
</div>




```python
BOISE_prices=df['BOISE'].head(100)
BOISE_prices
```




    Date
    1978-01-01   -0.079
    1978-02-01    0.013
    1978-03-01    0.070
    1978-04-01    0.120
    1978-05-01    0.071
    1978-06-01   -0.098
    1978-07-01    0.140
    1978-08-01    0.078
    1978-09-01   -0.059
    1978-10-01   -0.118
    1978-11-01   -0.060
    1978-12-01    0.067
    1979-01-01    0.168
    1979-02-01   -0.032
    1979-03-01    0.178
    1979-04-01   -0.043
    1979-05-01   -0.026
    1979-06-01    0.057
    1979-07-01    0.047
    1979-08-01    0.038
    1979-09-01    0.050
    1979-10-01   -0.151
    1979-11-01   -0.004
    1979-12-01    0.042
    1980-01-01    0.107
    1980-02-01   -0.070
    1980-03-01   -0.138
    1980-04-01    0.042
    1980-05-01    0.109
    1980-06-01    0.068
    1980-07-01    0.073
    1980-08-01   -0.045
    1980-09-01    0.019
    1980-10-01   -0.054
    1980-11-01    0.028
    1980-12-01   -0.047
    1981-01-01    0.011
    1981-02-01    0.152
    1981-03-01    0.056
    1981-04-01    0.045
    1981-05-01    0.032
    1981-06-01   -0.037
    1981-07-01   -0.065
    1981-08-01   -0.125
    1981-09-01   -0.062
    Name: BOISE, dtype: float64




```python

```


```python
plt.rcParams["figure.figsize"]=[16,9]
BOISE_prices.plot()
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_9_1.png)



```python
CONTIL_prices=df['CONTIL']
CONTIL_prices
```




    Date
    1978-01-01   -0.129
    1978-02-01    0.037
    1978-03-01    0.003
    1978-04-01    0.180
    1978-05-01    0.061
    1978-06-01   -0.059
    1978-07-01    0.066
    1978-08-01    0.033
    1978-09-01   -0.013
    1978-10-01   -0.123
    1978-11-01   -0.038
    1978-12-01    0.047
    1979-01-01   -0.024
    1979-02-01   -0.020
    1979-03-01    0.043
    1979-04-01    0.064
    1979-05-01    0.005
    1979-06-01    0.092
    1979-07-01   -0.034
    1979-08-01    0.058
    1979-09-01   -0.033
    1979-10-01   -0.136
    1979-11-01    0.081
    1979-12-01    0.104
    1980-01-01   -0.103
    1980-02-01   -0.087
    1980-03-01    0.085
    1980-04-01    0.074
    1980-05-01    0.023
    1980-06-01    0.064
    1980-07-01   -0.034
    1980-08-01   -0.018
    1980-09-01    0.034
    1980-10-01    0.035
    1980-11-01   -0.017
    1980-12-01    0.103
    1981-01-01    0.040
    1981-02-01    0.069
    1981-03-01    0.024
    1981-04-01   -0.025
    1981-05-01    0.117
    1981-06-01    0.077
    1981-07-01   -0.092
    1981-08-01   -0.030
    1981-09-01    0.003
    Name: CONTIL, dtype: float64




```python

```


```python
plt.rcParams["figure.figsize"]=[16,9]
CONTIL_prices.plot()
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_12_1.png)



```python
df.columns
```




    Index(['BOISE', 'CITCRP', 'CONED', 'CONTIL', 'DATGEN', 'DEC', 'DELTA',
           'GENMIL', 'GERBER', 'IBM', 'MARKET', 'MOBIL', 'MOTOR', 'PANAM', 'PSNH',
           'RKFREE', 'TANDY', 'TEXACO', 'WEYER', 'POIL', 'FRBIND', 'CPI', 'GPU',
           'DOW', 'DUPONT', 'GOLD', 'CONOCO'],
          dtype='object')




```python
df1=df[['BOISE','CONTIL','MARKET']].copy()
```


```python
df1.head(200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.034</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>-0.018</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>0.034</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>0.035</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>-0.017</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.103</td>
      <td>-0.056</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>0.040</td>
      <td>-0.014</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>0.069</td>
      <td>-0.009</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.024</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>-0.025</td>
      <td>-0.008</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.117</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>0.077</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.092</td>
      <td>-0.033</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.030</td>
      <td>-0.031</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.003</td>
      <td>-0.164</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
df1.head(150)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.034</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>-0.018</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>0.034</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>0.035</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>-0.017</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.103</td>
      <td>-0.056</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>0.040</td>
      <td>-0.014</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>0.069</td>
      <td>-0.009</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.024</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>-0.025</td>
      <td>-0.008</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.117</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>0.077</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.092</td>
      <td>-0.033</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.030</td>
      <td>-0.031</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.003</td>
      <td>-0.164</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set()
plt.figure(figsize=(8,5))
ax = sns.scatterplot(x='MARKET', y='CONTIL', data=df)
```


![png](output_18_0.png)



```python
df1.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.034</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>-0.018</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>0.034</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>0.035</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>-0.017</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.103</td>
      <td>-0.056</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>0.040</td>
      <td>-0.014</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>0.069</td>
      <td>-0.009</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.024</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>-0.025</td>
      <td>-0.008</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.117</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>0.077</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.092</td>
      <td>-0.033</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.030</td>
      <td>-0.031</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.003</td>
      <td>-0.164</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2=df1.copy()
df2=df2.sub(df.ix[:,-1],axis=0)
df2=df2.ix[:,:-1]
```


```python
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>0.03350</td>
      <td>-0.01650</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>-0.00015</td>
      <td>0.02385</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.06531</td>
      <td>-0.00169</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.06393</td>
      <td>0.12393</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.02321</td>
      <td>0.01321</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.00029</td>
      <td>0.03929</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.15422</td>
      <td>0.08022</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>-0.01719</td>
      <td>-0.06219</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.10789</td>
      <td>-0.06189</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>0.01336</td>
      <td>0.00836</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.08927</td>
      <td>-0.06727</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>-0.01473</td>
      <td>-0.03473</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.10133</td>
      <td>-0.09067</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.05700</td>
      <td>-0.04500</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.05043</td>
      <td>-0.08457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.04300</td>
      <td>0.06400</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.03111</td>
      <td>-0.00011</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.05697</td>
      <td>-0.02197</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.02720</td>
      <td>-0.05380</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.00860</td>
      <td>0.01140</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.04375</td>
      <td>-0.12675</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.11671</td>
      <td>-0.10171</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.07441</td>
      <td>0.01059</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>-0.01387</td>
      <td>0.04813</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.01734</td>
      <td>-0.22734</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.08600</td>
      <td>-0.10300</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.02789</td>
      <td>0.25089</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.01402</td>
      <td>0.01798</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.05754</td>
      <td>-0.02846</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00598</td>
      <td>-0.00998</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.05162</td>
      <td>-0.05538</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.02267</td>
      <td>0.00433</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.03388</td>
      <td>-0.01888</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.13482</td>
      <td>-0.04582</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.21081</td>
      <td>-0.25581</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.05283</td>
      <td>0.20283</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.09322</td>
      <td>0.12222</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.15992</td>
      <td>0.07692</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.09422</td>
      <td>0.06222</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.17083</td>
      <td>0.10083</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>-0.02406</td>
      <td>0.06094</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.30577</td>
      <td>-0.19177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.45813</td>
      <td>-0.48513</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.03897</td>
      <td>0.05603</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.16156</td>
      <td>0.22656</td>
    </tr>
  </tbody>
</table>
</div>




```python
f=['BOISE','CONTIL']
```


```python
f
```




    ['BOISE', 'CONTIL']




```python
df2.columns=f
```


```python
df2.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>0.03350</td>
      <td>-0.01650</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>-0.00015</td>
      <td>0.02385</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.06531</td>
      <td>-0.00169</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.06393</td>
      <td>0.12393</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.02321</td>
      <td>0.01321</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.00029</td>
      <td>0.03929</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.15422</td>
      <td>0.08022</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>-0.01719</td>
      <td>-0.06219</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.10789</td>
      <td>-0.06189</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>0.01336</td>
      <td>0.00836</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.08927</td>
      <td>-0.06727</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>-0.01473</td>
      <td>-0.03473</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.10133</td>
      <td>-0.09067</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.05700</td>
      <td>-0.04500</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.05043</td>
      <td>-0.08457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.04300</td>
      <td>0.06400</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.03111</td>
      <td>-0.00011</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.05697</td>
      <td>-0.02197</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.02720</td>
      <td>-0.05380</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.00860</td>
      <td>0.01140</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.04375</td>
      <td>-0.12675</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.11671</td>
      <td>-0.10171</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.07441</td>
      <td>0.01059</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>-0.01387</td>
      <td>0.04813</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.01734</td>
      <td>-0.22734</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.08600</td>
      <td>-0.10300</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.02789</td>
      <td>0.25089</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.01402</td>
      <td>0.01798</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.05754</td>
      <td>-0.02846</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00598</td>
      <td>-0.00998</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.05162</td>
      <td>-0.05538</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.02267</td>
      <td>0.00433</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.03388</td>
      <td>-0.01888</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.13482</td>
      <td>-0.04582</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.21081</td>
      <td>-0.25581</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.05283</td>
      <td>0.20283</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.09322</td>
      <td>0.12222</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.15992</td>
      <td>0.07692</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.09422</td>
      <td>0.06222</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.17083</td>
      <td>0.10083</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>-0.02406</td>
      <td>0.06094</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.30577</td>
      <td>-0.19177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.45813</td>
      <td>-0.48513</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.03897</td>
      <td>0.05603</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.16156</td>
      <td>0.22656</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set()
plt.figure(figsize=(8,5))
ax = sns.scatterplot(x='MARKET', y='BOISE', data=df)
```


![png](output_27_0.png)



```python
df1.index
```




    DatetimeIndex(['1978-01-01', '1978-02-01', '1978-03-01', '1978-04-01',
                   '1978-05-01', '1978-06-01', '1978-07-01', '1978-08-01',
                   '1978-09-01', '1978-10-01', '1978-11-01', '1978-12-01',
                   '1979-01-01', '1979-02-01', '1979-03-01', '1979-04-01',
                   '1979-05-01', '1979-06-01', '1979-07-01', '1979-08-01',
                   '1979-09-01', '1979-10-01', '1979-11-01', '1979-12-01',
                   '1980-01-01', '1980-02-01', '1980-03-01', '1980-04-01',
                   '1980-05-01', '1980-06-01', '1980-07-01', '1980-08-01',
                   '1980-09-01', '1980-10-01', '1980-11-01', '1980-12-01',
                   '1981-01-01', '1981-02-01', '1981-03-01', '1981-04-01',
                   '1981-05-01', '1981-06-01', '1981-07-01', '1981-08-01',
                   '1981-09-01'],
                  dtype='datetime64[ns]', name='Date', freq=None)




```python
len(df1)
```




    45




```python
df3=df1.groupby(np.arange(len(df1))//(3*3))
```


```python
np.arange(len(df1))//(3*3)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
           4])




```python
df1['Group column']=np.arange(len(df1))//(3*3)
```


```python
df1.head(200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
      <th>Group column</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.034</td>
      <td>0.065</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>-0.018</td>
      <td>0.025</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>0.034</td>
      <td>0.015</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>0.035</td>
      <td>0.006</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>-0.017</td>
      <td>0.092</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.103</td>
      <td>-0.056</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>0.040</td>
      <td>-0.014</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>0.069</td>
      <td>-0.009</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.024</td>
      <td>0.067</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>-0.025</td>
      <td>-0.008</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.117</td>
      <td>0.064</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>0.077</td>
      <td>-0.003</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.092</td>
      <td>-0.033</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.030</td>
      <td>-0.031</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.003</td>
      <td>-0.164</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
1986-1978
```




    8




```python
df_group_0=df3.get_group(0)
df_group_1=df3.get_group(1)
df_group_2=df3.get_group(2)
df_group_3=df3.get_group(3)
```


```python
df_group_0
df.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.032</td>
      <td>0.011</td>
      <td>0.066</td>
      <td>0.143</td>
      <td>0.107</td>
      <td>0.185</td>
      <td>0.075</td>
      <td>-0.012</td>
      <td>0.092</td>
      <td>...</td>
      <td>0.042</td>
      <td>0.164</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>196.7</td>
      <td>0.04405</td>
      <td>0.07107</td>
      <td>0.07813</td>
      <td>0.02814</td>
      <td>-0.01422</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.088</td>
      <td>0.024</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>-0.017</td>
      <td>-0.021</td>
      <td>-0.051</td>
      <td>-0.079</td>
      <td>0.049</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.039</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>197.8</td>
      <td>-0.04636</td>
      <td>0.04265</td>
      <td>0.03727</td>
      <td>0.09005</td>
      <td>0.09519</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.011</td>
      <td>0.048</td>
      <td>-0.013</td>
      <td>-0.031</td>
      <td>-0.037</td>
      <td>-0.081</td>
      <td>-0.012</td>
      <td>0.104</td>
      <td>-0.051</td>
      <td>...</td>
      <td>0.010</td>
      <td>-0.021</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>199.3</td>
      <td>0.03472</td>
      <td>0.04000</td>
      <td>0.03024</td>
      <td>0.02977</td>
      <td>0.04889</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.071</td>
      <td>-0.067</td>
      <td>-0.123</td>
      <td>-0.085</td>
      <td>-0.077</td>
      <td>-0.153</td>
      <td>-0.032</td>
      <td>-0.138</td>
      <td>-0.046</td>
      <td>...</td>
      <td>-0.066</td>
      <td>-0.090</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>200.9</td>
      <td>-0.07651</td>
      <td>-0.07522</td>
      <td>-0.06067</td>
      <td>0.07194</td>
      <td>-0.13136</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.005</td>
      <td>0.035</td>
      <td>-0.038</td>
      <td>0.044</td>
      <td>0.064</td>
      <td>0.055</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>202.0</td>
      <td>0.04478</td>
      <td>0.00478</td>
      <td>0.05000</td>
      <td>-0.09443</td>
      <td>0.02927</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.047</td>
      <td>0.034</td>
      <td>0.117</td>
      <td>-0.023</td>
      <td>0.022</td>
      <td>-0.086</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.000</td>
      <td>-0.034</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>203.3</td>
      <td>0.00000</td>
      <td>-0.03905</td>
      <td>0.02857</td>
      <td>0.00941</td>
      <td>0.08173</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>0.043</td>
      <td>0.076</td>
      <td>-0.024</td>
      <td>-0.008</td>
      <td>-0.012</td>
      <td>-0.054</td>
      <td>-0.032</td>
      <td>0.042</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.037</td>
      <td>0.203</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>204.7</td>
      <td>0.05429</td>
      <td>0.07035</td>
      <td>0.06151</td>
      <td>0.09336</td>
      <td>0.06667</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.082</td>
      <td>-0.011</td>
      <td>-0.020</td>
      <td>-0.015</td>
      <td>-0.066</td>
      <td>-0.060</td>
      <td>-0.079</td>
      <td>-0.023</td>
      <td>-0.017</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.038</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>207.1</td>
      <td>-0.05556</td>
      <td>-0.04225</td>
      <td>-0.02150</td>
      <td>0.08042</td>
      <td>0.02500</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.171</td>
      <td>0.088</td>
      <td>0.098</td>
      <td>-0.043</td>
      <td>0.065</td>
      <td>0.052</td>
      <td>...</td>
      <td>0.068</td>
      <td>0.097</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>209.1</td>
      <td>-0.04412</td>
      <td>0.11176</td>
      <td>0.09082</td>
      <td>-0.01428</td>
      <td>0.12757</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.000</td>
      <td>-0.057</td>
      <td>0.064</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>-0.056</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.069</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>211.5</td>
      <td>-0.33077</td>
      <td>-0.07143</td>
      <td>-0.06200</td>
      <td>-0.01333</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.005</td>
      <td>-0.045</td>
      <td>-0.028</td>
      <td>0.063</td>
      <td>0.035</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.040</td>
      <td>-0.013</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>214.1</td>
      <td>-0.17241</td>
      <td>-0.01923</td>
      <td>-0.03305</td>
      <td>0.07745</td>
      <td>0.00511</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.095</td>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>-0.006</td>
      <td>-0.043</td>
      <td>0.095</td>
      <td>-0.035</td>
      <td>...</td>
      <td>0.083</td>
      <td>0.053</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>216.6</td>
      <td>0.15714</td>
      <td>0.02843</td>
      <td>-0.02174</td>
      <td>0.08434</td>
      <td>0.11397</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.075</td>
      <td>0.015</td>
      <td>-0.034</td>
      <td>-0.059</td>
      <td>0.009</td>
      <td>0.075</td>
      <td>-0.013</td>
      <td>-0.096</td>
      <td>-0.049</td>
      <td>...</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>218.9</td>
      <td>-0.01235</td>
      <td>0.08213</td>
      <td>-0.00909</td>
      <td>0.05802</td>
      <td>0.01980</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.041</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>0.030</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>0.067</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>-0.029</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.229</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.161</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>0.111</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>-0.021</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>0.035</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>-0.052</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>-0.060</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>-0.015</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
  </tbody>
</table>
<p>45 rows × 27 columns</p>
</div>




```python
df2=df1.copy()
df2=df2.sub(df.ix[:,-1],axis=0)
df2=df2.ix[:,:-1]
df2.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>0.03350</td>
      <td>-0.01650</td>
      <td>0.06750</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>-0.00015</td>
      <td>0.02385</td>
      <td>-0.00315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.06531</td>
      <td>-0.00169</td>
      <td>0.04531</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.06393</td>
      <td>0.12393</td>
      <td>0.00693</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.02321</td>
      <td>0.01321</td>
      <td>0.01921</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.00029</td>
      <td>0.03929</td>
      <td>0.10529</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.15422</td>
      <td>0.08022</td>
      <td>0.08522</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>-0.01719</td>
      <td>-0.06219</td>
      <td>-0.01619</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.10789</td>
      <td>-0.06189</td>
      <td>-0.04689</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>0.01336</td>
      <td>0.00836</td>
      <td>-0.05764</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.08927</td>
      <td>-0.06727</td>
      <td>0.05473</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>-0.01473</td>
      <td>-0.03473</td>
      <td>-0.06673</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.10133</td>
      <td>-0.09067</td>
      <td>-0.00867</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.05700</td>
      <td>-0.04500</td>
      <td>-0.01400</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.05043</td>
      <td>-0.08457</td>
      <td>-0.00457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.04300</td>
      <td>0.06400</td>
      <td>0.02600</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.03111</td>
      <td>-0.00011</td>
      <td>0.00889</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.05697</td>
      <td>-0.02197</td>
      <td>-0.03897</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.02720</td>
      <td>-0.05380</td>
      <td>-0.03280</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.00860</td>
      <td>0.01140</td>
      <td>0.04840</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.04375</td>
      <td>-0.12675</td>
      <td>-0.05475</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.11671</td>
      <td>-0.10171</td>
      <td>-0.06271</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.07441</td>
      <td>0.01059</td>
      <td>0.04559</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>-0.01387</td>
      <td>0.04813</td>
      <td>0.03013</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.01734</td>
      <td>-0.22734</td>
      <td>-0.00034</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.08600</td>
      <td>-0.10300</td>
      <td>0.09600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.02789</td>
      <td>0.25089</td>
      <td>-0.07711</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.01402</td>
      <td>0.01798</td>
      <td>0.02398</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.05754</td>
      <td>-0.02846</td>
      <td>0.01054</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00598</td>
      <td>-0.00998</td>
      <td>0.01202</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.05162</td>
      <td>-0.05538</td>
      <td>0.04362</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.02267</td>
      <td>0.00433</td>
      <td>0.04733</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.03388</td>
      <td>-0.01888</td>
      <td>-0.03788</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.13482</td>
      <td>-0.04582</td>
      <td>-0.07482</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.21081</td>
      <td>-0.25581</td>
      <td>-0.14681</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.05283</td>
      <td>0.20283</td>
      <td>0.04383</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.09322</td>
      <td>0.12222</td>
      <td>0.06822</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.15992</td>
      <td>0.07692</td>
      <td>-0.00108</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.09422</td>
      <td>0.06222</td>
      <td>0.10522</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.17083</td>
      <td>0.10083</td>
      <td>0.11783</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>-0.02406</td>
      <td>0.06094</td>
      <td>0.00794</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.30577</td>
      <td>-0.19177</td>
      <td>-0.27177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.45813</td>
      <td>-0.48513</td>
      <td>-0.42613</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.03897</td>
      <td>0.05603</td>
      <td>0.05503</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.16156</td>
      <td>0.22656</td>
      <td>0.05956</td>
    </tr>
  </tbody>
</table>
</div>




```python
f=['BOISE','CONTIL','MARKET']
```


```python
f
```




    ['BOISE', 'CONTIL', 'MARKET']




```python
df2.columns=f
```


```python
df2.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>0.03350</td>
      <td>-0.01650</td>
      <td>0.06750</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>-0.00015</td>
      <td>0.02385</td>
      <td>-0.00315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.06531</td>
      <td>-0.00169</td>
      <td>0.04531</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.06393</td>
      <td>0.12393</td>
      <td>0.00693</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.02321</td>
      <td>0.01321</td>
      <td>0.01921</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.00029</td>
      <td>0.03929</td>
      <td>0.10529</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.15422</td>
      <td>0.08022</td>
      <td>0.08522</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>-0.01719</td>
      <td>-0.06219</td>
      <td>-0.01619</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.10789</td>
      <td>-0.06189</td>
      <td>-0.04689</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>0.01336</td>
      <td>0.00836</td>
      <td>-0.05764</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.08927</td>
      <td>-0.06727</td>
      <td>0.05473</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>-0.01473</td>
      <td>-0.03473</td>
      <td>-0.06673</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.10133</td>
      <td>-0.09067</td>
      <td>-0.00867</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.05700</td>
      <td>-0.04500</td>
      <td>-0.01400</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.05043</td>
      <td>-0.08457</td>
      <td>-0.00457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.04300</td>
      <td>0.06400</td>
      <td>0.02600</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.03111</td>
      <td>-0.00011</td>
      <td>0.00889</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.05697</td>
      <td>-0.02197</td>
      <td>-0.03897</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.02720</td>
      <td>-0.05380</td>
      <td>-0.03280</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.00860</td>
      <td>0.01140</td>
      <td>0.04840</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.04375</td>
      <td>-0.12675</td>
      <td>-0.05475</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.11671</td>
      <td>-0.10171</td>
      <td>-0.06271</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.07441</td>
      <td>0.01059</td>
      <td>0.04559</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>-0.01387</td>
      <td>0.04813</td>
      <td>0.03013</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.01734</td>
      <td>-0.22734</td>
      <td>-0.00034</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.08600</td>
      <td>-0.10300</td>
      <td>0.09600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.02789</td>
      <td>0.25089</td>
      <td>-0.07711</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.01402</td>
      <td>0.01798</td>
      <td>0.02398</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.05754</td>
      <td>-0.02846</td>
      <td>0.01054</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00598</td>
      <td>-0.00998</td>
      <td>0.01202</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.05162</td>
      <td>-0.05538</td>
      <td>0.04362</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.02267</td>
      <td>0.00433</td>
      <td>0.04733</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.03388</td>
      <td>-0.01888</td>
      <td>-0.03788</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.13482</td>
      <td>-0.04582</td>
      <td>-0.07482</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.21081</td>
      <td>-0.25581</td>
      <td>-0.14681</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.05283</td>
      <td>0.20283</td>
      <td>0.04383</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.09322</td>
      <td>0.12222</td>
      <td>0.06822</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.15992</td>
      <td>0.07692</td>
      <td>-0.00108</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.09422</td>
      <td>0.06222</td>
      <td>0.10522</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.17083</td>
      <td>0.10083</td>
      <td>0.11783</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>-0.02406</td>
      <td>0.06094</td>
      <td>0.00794</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.30577</td>
      <td>-0.19177</td>
      <td>-0.27177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.45813</td>
      <td>-0.48513</td>
      <td>-0.42613</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.03897</td>
      <td>0.05603</td>
      <td>0.05503</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.16156</td>
      <td>0.22656</td>
      <td>0.05956</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set()
plt.figure(figsize=(8,5))
ax = sns.scatterplot(x='MARKET', y='BOISE', data=df2)
```


![png](output_42_0.png)



```python
start_date = dt.datetime(1979,3,1)
end_date = dt.datetime(1987,2,1)
#greater than the start date and smaller than the end date
select = (df2.index>=start_date)*(df2.index<=end_date)

# Copy the selected dataframe into df2
df3=df2[select].copy()
```


```python
df3.head(200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-03-01</th>
      <td>0.05043</td>
      <td>-0.08457</td>
      <td>-0.00457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.04300</td>
      <td>0.06400</td>
      <td>0.02600</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.03111</td>
      <td>-0.00011</td>
      <td>0.00889</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.05697</td>
      <td>-0.02197</td>
      <td>-0.03897</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.02720</td>
      <td>-0.05380</td>
      <td>-0.03280</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.00860</td>
      <td>0.01140</td>
      <td>0.04840</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.04375</td>
      <td>-0.12675</td>
      <td>-0.05475</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.11671</td>
      <td>-0.10171</td>
      <td>-0.06271</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.07441</td>
      <td>0.01059</td>
      <td>0.04559</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>-0.01387</td>
      <td>0.04813</td>
      <td>0.03013</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.01734</td>
      <td>-0.22734</td>
      <td>-0.00034</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.08600</td>
      <td>-0.10300</td>
      <td>0.09600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.02789</td>
      <td>0.25089</td>
      <td>-0.07711</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.01402</td>
      <td>0.01798</td>
      <td>0.02398</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.05754</td>
      <td>-0.02846</td>
      <td>0.01054</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00598</td>
      <td>-0.00998</td>
      <td>0.01202</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.05162</td>
      <td>-0.05538</td>
      <td>0.04362</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.02267</td>
      <td>0.00433</td>
      <td>0.04733</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.03388</td>
      <td>-0.01888</td>
      <td>-0.03788</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.13482</td>
      <td>-0.04582</td>
      <td>-0.07482</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.21081</td>
      <td>-0.25581</td>
      <td>-0.14681</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.05283</td>
      <td>0.20283</td>
      <td>0.04383</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.09322</td>
      <td>0.12222</td>
      <td>0.06822</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.15992</td>
      <td>0.07692</td>
      <td>-0.00108</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.09422</td>
      <td>0.06222</td>
      <td>0.10522</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.17083</td>
      <td>0.10083</td>
      <td>0.11783</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>-0.02406</td>
      <td>0.06094</td>
      <td>0.00794</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.30577</td>
      <td>-0.19177</td>
      <td>-0.27177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.45813</td>
      <td>-0.48513</td>
      <td>-0.42613</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.03897</td>
      <td>0.05603</td>
      <td>0.05503</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.16156</td>
      <td>0.22656</td>
      <td>0.05956</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4=df3[['CONTIL','MARKET']].copy()
df4.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-03-01</th>
      <td>-0.08457</td>
      <td>-0.00457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>0.06400</td>
      <td>0.02600</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.00011</td>
      <td>0.00889</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.02197</td>
      <td>-0.03897</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>-0.05380</td>
      <td>-0.03280</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.01140</td>
      <td>0.04840</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.12675</td>
      <td>-0.05475</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.10171</td>
      <td>-0.06271</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>0.01059</td>
      <td>0.04559</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.04813</td>
      <td>0.03013</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.22734</td>
      <td>-0.00034</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.10300</td>
      <td>0.09600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.25089</td>
      <td>-0.07711</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.01798</td>
      <td>0.02398</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>-0.02846</td>
      <td>0.01054</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00998</td>
      <td>0.01202</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>-0.05538</td>
      <td>0.04362</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>0.00433</td>
      <td>0.04733</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.01888</td>
      <td>-0.03788</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.04582</td>
      <td>-0.07482</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.25581</td>
      <td>-0.14681</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.20283</td>
      <td>0.04383</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.12222</td>
      <td>0.06822</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.07692</td>
      <td>-0.00108</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.06222</td>
      <td>0.10522</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.10083</td>
      <td>0.11783</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.06094</td>
      <td>0.00794</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.19177</td>
      <td>-0.27177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.48513</td>
      <td>-0.42613</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>0.05603</td>
      <td>0.05503</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.22656</td>
      <td>0.05956</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set()
plt.figure(figsize=(8,5))
ax = sns.scatterplot(x='MARKET', y='CONTIL', data=df3)
```


![png](output_46_0.png)



```python
choice_of_columns=['CONTIL', 'BOISE']
df6=Joining_columns(df3, 'MARKET', choice_of_columns, Name_of_new_column='Stock Return')
df6.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MARKET</th>
      <th>variable</th>
      <th>Stock Return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.00457</td>
      <td>CONTIL</td>
      <td>-0.08457</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02600</td>
      <td>CONTIL</td>
      <td>0.06400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00889</td>
      <td>CONTIL</td>
      <td>-0.00011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.03897</td>
      <td>CONTIL</td>
      <td>-0.02197</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.03280</td>
      <td>CONTIL</td>
      <td>-0.05380</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.04840</td>
      <td>CONTIL</td>
      <td>0.01140</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.05475</td>
      <td>CONTIL</td>
      <td>-0.12675</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.06271</td>
      <td>CONTIL</td>
      <td>-0.10171</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.04559</td>
      <td>CONTIL</td>
      <td>0.01059</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.03013</td>
      <td>CONTIL</td>
      <td>0.04813</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.00034</td>
      <td>CONTIL</td>
      <td>-0.22734</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.09600</td>
      <td>CONTIL</td>
      <td>-0.10300</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.07711</td>
      <td>CONTIL</td>
      <td>0.25089</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.02398</td>
      <td>CONTIL</td>
      <td>0.01798</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01054</td>
      <td>CONTIL</td>
      <td>-0.02846</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.01202</td>
      <td>CONTIL</td>
      <td>-0.00998</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.04362</td>
      <td>CONTIL</td>
      <td>-0.05538</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.04733</td>
      <td>CONTIL</td>
      <td>0.00433</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.03788</td>
      <td>CONTIL</td>
      <td>-0.01888</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.07482</td>
      <td>CONTIL</td>
      <td>-0.04582</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.14681</td>
      <td>CONTIL</td>
      <td>-0.25581</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.04383</td>
      <td>CONTIL</td>
      <td>0.20283</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.06822</td>
      <td>CONTIL</td>
      <td>0.12222</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.00108</td>
      <td>CONTIL</td>
      <td>0.07692</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.10522</td>
      <td>CONTIL</td>
      <td>0.06222</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.11783</td>
      <td>CONTIL</td>
      <td>0.10083</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.00794</td>
      <td>CONTIL</td>
      <td>0.06094</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.27177</td>
      <td>CONTIL</td>
      <td>-0.19177</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.42613</td>
      <td>CONTIL</td>
      <td>-0.48513</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.05503</td>
      <td>CONTIL</td>
      <td>0.05603</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.02600</td>
      <td>BOISE</td>
      <td>-0.04300</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.00889</td>
      <td>BOISE</td>
      <td>-0.03111</td>
    </tr>
    <tr>
      <th>34</th>
      <td>-0.03897</td>
      <td>BOISE</td>
      <td>-0.05697</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-0.03280</td>
      <td>BOISE</td>
      <td>0.02720</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.04840</td>
      <td>BOISE</td>
      <td>-0.00860</td>
    </tr>
    <tr>
      <th>37</th>
      <td>-0.05475</td>
      <td>BOISE</td>
      <td>-0.04375</td>
    </tr>
    <tr>
      <th>38</th>
      <td>-0.06271</td>
      <td>BOISE</td>
      <td>-0.11671</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.04559</td>
      <td>BOISE</td>
      <td>-0.07441</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.03013</td>
      <td>BOISE</td>
      <td>-0.01387</td>
    </tr>
    <tr>
      <th>41</th>
      <td>-0.00034</td>
      <td>BOISE</td>
      <td>-0.01734</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.09600</td>
      <td>BOISE</td>
      <td>-0.08600</td>
    </tr>
    <tr>
      <th>43</th>
      <td>-0.07711</td>
      <td>BOISE</td>
      <td>0.02789</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.02398</td>
      <td>BOISE</td>
      <td>-0.01402</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.01054</td>
      <td>BOISE</td>
      <td>0.05754</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.01202</td>
      <td>BOISE</td>
      <td>-0.00598</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.04362</td>
      <td>BOISE</td>
      <td>0.05162</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.04733</td>
      <td>BOISE</td>
      <td>-0.02267</td>
    </tr>
    <tr>
      <th>49</th>
      <td>-0.03788</td>
      <td>BOISE</td>
      <td>-0.03388</td>
    </tr>
    <tr>
      <th>50</th>
      <td>-0.07482</td>
      <td>BOISE</td>
      <td>-0.13482</td>
    </tr>
    <tr>
      <th>51</th>
      <td>-0.14681</td>
      <td>BOISE</td>
      <td>-0.21081</td>
    </tr>
    <tr>
      <th>52</th>
      <td>0.04383</td>
      <td>BOISE</td>
      <td>0.05283</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.06822</td>
      <td>BOISE</td>
      <td>0.09322</td>
    </tr>
    <tr>
      <th>54</th>
      <td>-0.00108</td>
      <td>BOISE</td>
      <td>0.15992</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.10522</td>
      <td>BOISE</td>
      <td>0.09422</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.11783</td>
      <td>BOISE</td>
      <td>0.17083</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.00794</td>
      <td>BOISE</td>
      <td>-0.02406</td>
    </tr>
    <tr>
      <th>58</th>
      <td>-0.27177</td>
      <td>BOISE</td>
      <td>-0.30577</td>
    </tr>
    <tr>
      <th>59</th>
      <td>-0.42613</td>
      <td>BOISE</td>
      <td>-0.45813</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.05503</td>
      <td>BOISE</td>
      <td>-0.03897</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.05956</td>
      <td>BOISE</td>
      <td>0.16156</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 3 columns</p>
</div>




```python
ig=sns.lmplot(x="MARKET", y="Stock Return", hue="variable", data=df6, height=7, aspect=1.8/1)
```


![png](output_48_0.png)



```python
import statsmodels.formula.api as smf
```


```python
formula = 'Q("CONTIL") ~ Q("MARKET")'
results = smf.ols(formula, df3).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CONTIL")   R-squared:                       0.500
    Model:                            OLS   Adj. R-squared:                  0.483
    Method:                 Least Squares   F-statistic:                     29.04
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           8.60e-06
    Time:                        14:38:32   Log-Likelihood:                 26.925
    No. Observations:                  31   AIC:                            -49.85
    Df Residuals:                      29   BIC:                            -46.98
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0043      0.019     -0.227      0.822      -0.043       0.035
    Q("MARKET")     0.9517      0.177      5.389      0.000       0.590       1.313
    ==============================================================================
    Omnibus:                        9.455   Durbin-Watson:                   2.064
    Prob(Omnibus):                  0.009   Jarque-Bera (JB):               10.072
    Skew:                           0.777   Prob(JB):                      0.00650
    Kurtosis:                       5.320   Cond. No.                         9.37
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
formula = 'Q("BOISE") ~ Q("MARKET")'
results = smf.ols(formula, df3).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             Q("BOISE")   R-squared:                       0.711
    Model:                            OLS   Adj. R-squared:                  0.701
    Method:                 Least Squares   F-statistic:                     71.25
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           2.66e-09
    Time:                        14:38:32   Log-Likelihood:                 39.332
    No. Observations:                  31   AIC:                            -74.66
    Df Residuals:                      29   BIC:                            -71.80
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0134      0.013     -1.051      0.302      -0.039       0.013
    Q("MARKET")     0.9989      0.118      8.441      0.000       0.757       1.241
    ==============================================================================
    Omnibus:                        1.931   Durbin-Watson:                   2.237
    Prob(Omnibus):                  0.381   Jarque-Bera (JB):                0.823
    Skew:                           0.254   Prob(JB):                        0.662
    Kurtosis:                       3.616   Cond. No.                         9.37
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
df = pd.read_excel('Data_For_Analysis.xlsx')
df.set_index('Date', inplace=True)
df.tail(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.041</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>0.030</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>0.067</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>-0.029</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.229</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.161</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>0.111</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>-0.021</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>0.035</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>-0.052</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>-0.060</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>-0.015</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
    <tr>
      <th>1981-10-01</th>
      <td>0.016</td>
      <td>0.052</td>
      <td>0.112</td>
      <td>0.049</td>
      <td>0.094</td>
      <td>0.102</td>
      <td>-0.072</td>
      <td>0.067</td>
      <td>-0.012</td>
      <td>-0.048</td>
      <td>...</td>
      <td>0.008</td>
      <td>-0.020</td>
      <td>31.00</td>
      <td>149.1</td>
      <td>279.9</td>
      <td>0.10256</td>
      <td>0.06701</td>
      <td>0.06230</td>
      <td>-0.01430</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1981-11-01</th>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>0.075</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1981-12-01</th>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-01-01</th>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>0.119</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.086</td>
      <td>0.043</td>
      <td>0.046</td>
      <td>-0.050</td>
      <td>0.060</td>
      <td>-0.101</td>
      <td>0.021</td>
      <td>0.060</td>
      <td>-0.031</td>
      <td>-0.038</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.004</td>
      <td>24.03</td>
      <td>166.5</td>
      <td>322.3</td>
      <td>0.00893</td>
      <td>0.06471</td>
      <td>-0.03727</td>
      <td>-0.01393</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>-0.026</td>
      <td>-0.030</td>
      <td>-0.084</td>
      <td>0.018</td>
      <td>0.043</td>
      <td>0.080</td>
      <td>0.008</td>
      <td>-0.099</td>
      <td>-0.036</td>
      <td>0.062</td>
      <td>...</td>
      <td>-0.030</td>
      <td>0.020</td>
      <td>24.00</td>
      <td>166.2</td>
      <td>322.8</td>
      <td>-0.06195</td>
      <td>0.03147</td>
      <td>0.03011</td>
      <td>-0.00816</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.011</td>
      <td>-0.063</td>
      <td>0.043</td>
      <td>-0.052</td>
      <td>-0.006</td>
      <td>0.032</td>
      <td>-0.066</td>
      <td>0.002</td>
      <td>0.025</td>
      <td>-0.028</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.013</td>
      <td>23.92</td>
      <td>167.7</td>
      <td>323.5</td>
      <td>0.08491</td>
      <td>-0.02712</td>
      <td>-0.02296</td>
      <td>0.03275</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>-0.095</td>
      <td>-0.085</td>
      <td>-0.032</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.112</td>
      <td>0.081</td>
      <td>-0.048</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.007</td>
      <td>-0.074</td>
      <td>23.93</td>
      <td>167.6</td>
      <td>324.5</td>
      <td>-0.04348</td>
      <td>-0.02927</td>
      <td>-0.00649</td>
      <td>-0.01440</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>-0.035</td>
      <td>0.090</td>
      <td>0.066</td>
      <td>0.105</td>
      <td>0.032</td>
      <td>0.040</td>
      <td>-0.083</td>
      <td>0.013</td>
      <td>0.097</td>
      <td>0.048</td>
      <td>...</td>
      <td>0.099</td>
      <td>0.008</td>
      <td>24.06</td>
      <td>166.6</td>
      <td>325.5</td>
      <td>0.14545</td>
      <td>0.06182</td>
      <td>0.08497</td>
      <td>0.01663</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.088</td>
      <td>0.062</td>
      <td>0.032</td>
      <td>0.048</td>
      <td>0.109</td>
      <td>0.073</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.137</td>
      <td>0.085</td>
      <td>...</td>
      <td>-0.175</td>
      <td>0.171</td>
      <td>24.31</td>
      <td>167.6</td>
      <td>326.6</td>
      <td>0.03175</td>
      <td>0.06507</td>
      <td>0.03012</td>
      <td>-0.00413</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.064</td>
      <td>0.065</td>
      <td>0.082</td>
      <td>0.197</td>
      <td>0.023</td>
      <td>0.095</td>
      <td>0.030</td>
      <td>0.027</td>
      <td>0.063</td>
      <td>0.113</td>
      <td>...</td>
      <td>-0.077</td>
      <td>-0.004</td>
      <td>24.53</td>
      <td>168.8</td>
      <td>327.4</td>
      <td>0.04615</td>
      <td>0.06624</td>
      <td>0.07101</td>
      <td>-0.02318</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.032</td>
      <td>0.005</td>
      <td>0.022</td>
      <td>0.000</td>
      <td>-0.055</td>
      <td>0.162</td>
      <td>0.122</td>
      <td>0.019</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>...</td>
      <td>-0.038</td>
      <td>0.072</td>
      <td>23.12</td>
      <td>169.6</td>
      <td>328.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>0.093</td>
      <td>0.101</td>
      <td>0.048</td>
      <td>-0.051</td>
      <td>-0.044</td>
      <td>0.093</td>
      <td>-0.055</td>
      <td>0.121</td>
      <td>0.034</td>
      <td>0.003</td>
      <td>...</td>
      <td>0.071</td>
      <td>0.123</td>
      <td>17.65</td>
      <td>168.4</td>
      <td>327.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>0.066</td>
      <td>0.153</td>
      <td>0.021</td>
      <td>-0.040</td>
      <td>-0.043</td>
      <td>-0.063</td>
      <td>0.076</td>
      <td>0.072</td>
      <td>0.174</td>
      <td>0.004</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.051</td>
      <td>12.62</td>
      <td>166.1</td>
      <td>326.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.013</td>
      <td>-0.042</td>
      <td>-0.006</td>
      <td>-0.097</td>
      <td>0.061</td>
      <td>0.119</td>
      <td>0.059</td>
      <td>-0.051</td>
      <td>0.113</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.050</td>
      <td>-0.037</td>
      <td>10.68</td>
      <td>167.6</td>
      <td>325.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-05-01</th>
      <td>0.072</td>
      <td>0.038</td>
      <td>0.042</td>
      <td>-0.046</td>
      <td>-0.015</td>
      <td>0.037</td>
      <td>-0.043</td>
      <td>0.109</td>
      <td>-0.040</td>
      <td>-0.018</td>
      <td>...</td>
      <td>0.069</td>
      <td>0.010</td>
      <td>10.75</td>
      <td>166.9</td>
      <td>326.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-06-01</th>
      <td>-0.013</td>
      <td>-0.036</td>
      <td>0.017</td>
      <td>-0.161</td>
      <td>-0.155</td>
      <td>-0.063</td>
      <td>-0.070</td>
      <td>0.071</td>
      <td>-0.038</td>
      <td>-0.039</td>
      <td>...</td>
      <td>-0.042</td>
      <td>-0.061</td>
      <td>10.68</td>
      <td>166.9</td>
      <td>327.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-07-01</th>
      <td>-0.060</td>
      <td>-0.117</td>
      <td>0.125</td>
      <td>-0.038</td>
      <td>-0.072</td>
      <td>0.066</td>
      <td>0.018</td>
      <td>0.049</td>
      <td>-0.105</td>
      <td>-0.096</td>
      <td>...</td>
      <td>-0.036</td>
      <td>-0.048</td>
      <td>9.25</td>
      <td>167.9</td>
      <td>328.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-08-01</th>
      <td>0.115</td>
      <td>0.082</td>
      <td>0.061</td>
      <td>-0.040</td>
      <td>0.167</td>
      <td>0.105</td>
      <td>0.018</td>
      <td>0.003</td>
      <td>0.111</td>
      <td>0.055</td>
      <td>...</td>
      <td>0.135</td>
      <td>0.122</td>
      <td>9.77</td>
      <td>168.1</td>
      <td>328.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-09-01</th>
      <td>-0.052</td>
      <td>-0.111</td>
      <td>-0.139</td>
      <td>0.021</td>
      <td>-0.240</td>
      <td>-0.110</td>
      <td>0.026</td>
      <td>-0.088</td>
      <td>0.037</td>
      <td>-0.031</td>
      <td>...</td>
      <td>0.026</td>
      <td>-0.058</td>
      <td>11.09</td>
      <td>167.9</td>
      <td>330.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-10-01</th>
      <td>0.059</td>
      <td>0.040</td>
      <td>0.045</td>
      <td>0.000</td>
      <td>0.105</td>
      <td>0.103</td>
      <td>0.134</td>
      <td>0.123</td>
      <td>-0.069</td>
      <td>-0.081</td>
      <td>...</td>
      <td>0.043</td>
      <td>0.135</td>
      <td>11.00</td>
      <td>168.4</td>
      <td>330.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-11-01</th>
      <td>0.023</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.143</td>
      <td>0.020</td>
      <td>0.048</td>
      <td>-0.018</td>
      <td>0.011</td>
      <td>-0.020</td>
      <td>0.037</td>
      <td>...</td>
      <td>-0.028</td>
      <td>0.006</td>
      <td>11.05</td>
      <td>169.3</td>
      <td>330.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-12-01</th>
      <td>-0.027</td>
      <td>0.019</td>
      <td>-0.046</td>
      <td>0.028</td>
      <td>-0.078</td>
      <td>0.008</td>
      <td>-0.010</td>
      <td>-0.034</td>
      <td>-0.060</td>
      <td>-0.056</td>
      <td>...</td>
      <td>0.047</td>
      <td>-0.041</td>
      <td>11.75</td>
      <td>170.2</td>
      <td>331.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-01-01</th>
      <td>0.276</td>
      <td>0.087</td>
      <td>0.040</td>
      <td>0.093</td>
      <td>0.135</td>
      <td>0.385</td>
      <td>0.161</td>
      <td>0.123</td>
      <td>0.057</td>
      <td>0.073</td>
      <td>...</td>
      <td>0.049</td>
      <td>0.270</td>
      <td>13.89</td>
      <td>169.6</td>
      <td>333.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-02-01</th>
      <td>-0.008</td>
      <td>-0.066</td>
      <td>-0.067</td>
      <td>-0.064</td>
      <td>0.045</td>
      <td>0.056</td>
      <td>0.133</td>
      <td>0.049</td>
      <td>0.019</td>
      <td>0.092</td>
      <td>...</td>
      <td>-0.080</td>
      <td>0.094</td>
      <td>14.50</td>
      <td>170.8</td>
      <td>334.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-03-01</th>
      <td>0.071</td>
      <td>-0.052</td>
      <td>-0.050</td>
      <td>-0.087</td>
      <td>-0.096</td>
      <td>0.061</td>
      <td>-0.129</td>
      <td>0.010</td>
      <td>0.040</td>
      <td>0.076</td>
      <td>...</td>
      <td>0.103</td>
      <td>0.089</td>
      <td>14.53</td>
      <td>171.2</td>
      <td>335.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-04-01</th>
      <td>-0.037</td>
      <td>0.070</td>
      <td>0.020</td>
      <td>-0.025</td>
      <td>-0.020</td>
      <td>0.055</td>
      <td>-0.121</td>
      <td>-0.104</td>
      <td>-0.063</td>
      <td>0.067</td>
      <td>...</td>
      <td>-0.094</td>
      <td>-0.027</td>
      <td>14.95</td>
      <td>171.2</td>
      <td>337.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-05-01</th>
      <td>-0.111</td>
      <td>0.052</td>
      <td>-0.012</td>
      <td>0.000</td>
      <td>0.161</td>
      <td>-0.082</td>
      <td>0.151</td>
      <td>0.190</td>
      <td>0.138</td>
      <td>0.006</td>
      <td>...</td>
      <td>0.114</td>
      <td>-0.107</td>
      <td>15.29</td>
      <td>172.3</td>
      <td>338.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-06-01</th>
      <td>0.063</td>
      <td>0.051</td>
      <td>0.059</td>
      <td>0.081</td>
      <td>-0.145</td>
      <td>0.041</td>
      <td>0.014</td>
      <td>0.030</td>
      <td>0.005</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.073</td>
      <td>0.026</td>
      <td>15.95</td>
      <td>173.5</td>
      <td>340.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-07-01</th>
      <td>0.064</td>
      <td>0.041</td>
      <td>-0.039</td>
      <td>0.071</td>
      <td>0.057</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.036</td>
      <td>0.232</td>
      <td>-0.009</td>
      <td>...</td>
      <td>0.142</td>
      <td>0.021</td>
      <td>16.88</td>
      <td>175.5</td>
      <td>340.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-08-01</th>
      <td>0.061</td>
      <td>0.033</td>
      <td>0.043</td>
      <td>-0.044</td>
      <td>-0.008</td>
      <td>0.157</td>
      <td>-0.037</td>
      <td>0.022</td>
      <td>-0.113</td>
      <td>0.053</td>
      <td>...</td>
      <td>-0.076</td>
      <td>0.081</td>
      <td>17.06</td>
      <td>176.3</td>
      <td>342.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-09-01</th>
      <td>-0.029</td>
      <td>-0.086</td>
      <td>-0.006</td>
      <td>0.004</td>
      <td>0.015</td>
      <td>0.001</td>
      <td>-0.067</td>
      <td>-0.009</td>
      <td>-0.061</td>
      <td>-0.105</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.054</td>
      <td>16.29</td>
      <td>176.1</td>
      <td>344.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-10-01</th>
      <td>-0.274</td>
      <td>-0.282</td>
      <td>-0.017</td>
      <td>-0.372</td>
      <td>-0.342</td>
      <td>-0.281</td>
      <td>-0.260</td>
      <td>-0.148</td>
      <td>-0.288</td>
      <td>-0.187</td>
      <td>...</td>
      <td>-0.194</td>
      <td>-0.271</td>
      <td>15.95</td>
      <td>178.1</td>
      <td>345.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1987-11-01</th>
      <td>0.043</td>
      <td>-0.136</td>
      <td>-0.012</td>
      <td>-0.148</td>
      <td>-0.075</td>
      <td>-0.127</td>
      <td>-0.137</td>
      <td>-0.102</td>
      <td>-0.085</td>
      <td>-0.087</td>
      <td>...</td>
      <td>-0.031</td>
      <td>-0.066</td>
      <td>15.46</td>
      <td>179.0</td>
      <td>345.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 27 columns</p>
</div>




```python
hypotheses = 'Q("MARKET")=0'
Simple_ttest_Ols(results, hypotheses, alternative='larger', level_of_sig = 0.05)
```

    We reject the null hypothesis: Q("MARKET")=0 with a 5.0 % significance level



```python
hypotheses = 'Q("MARKET")=1'
Simple_ttest_Ols(results, hypotheses, alternative='larger', level_of_sig = 0.05)
```

    We accept the null hypothesis: Q("MARKET")=1 with a 5.0 % significance level



```python
hypotheses = 'Q("MARKET")=1'
Simple_ttest_Ols(results, hypotheses, alternative='larger', level_of_sig = 0.05)
```

    We accept the null hypothesis: Q("MARKET")=1 with a 5.0 % significance level



```python
formula = 'CONTIL ~ MARKET'
results = smf.ols(formula, df1).fit()
std_residuals_CONTIL=results.resid
print(std_residuals_CONTIL.std())
```

    0.06839421589721249



```python
formula = 'BOISE ~ MARKET'
results = smf.ols(formula, df1).fit()
std_residuals_BOISE=results.resid
print(std_residuals_BOISE.std())
```

    0.06435324835690541



```python
df1.head(200)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
      <th>Group column</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.034</td>
      <td>0.065</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>-0.018</td>
      <td>0.025</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>0.034</td>
      <td>0.015</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>0.035</td>
      <td>0.006</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>-0.017</td>
      <td>0.092</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.103</td>
      <td>-0.056</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>0.040</td>
      <td>-0.014</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>0.069</td>
      <td>-0.009</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.024</td>
      <td>0.067</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>-0.025</td>
      <td>-0.008</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.117</td>
      <td>0.064</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>0.077</td>
      <td>-0.003</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.092</td>
      <td>-0.033</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.030</td>
      <td>-0.031</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.003</td>
      <td>-0.164</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.head(200)
df.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.032</td>
      <td>0.011</td>
      <td>0.066</td>
      <td>0.143</td>
      <td>0.107</td>
      <td>0.185</td>
      <td>0.075</td>
      <td>-0.012</td>
      <td>0.092</td>
      <td>...</td>
      <td>0.042</td>
      <td>0.164</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>196.7</td>
      <td>0.04405</td>
      <td>0.07107</td>
      <td>0.07813</td>
      <td>0.02814</td>
      <td>-0.01422</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.088</td>
      <td>0.024</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>-0.017</td>
      <td>-0.021</td>
      <td>-0.051</td>
      <td>-0.079</td>
      <td>0.049</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.039</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>197.8</td>
      <td>-0.04636</td>
      <td>0.04265</td>
      <td>0.03727</td>
      <td>0.09005</td>
      <td>0.09519</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.011</td>
      <td>0.048</td>
      <td>-0.013</td>
      <td>-0.031</td>
      <td>-0.037</td>
      <td>-0.081</td>
      <td>-0.012</td>
      <td>0.104</td>
      <td>-0.051</td>
      <td>...</td>
      <td>0.010</td>
      <td>-0.021</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>199.3</td>
      <td>0.03472</td>
      <td>0.04000</td>
      <td>0.03024</td>
      <td>0.02977</td>
      <td>0.04889</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.071</td>
      <td>-0.067</td>
      <td>-0.123</td>
      <td>-0.085</td>
      <td>-0.077</td>
      <td>-0.153</td>
      <td>-0.032</td>
      <td>-0.138</td>
      <td>-0.046</td>
      <td>...</td>
      <td>-0.066</td>
      <td>-0.090</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>200.9</td>
      <td>-0.07651</td>
      <td>-0.07522</td>
      <td>-0.06067</td>
      <td>0.07194</td>
      <td>-0.13136</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.005</td>
      <td>0.035</td>
      <td>-0.038</td>
      <td>0.044</td>
      <td>0.064</td>
      <td>0.055</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>202.0</td>
      <td>0.04478</td>
      <td>0.00478</td>
      <td>0.05000</td>
      <td>-0.09443</td>
      <td>0.02927</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.047</td>
      <td>0.034</td>
      <td>0.117</td>
      <td>-0.023</td>
      <td>0.022</td>
      <td>-0.086</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.000</td>
      <td>-0.034</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>203.3</td>
      <td>0.00000</td>
      <td>-0.03905</td>
      <td>0.02857</td>
      <td>0.00941</td>
      <td>0.08173</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>0.043</td>
      <td>0.076</td>
      <td>-0.024</td>
      <td>-0.008</td>
      <td>-0.012</td>
      <td>-0.054</td>
      <td>-0.032</td>
      <td>0.042</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.037</td>
      <td>0.203</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>204.7</td>
      <td>0.05429</td>
      <td>0.07035</td>
      <td>0.06151</td>
      <td>0.09336</td>
      <td>0.06667</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.082</td>
      <td>-0.011</td>
      <td>-0.020</td>
      <td>-0.015</td>
      <td>-0.066</td>
      <td>-0.060</td>
      <td>-0.079</td>
      <td>-0.023</td>
      <td>-0.017</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.038</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>207.1</td>
      <td>-0.05556</td>
      <td>-0.04225</td>
      <td>-0.02150</td>
      <td>0.08042</td>
      <td>0.02500</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.171</td>
      <td>0.088</td>
      <td>0.098</td>
      <td>-0.043</td>
      <td>0.065</td>
      <td>0.052</td>
      <td>...</td>
      <td>0.068</td>
      <td>0.097</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>209.1</td>
      <td>-0.04412</td>
      <td>0.11176</td>
      <td>0.09082</td>
      <td>-0.01428</td>
      <td>0.12757</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.000</td>
      <td>-0.057</td>
      <td>0.064</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>-0.056</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.069</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>211.5</td>
      <td>-0.33077</td>
      <td>-0.07143</td>
      <td>-0.06200</td>
      <td>-0.01333</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.005</td>
      <td>-0.045</td>
      <td>-0.028</td>
      <td>0.063</td>
      <td>0.035</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.040</td>
      <td>-0.013</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>214.1</td>
      <td>-0.17241</td>
      <td>-0.01923</td>
      <td>-0.03305</td>
      <td>0.07745</td>
      <td>0.00511</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.095</td>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>-0.006</td>
      <td>-0.043</td>
      <td>0.095</td>
      <td>-0.035</td>
      <td>...</td>
      <td>0.083</td>
      <td>0.053</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>216.6</td>
      <td>0.15714</td>
      <td>0.02843</td>
      <td>-0.02174</td>
      <td>0.08434</td>
      <td>0.11397</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.075</td>
      <td>0.015</td>
      <td>-0.034</td>
      <td>-0.059</td>
      <td>0.009</td>
      <td>0.075</td>
      <td>-0.013</td>
      <td>-0.096</td>
      <td>-0.049</td>
      <td>...</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>218.9</td>
      <td>-0.01235</td>
      <td>0.08213</td>
      <td>-0.00909</td>
      <td>0.05802</td>
      <td>0.01980</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.041</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>0.030</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>0.067</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>-0.029</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.229</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.161</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>0.111</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>-0.021</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>0.035</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>-0.052</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>-0.060</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>-0.015</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
  </tbody>
</table>
<p>45 rows × 27 columns</p>
</div>




```python
formula_with_intercept = 'MARKET ~ BOISE + CONTIL'

results_with_intercept = smf.ols(formula_with_intercept, df1).fit()
print(results_with_intercept.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 MARKET   R-squared:                       0.400
    Model:                            OLS   Adj. R-squared:                  0.372
    Method:                 Least Squares   F-statistic:                     14.03
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           2.16e-05
    Time:                        14:38:45   Log-Likelihood:                 62.669
    No. Observations:                  45   AIC:                            -119.3
    Df Residuals:                      42   BIC:                            -113.9
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.0146      0.009      1.543      0.130      -0.005       0.034
    BOISE          0.6088      0.124      4.927      0.000       0.359       0.858
    CONTIL        -0.0327      0.146     -0.224      0.824      -0.327       0.262
    ==============================================================================
    Omnibus:                        6.784   Durbin-Watson:                   2.511
    Prob(Omnibus):                  0.034   Jarque-Bera (JB):                5.819
    Skew:                          -0.662   Prob(JB):                       0.0545
    Kurtosis:                       4.161   Cond. No.                         17.5
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
R=GQTest(results_with_intercept)
```

    The P vale of this test is 0.01935, which is smaller than the level of significance 0.05 therefore, we reject the null, hence the error terms are hetroscedastic



```python
formula_without_intercept = 'MARKET ~ BOISE + CONTIL '

results_without_intercept = smf.ols(formula_without_intercept, df1).fit()
print(results_without_intercept.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 MARKET   R-squared:                       0.400
    Model:                            OLS   Adj. R-squared:                  0.372
    Method:                 Least Squares   F-statistic:                     14.03
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           2.16e-05
    Time:                        14:38:45   Log-Likelihood:                 62.669
    No. Observations:                  45   AIC:                            -119.3
    Df Residuals:                      42   BIC:                            -113.9
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.0146      0.009      1.543      0.130      -0.005       0.034
    BOISE          0.6088      0.124      4.927      0.000       0.359       0.858
    CONTIL        -0.0327      0.146     -0.224      0.824      -0.327       0.262
    ==============================================================================
    Omnibus:                        6.784   Durbin-Watson:                   2.511
    Prob(Omnibus):                  0.034   Jarque-Bera (JB):                5.819
    Skew:                          -0.662   Prob(JB):                       0.0545
    Kurtosis:                       4.161   Cond. No.                         17.5
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
R=GQTest(results_without_intercept)
```

    The P vale of this test is 0.01935, which is smaller than the level of significance 0.05 therefore, we reject the null, hence the error terms are hetroscedastic



```python
R=WhiteTest(results_with_intercept)
```

    The P vale of this test is 0.02118, which is smaller than the level of significance 0.05 therefore, we reject the null, hence the error terms are hetroscedastic



```python
resid_model1=results_with_intercept.resid
resid_model1
```




    Date
    1978-01-01   -0.015748
    1978-02-01   -0.011327
    1978-03-01   -0.007143
    1978-04-01   -0.018791
    1978-05-01    0.011147
    1978-06-01    0.050110
    1978-07-01   -0.026699
    1978-08-01    0.017968
    1978-09-01    0.022872
    1978-10-01   -0.135808
    1978-11-01    0.104662
    1978-12-01   -0.038876
    1979-01-01   -0.059692
    1979-02-01    0.015204
    1979-03-01    0.001413
    1979-04-01    0.039651
    1979-05-01    0.015370
    1979-06-01    0.028685
    1979-07-01   -0.057351
    1979-08-01    0.059140
    1979-09-01   -0.007145
    1979-10-01   -0.024142
    1979-11-01    0.106463
    1979-12-01    0.049210
    1980-01-01    0.040861
    1980-02-01    0.137147
    1980-03-01   -0.170823
    1980-04-01    0.042228
    1980-05-01   -0.018233
    1980-06-01    0.032071
    1980-07-01    0.004819
    1980-08-01    0.037185
    1980-09-01   -0.010078
    1980-10-01    0.025399
    1980-11-01    0.059773
    1980-12-01   -0.038637
    1981-01-01   -0.034011
    1981-02-01   -0.113907
    1981-03-01    0.019068
    1981-04-01   -0.050839
    1981-05-01    0.033724
    1981-06-01    0.007423
    1981-07-01   -0.011061
    1981-08-01    0.029498
    1981-09-01   -0.140778
    dtype: float64




```python
formula = 'Q("CONTIL") ~ Q("MARKET")'

results = smf.ols(formula, df).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CONTIL")   R-squared:                       0.111
    Model:                            OLS   Adj. R-squared:                  0.104
    Method:                 Least Squares   F-statistic:                     14.65
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           0.000209
    Time:                        14:38:46   Log-Likelihood:                 63.430
    No. Observations:                 119   AIC:                            -122.9
    Df Residuals:                     117   BIC:                            -117.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0115      0.013     -0.858      0.393      -0.038       0.015
    Q("MARKET")     0.7375      0.193      3.828      0.000       0.356       1.119
    ==============================================================================
    Omnibus:                      106.122   Durbin-Watson:                   2.068
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2224.184
    Skew:                           2.697   Prob(JB):                         0.00
    Kurtosis:                      23.481   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
Plot_resi_corr(results)
```


![png](output_67_0.png)



```python
Plot_resi_corr_time(results,df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Residuals</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976-01-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-02-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-03-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-04-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-05-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-06-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-07-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-08-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-09-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-10-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-11-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1976-12-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-01-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-02-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-03-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-04-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-05-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-06-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-07-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-08-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-09-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-10-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-11-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1977-12-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1978-01-01</th>
      <td>-0.084327</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.041108</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>-0.022394</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.145018</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.023068</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.052680</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>-0.036304</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>0.020633</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>-0.044205</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>0.088048</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>0.097307</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.015968</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.198895</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.018121</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>-0.075657</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>-0.063919</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.078879</td>
    </tr>
    <tr>
      <th>1986-05-01</th>
      <td>-0.070657</td>
    </tr>
    <tr>
      <th>1986-06-01</th>
      <td>-0.152467</td>
    </tr>
    <tr>
      <th>1986-07-01</th>
      <td>0.029537</td>
    </tr>
    <tr>
      <th>1986-08-01</th>
      <td>-0.064657</td>
    </tr>
    <tr>
      <th>1986-09-01</th>
      <td>0.067148</td>
    </tr>
    <tr>
      <th>1986-10-01</th>
      <td>-0.001793</td>
    </tr>
    <tr>
      <th>1986-11-01</th>
      <td>-0.131517</td>
    </tr>
    <tr>
      <th>1986-12-01</th>
      <td>0.043171</td>
    </tr>
    <tr>
      <th>1987-01-01</th>
      <td>-0.004674</td>
    </tr>
    <tr>
      <th>1987-02-01</th>
      <td>-0.100457</td>
    </tr>
    <tr>
      <th>1987-03-01</th>
      <td>-0.102806</td>
    </tr>
    <tr>
      <th>1987-04-01</th>
      <td>0.004922</td>
    </tr>
    <tr>
      <th>1987-05-01</th>
      <td>0.008533</td>
    </tr>
    <tr>
      <th>1987-06-01</th>
      <td>0.064456</td>
    </tr>
    <tr>
      <th>1987-07-01</th>
      <td>0.041918</td>
    </tr>
    <tr>
      <th>1987-08-01</th>
      <td>-0.043580</td>
    </tr>
    <tr>
      <th>1987-09-01</th>
      <td>0.026546</td>
    </tr>
    <tr>
      <th>1987-10-01</th>
      <td>-0.168755</td>
    </tr>
    <tr>
      <th>1987-11-01</th>
      <td>-0.084889</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 1 columns</p>
</div>




![png](output_68_1.png)



```python
import statsmodels.stats.stattools as tools
tools.durbin_watson(results.resid)
```




    2.06843613615565




```python
Breusch_Godfrey(results, lags=6, level_of_sig=0.1)
```

    The P vale of this test is 0.81024, which is greater than the level of significance 0.1 therefore, we accept the null that the error terms are not Auto-corrolated





    {'LM Statistic': 3.10673167028625,
     'LM-Test p-value': 0.7953366307235248,
     'F-Statistic': 0.4959264392887972,
     'F-Test p-value': 0.8102445238729414}




```python
# Break the data into two sets
start_date = dt.datetime(1978,1,1)
end_date = dt.datetime(1987,12,1)

select = (df.index>=start_date)*(df.index<=end_date)

# Copy the selected dataframe into df1
df1=df[select].copy()
df1.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.032</td>
      <td>0.011</td>
      <td>0.066</td>
      <td>0.143</td>
      <td>0.107</td>
      <td>0.185</td>
      <td>0.075</td>
      <td>-0.012</td>
      <td>0.092</td>
      <td>...</td>
      <td>0.042</td>
      <td>0.164</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>196.7</td>
      <td>0.04405</td>
      <td>0.07107</td>
      <td>0.07813</td>
      <td>0.02814</td>
      <td>-0.01422</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.088</td>
      <td>0.024</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>-0.017</td>
      <td>-0.021</td>
      <td>-0.051</td>
      <td>-0.079</td>
      <td>0.049</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.039</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>197.8</td>
      <td>-0.04636</td>
      <td>0.04265</td>
      <td>0.03727</td>
      <td>0.09005</td>
      <td>0.09519</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.011</td>
      <td>0.048</td>
      <td>-0.013</td>
      <td>-0.031</td>
      <td>-0.037</td>
      <td>-0.081</td>
      <td>-0.012</td>
      <td>0.104</td>
      <td>-0.051</td>
      <td>...</td>
      <td>0.010</td>
      <td>-0.021</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>199.3</td>
      <td>0.03472</td>
      <td>0.04000</td>
      <td>0.03024</td>
      <td>0.02977</td>
      <td>0.04889</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.071</td>
      <td>-0.067</td>
      <td>-0.123</td>
      <td>-0.085</td>
      <td>-0.077</td>
      <td>-0.153</td>
      <td>-0.032</td>
      <td>-0.138</td>
      <td>-0.046</td>
      <td>...</td>
      <td>-0.066</td>
      <td>-0.090</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>200.9</td>
      <td>-0.07651</td>
      <td>-0.07522</td>
      <td>-0.06067</td>
      <td>0.07194</td>
      <td>-0.13136</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.005</td>
      <td>0.035</td>
      <td>-0.038</td>
      <td>0.044</td>
      <td>0.064</td>
      <td>0.055</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>202.0</td>
      <td>0.04478</td>
      <td>0.00478</td>
      <td>0.05000</td>
      <td>-0.09443</td>
      <td>0.02927</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.047</td>
      <td>0.034</td>
      <td>0.117</td>
      <td>-0.023</td>
      <td>0.022</td>
      <td>-0.086</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.000</td>
      <td>-0.034</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>203.3</td>
      <td>0.00000</td>
      <td>-0.03905</td>
      <td>0.02857</td>
      <td>0.00941</td>
      <td>0.08173</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>0.043</td>
      <td>0.076</td>
      <td>-0.024</td>
      <td>-0.008</td>
      <td>-0.012</td>
      <td>-0.054</td>
      <td>-0.032</td>
      <td>0.042</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.037</td>
      <td>0.203</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>204.7</td>
      <td>0.05429</td>
      <td>0.07035</td>
      <td>0.06151</td>
      <td>0.09336</td>
      <td>0.06667</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.082</td>
      <td>-0.011</td>
      <td>-0.020</td>
      <td>-0.015</td>
      <td>-0.066</td>
      <td>-0.060</td>
      <td>-0.079</td>
      <td>-0.023</td>
      <td>-0.017</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.038</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>207.1</td>
      <td>-0.05556</td>
      <td>-0.04225</td>
      <td>-0.02150</td>
      <td>0.08042</td>
      <td>0.02500</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.171</td>
      <td>0.088</td>
      <td>0.098</td>
      <td>-0.043</td>
      <td>0.065</td>
      <td>0.052</td>
      <td>...</td>
      <td>0.068</td>
      <td>0.097</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>209.1</td>
      <td>-0.04412</td>
      <td>0.11176</td>
      <td>0.09082</td>
      <td>-0.01428</td>
      <td>0.12757</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.000</td>
      <td>-0.057</td>
      <td>0.064</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>-0.056</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.069</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>211.5</td>
      <td>-0.33077</td>
      <td>-0.07143</td>
      <td>-0.06200</td>
      <td>-0.01333</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.005</td>
      <td>-0.045</td>
      <td>-0.028</td>
      <td>0.063</td>
      <td>0.035</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.040</td>
      <td>-0.013</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>214.1</td>
      <td>-0.17241</td>
      <td>-0.01923</td>
      <td>-0.03305</td>
      <td>0.07745</td>
      <td>0.00511</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.095</td>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>-0.006</td>
      <td>-0.043</td>
      <td>0.095</td>
      <td>-0.035</td>
      <td>...</td>
      <td>0.083</td>
      <td>0.053</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>216.6</td>
      <td>0.15714</td>
      <td>0.02843</td>
      <td>-0.02174</td>
      <td>0.08434</td>
      <td>0.11397</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.075</td>
      <td>0.015</td>
      <td>-0.034</td>
      <td>-0.059</td>
      <td>0.009</td>
      <td>0.075</td>
      <td>-0.013</td>
      <td>-0.096</td>
      <td>-0.049</td>
      <td>...</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>218.9</td>
      <td>-0.01235</td>
      <td>0.08213</td>
      <td>-0.00909</td>
      <td>0.05802</td>
      <td>0.01980</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.041</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>0.030</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>0.067</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>-0.029</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.229</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.161</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.147</td>
      <td>0.162</td>
      <td>-0.025</td>
      <td>0.096</td>
      <td>-0.014</td>
      <td>0.065</td>
      <td>0.120</td>
      <td>-0.014</td>
      <td>0.077</td>
      <td>-0.066</td>
      <td>...</td>
      <td>0.011</td>
      <td>0.151</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>303.1</td>
      <td>0.06667</td>
      <td>-0.05072</td>
      <td>0.06214</td>
      <td>-0.02957</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.012</td>
      <td>0.023</td>
      <td>0.005</td>
      <td>-0.016</td>
      <td>0.068</td>
      <td>0.034</td>
      <td>-0.028</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.069</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>303.5</td>
      <td>-0.03125</td>
      <td>0.03282</td>
      <td>-0.03704</td>
      <td>0.01488</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.054</td>
      <td>0.024</td>
      <td>0.005</td>
      <td>-0.034</td>
      <td>0.117</td>
      <td>0.208</td>
      <td>-0.013</td>
      <td>-0.009</td>
      <td>0.065</td>
      <td>-0.065</td>
      <td>...</td>
      <td>0.108</td>
      <td>-0.039</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>305.4</td>
      <td>-0.01613</td>
      <td>-0.05243</td>
      <td>-0.04327</td>
      <td>-0.04322</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.088</td>
      <td>-0.039</td>
      <td>-0.069</td>
      <td>-0.101</td>
      <td>0.027</td>
      <td>-0.024</td>
      <td>-0.117</td>
      <td>-0.073</td>
      <td>-0.091</td>
      <td>-0.026</td>
      <td>...</td>
      <td>0.151</td>
      <td>-0.093</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>306.6</td>
      <td>0.03279</td>
      <td>-0.11462</td>
      <td>-0.03367</td>
      <td>0.04059</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>0.079</td>
      <td>-0.054</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>0.056</td>
      <td>0.057</td>
      <td>0.065</td>
      <td>-0.018</td>
      <td>-0.003</td>
      <td>0.034</td>
      <td>...</td>
      <td>-0.122</td>
      <td>0.094</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>307.3</td>
      <td>0.04762</td>
      <td>0.15000</td>
      <td>0.03958</td>
      <td>0.02159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>0.012</td>
      <td>-0.004</td>
      <td>0.031</td>
      <td>-0.231</td>
      <td>0.089</td>
      <td>0.053</td>
      <td>-0.085</td>
      <td>0.065</td>
      <td>-0.025</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>308.8</td>
      <td>-0.01515</td>
      <td>0.01969</td>
      <td>0.02538</td>
      <td>-0.03213</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-05-01</th>
      <td>-0.172</td>
      <td>-0.148</td>
      <td>0.021</td>
      <td>-0.600</td>
      <td>-0.094</td>
      <td>-0.071</td>
      <td>-0.070</td>
      <td>0.018</td>
      <td>-0.087</td>
      <td>-0.044</td>
      <td>...</td>
      <td>-0.105</td>
      <td>-0.087</td>
      <td>26.00</td>
      <td>162.8</td>
      <td>309.7</td>
      <td>0.07692</td>
      <td>-0.12355</td>
      <td>-0.05545</td>
      <td>-0.01126</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-06-01</th>
      <td>0.025</td>
      <td>0.078</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>-0.043</td>
      <td>-0.012</td>
      <td>0.055</td>
      <td>0.105</td>
      <td>-0.019</td>
      <td>...</td>
      <td>-0.046</td>
      <td>0.019</td>
      <td>26.09</td>
      <td>164.4</td>
      <td>310.7</td>
      <td>0.02857</td>
      <td>0.00264</td>
      <td>-0.02926</td>
      <td>0.00103</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-07-01</th>
      <td>0.015</td>
      <td>-0.029</td>
      <td>0.054</td>
      <td>-0.205</td>
      <td>-0.061</td>
      <td>-0.009</td>
      <td>0.045</td>
      <td>-0.018</td>
      <td>-0.112</td>
      <td>0.047</td>
      <td>...</td>
      <td>-0.044</td>
      <td>0.036</td>
      <td>26.11</td>
      <td>165.9</td>
      <td>311.7</td>
      <td>0.08333</td>
      <td>0.01339</td>
      <td>-0.03014</td>
      <td>-0.08266</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-08-01</th>
      <td>0.177</td>
      <td>0.164</td>
      <td>0.029</td>
      <td>0.086</td>
      <td>0.312</td>
      <td>0.159</td>
      <td>0.040</td>
      <td>0.061</td>
      <td>0.018</td>
      <td>0.127</td>
      <td>...</td>
      <td>0.140</td>
      <td>0.055</td>
      <td>26.02</td>
      <td>166.0</td>
      <td>313.0</td>
      <td>0.02564</td>
      <td>0.10132</td>
      <td>0.14689</td>
      <td>0.00360</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-09-01</th>
      <td>-0.056</td>
      <td>0.076</td>
      <td>0.051</td>
      <td>0.974</td>
      <td>-0.132</td>
      <td>-0.025</td>
      <td>0.008</td>
      <td>0.011</td>
      <td>0.165</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.069</td>
      <td>25.97</td>
      <td>165.0</td>
      <td>314.5</td>
      <td>0.01250</td>
      <td>-0.08160</td>
      <td>-0.00750</td>
      <td>-0.01942</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-10-01</th>
      <td>0.053</td>
      <td>-0.027</td>
      <td>0.019</td>
      <td>-0.232</td>
      <td>0.047</td>
      <td>0.093</td>
      <td>0.161</td>
      <td>-0.010</td>
      <td>-0.160</td>
      <td>0.012</td>
      <td>...</td>
      <td>-0.080</td>
      <td>0.035</td>
      <td>25.92</td>
      <td>164.5</td>
      <td>315.3</td>
      <td>0.08642</td>
      <td>0.02655</td>
      <td>-0.05542</td>
      <td>-0.00217</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-11-01</th>
      <td>-0.038</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>-0.023</td>
      <td>0.019</td>
      <td>0.006</td>
      <td>-0.026</td>
      <td>-0.072</td>
      <td>0.094</td>
      <td>-0.023</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.032</td>
      <td>25.44</td>
      <td>165.2</td>
      <td>315.3</td>
      <td>0.02273</td>
      <td>-0.00862</td>
      <td>0.00800</td>
      <td>0.00285</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-12-01</th>
      <td>0.068</td>
      <td>0.098</td>
      <td>0.084</td>
      <td>0.095</td>
      <td>0.096</td>
      <td>0.070</td>
      <td>0.156</td>
      <td>0.017</td>
      <td>-0.005</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.026</td>
      <td>25.05</td>
      <td>166.2</td>
      <td>315.5</td>
      <td>0.02222</td>
      <td>-0.02783</td>
      <td>0.06452</td>
      <td>-0.06479</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-01-01</th>
      <td>0.046</td>
      <td>0.097</td>
      <td>-0.021</td>
      <td>0.587</td>
      <td>0.215</td>
      <td>0.084</td>
      <td>-0.010</td>
      <td>0.095</td>
      <td>0.091</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.044</td>
      <td>0.084</td>
      <td>24.28</td>
      <td>165.6</td>
      <td>316.1</td>
      <td>0.00000</td>
      <td>0.07727</td>
      <td>0.04293</td>
      <td>-0.06265</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>-0.059</td>
      <td>-0.015</td>
      <td>0.034</td>
      <td>-0.096</td>
      <td>-0.210</td>
      <td>-0.067</td>
      <td>0.087</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>-0.009</td>
      <td>...</td>
      <td>0.022</td>
      <td>-0.016</td>
      <td>23.63</td>
      <td>165.7</td>
      <td>317.4</td>
      <td>0.08696</td>
      <td>0.00844</td>
      <td>0.03148</td>
      <td>0.01720</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>-0.029</td>
      <td>0.046</td>
      <td>0.057</td>
      <td>0.030</td>
      <td>-0.195</td>
      <td>-0.071</td>
      <td>-0.003</td>
      <td>0.054</td>
      <td>0.130</td>
      <td>-0.052</td>
      <td>...</td>
      <td>0.014</td>
      <td>-0.081</td>
      <td>23.88</td>
      <td>166.1</td>
      <td>318.8</td>
      <td>-0.04000</td>
      <td>-0.01423</td>
      <td>-0.01190</td>
      <td>-0.04400</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>0.010</td>
      <td>0.012</td>
      <td>0.019</td>
      <td>-0.029</td>
      <td>-0.157</td>
      <td>-0.050</td>
      <td>-0.123</td>
      <td>-0.083</td>
      <td>-0.037</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.111</td>
      <td>0.003</td>
      <td>24.15</td>
      <td>166.2</td>
      <td>320.1</td>
      <td>-0.01042</td>
      <td>0.03448</td>
      <td>0.06988</td>
      <td>0.13447</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.158</td>
      <td>0.094</td>
      <td>0.098</td>
      <td>-0.091</td>
      <td>-0.078</td>
      <td>0.057</td>
      <td>0.179</td>
      <td>0.137</td>
      <td>0.234</td>
      <td>0.025</td>
      <td>...</td>
      <td>-0.065</td>
      <td>0.031</td>
      <td>24.18</td>
      <td>166.2</td>
      <td>321.3</td>
      <td>0.17895</td>
      <td>0.13333</td>
      <td>0.10135</td>
      <td>-0.02165</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.086</td>
      <td>0.043</td>
      <td>0.046</td>
      <td>-0.050</td>
      <td>0.060</td>
      <td>-0.101</td>
      <td>0.021</td>
      <td>0.060</td>
      <td>-0.031</td>
      <td>-0.038</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.004</td>
      <td>24.03</td>
      <td>166.5</td>
      <td>322.3</td>
      <td>0.00893</td>
      <td>0.06471</td>
      <td>-0.03727</td>
      <td>-0.01393</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>-0.026</td>
      <td>-0.030</td>
      <td>-0.084</td>
      <td>0.018</td>
      <td>0.043</td>
      <td>0.080</td>
      <td>0.008</td>
      <td>-0.099</td>
      <td>-0.036</td>
      <td>0.062</td>
      <td>...</td>
      <td>-0.030</td>
      <td>0.020</td>
      <td>24.00</td>
      <td>166.2</td>
      <td>322.8</td>
      <td>-0.06195</td>
      <td>0.03147</td>
      <td>0.03011</td>
      <td>-0.00816</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.011</td>
      <td>-0.063</td>
      <td>0.043</td>
      <td>-0.052</td>
      <td>-0.006</td>
      <td>0.032</td>
      <td>-0.066</td>
      <td>0.002</td>
      <td>0.025</td>
      <td>-0.028</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.013</td>
      <td>23.92</td>
      <td>167.7</td>
      <td>323.5</td>
      <td>0.08491</td>
      <td>-0.02712</td>
      <td>-0.02296</td>
      <td>0.03275</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>-0.095</td>
      <td>-0.085</td>
      <td>-0.032</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.112</td>
      <td>0.081</td>
      <td>-0.048</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.007</td>
      <td>-0.074</td>
      <td>23.93</td>
      <td>167.6</td>
      <td>324.5</td>
      <td>-0.04348</td>
      <td>-0.02927</td>
      <td>-0.00649</td>
      <td>-0.01440</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>-0.035</td>
      <td>0.090</td>
      <td>0.066</td>
      <td>0.105</td>
      <td>0.032</td>
      <td>0.040</td>
      <td>-0.083</td>
      <td>0.013</td>
      <td>0.097</td>
      <td>0.048</td>
      <td>...</td>
      <td>0.099</td>
      <td>0.008</td>
      <td>24.06</td>
      <td>166.6</td>
      <td>325.5</td>
      <td>0.14545</td>
      <td>0.06182</td>
      <td>0.08497</td>
      <td>0.01663</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.088</td>
      <td>0.062</td>
      <td>0.032</td>
      <td>0.048</td>
      <td>0.109</td>
      <td>0.073</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.137</td>
      <td>0.085</td>
      <td>...</td>
      <td>-0.175</td>
      <td>0.171</td>
      <td>24.31</td>
      <td>167.6</td>
      <td>326.6</td>
      <td>0.03175</td>
      <td>0.06507</td>
      <td>0.03012</td>
      <td>-0.00413</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.064</td>
      <td>0.065</td>
      <td>0.082</td>
      <td>0.197</td>
      <td>0.023</td>
      <td>0.095</td>
      <td>0.030</td>
      <td>0.027</td>
      <td>0.063</td>
      <td>0.113</td>
      <td>...</td>
      <td>-0.077</td>
      <td>-0.004</td>
      <td>24.53</td>
      <td>168.8</td>
      <td>327.4</td>
      <td>0.04615</td>
      <td>0.06624</td>
      <td>0.07101</td>
      <td>-0.02318</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.032</td>
      <td>0.005</td>
      <td>0.022</td>
      <td>0.000</td>
      <td>-0.055</td>
      <td>0.162</td>
      <td>0.122</td>
      <td>0.019</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>...</td>
      <td>-0.038</td>
      <td>0.072</td>
      <td>23.12</td>
      <td>169.6</td>
      <td>328.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>0.093</td>
      <td>0.101</td>
      <td>0.048</td>
      <td>-0.051</td>
      <td>-0.044</td>
      <td>0.093</td>
      <td>-0.055</td>
      <td>0.121</td>
      <td>0.034</td>
      <td>0.003</td>
      <td>...</td>
      <td>0.071</td>
      <td>0.123</td>
      <td>17.65</td>
      <td>168.4</td>
      <td>327.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>0.066</td>
      <td>0.153</td>
      <td>0.021</td>
      <td>-0.040</td>
      <td>-0.043</td>
      <td>-0.063</td>
      <td>0.076</td>
      <td>0.072</td>
      <td>0.174</td>
      <td>0.004</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.051</td>
      <td>12.62</td>
      <td>166.1</td>
      <td>326.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.013</td>
      <td>-0.042</td>
      <td>-0.006</td>
      <td>-0.097</td>
      <td>0.061</td>
      <td>0.119</td>
      <td>0.059</td>
      <td>-0.051</td>
      <td>0.113</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.050</td>
      <td>-0.037</td>
      <td>10.68</td>
      <td>167.6</td>
      <td>325.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 27 columns</p>
</div>




```python
select1 = (df.index>end_date)
df2=df[select1].copy()
df2.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 27 columns</p>
</div>




```python
formula = 'Q("CONTIL") ~ Q("MARKET")'

results = smf.ols(formula, df1).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CONTIL")   R-squared:                       0.111
    Model:                            OLS   Adj. R-squared:                  0.104
    Method:                 Least Squares   F-statistic:                     14.65
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           0.000209
    Time:                        14:39:02   Log-Likelihood:                 63.430
    No. Observations:                 119   AIC:                            -122.9
    Df Residuals:                     117   BIC:                            -117.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0115      0.013     -0.858      0.393      -0.038       0.015
    Q("MARKET")     0.7375      0.193      3.828      0.000       0.356       1.119
    ==============================================================================
    Omnibus:                      106.122   Durbin-Watson:                   2.068
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2224.184
    Skew:                           2.697   Prob(JB):                         0.00
    Kurtosis:                      23.481   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
formula = 'Q("BOISE") ~ Q("MARKET")'

results = smf.ols(formula, df1).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             Q("BOISE")   R-squared:                       0.422
    Model:                            OLS   Adj. R-squared:                  0.417
    Method:                 Least Squares   F-statistic:                     85.45
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           1.31e-15
    Time:                        14:39:02   Log-Likelihood:                 141.64
    No. Observations:                 119   AIC:                            -279.3
    Df Residuals:                     117   BIC:                            -273.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       0.0032      0.007      0.456      0.649      -0.011       0.017
    Q("MARKET")     0.9230      0.100      9.244      0.000       0.725       1.121
    ==============================================================================
    Omnibus:                        4.937   Durbin-Watson:                   2.183
    Prob(Omnibus):                  0.085   Jarque-Bera (JB):                5.734
    Skew:                           0.215   Prob(JB):                       0.0569
    Kurtosis:                       3.986   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
RF.Chow_Test(df1, y='CONTIL', x='MARKET', special_date='1986-11-01')
```




    (1.0162280527443794, 0.36510313332750644)




![png](output_75_1.png)



```python
df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1976-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>1976-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>1976-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>1976-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>1976-06-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>1976-07-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>1976-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>1976-09-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>1976-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>1976-11-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>1976-12-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>1977-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>1977-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>1977-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>1977-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>1977-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>1977-06-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>1977-07-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>1977-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>1977-09-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>1977-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>1977-11-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>1977-12-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1981-11-01</th>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>0.075</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1981-12-01</th>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-01-01</th>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>0.119</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-02-01</th>
      <td>-0.159</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.175</td>
      <td>-0.152</td>
      <td>-0.049</td>
      <td>0.145</td>
      <td>0.098</td>
      <td>-0.111</td>
      <td>-0.014</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>29.76</td>
      <td>142.9</td>
      <td>283.4</td>
      <td>-0.17021</td>
      <td>-0.10825</td>
      <td>0.00276</td>
      <td>-0.02619</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-03-01</th>
      <td>0.108</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>-0.017</td>
      <td>-0.302</td>
      <td>-0.104</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.136</td>
      <td>-0.034</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.009</td>
      <td>28.31</td>
      <td>141.7</td>
      <td>283.1</td>
      <td>-0.05128</td>
      <td>0.09595</td>
      <td>-0.06643</td>
      <td>-0.11714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-04-01</th>
      <td>-0.009</td>
      <td>0.101</td>
      <td>0.050</td>
      <td>-0.013</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>-0.025</td>
      <td>0.076</td>
      <td>0.044</td>
      <td>0.075</td>
      <td>...</td>
      <td>-0.008</td>
      <td>0.059</td>
      <td>27.65</td>
      <td>140.2</td>
      <td>284.3</td>
      <td>0.13514</td>
      <td>-0.02151</td>
      <td>0.05993</td>
      <td>0.06141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-05-01</th>
      <td>-0.189</td>
      <td>-0.101</td>
      <td>0.016</td>
      <td>-0.091</td>
      <td>-0.180</td>
      <td>-0.056</td>
      <td>0.042</td>
      <td>-0.027</td>
      <td>0.043</td>
      <td>-0.029</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.086</td>
      <td>27.67</td>
      <td>139.2</td>
      <td>287.1</td>
      <td>-0.04762</td>
      <td>-0.05495</td>
      <td>-0.02898</td>
      <td>-0.04610</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-06-01</th>
      <td>-0.044</td>
      <td>-0.003</td>
      <td>-0.024</td>
      <td>-0.096</td>
      <td>-0.060</td>
      <td>-0.073</td>
      <td>0.106</td>
      <td>0.050</td>
      <td>-0.033</td>
      <td>-0.014</td>
      <td>...</td>
      <td>-0.017</td>
      <td>-0.015</td>
      <td>28.11</td>
      <td>138.7</td>
      <td>290.6</td>
      <td>-0.02500</td>
      <td>-0.01395</td>
      <td>-0.02222</td>
      <td>-0.05799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-07-01</th>
      <td>0.006</td>
      <td>-0.025</td>
      <td>-0.032</td>
      <td>-0.303</td>
      <td>-0.054</td>
      <td>-0.055</td>
      <td>-0.118</td>
      <td>0.038</td>
      <td>0.019</td>
      <td>0.082</td>
      <td>...</td>
      <td>-0.060</td>
      <td>-0.012</td>
      <td>28.33</td>
      <td>138.8</td>
      <td>292.6</td>
      <td>0.10256</td>
      <td>-0.02410</td>
      <td>-0.08333</td>
      <td>0.07975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-08-01</th>
      <td>0.379</td>
      <td>0.077</td>
      <td>0.133</td>
      <td>0.070</td>
      <td>0.216</td>
      <td>0.273</td>
      <td>0.055</td>
      <td>0.032</td>
      <td>0.130</td>
      <td>0.087</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.221</td>
      <td>28.18</td>
      <td>138.4</td>
      <td>292.8</td>
      <td>0.09302</td>
      <td>0.22222</td>
      <td>0.17273</td>
      <td>0.07607</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-09-01</th>
      <td>-0.109</td>
      <td>0.059</td>
      <td>0.039</td>
      <td>0.058</td>
      <td>-0.165</td>
      <td>-0.061</td>
      <td>-0.139</td>
      <td>0.000</td>
      <td>0.209</td>
      <td>0.041</td>
      <td>...</td>
      <td>0.027</td>
      <td>-0.029</td>
      <td>27.99</td>
      <td>137.3</td>
      <td>293.3</td>
      <td>-0.02128</td>
      <td>-0.06566</td>
      <td>-0.01075</td>
      <td>0.19015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-10-01</th>
      <td>0.314</td>
      <td>0.318</td>
      <td>-0.050</td>
      <td>0.268</td>
      <td>0.528</td>
      <td>0.133</td>
      <td>0.171</td>
      <td>0.160</td>
      <td>-0.009</td>
      <td>0.089</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.150</td>
      <td>28.74</td>
      <td>135.8</td>
      <td>294.1</td>
      <td>0.10870</td>
      <td>0.13297</td>
      <td>0.11594</td>
      <td>-0.03079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-11-01</th>
      <td>0.145</td>
      <td>0.007</td>
      <td>-0.011</td>
      <td>-0.106</td>
      <td>0.003</td>
      <td>0.175</td>
      <td>0.289</td>
      <td>-0.025</td>
      <td>-0.072</td>
      <td>0.094</td>
      <td>...</td>
      <td>0.012</td>
      <td>0.141</td>
      <td>28.70</td>
      <td>134.8</td>
      <td>293.6</td>
      <td>-0.01961</td>
      <td>0.01942</td>
      <td>-0.00714</td>
      <td>-0.01897</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-12-01</th>
      <td>-0.001</td>
      <td>-0.098</td>
      <td>0.123</td>
      <td>0.037</td>
      <td>0.053</td>
      <td>-0.052</td>
      <td>0.093</td>
      <td>-0.020</td>
      <td>0.015</td>
      <td>0.113</td>
      <td>...</td>
      <td>0.029</td>
      <td>-0.040</td>
      <td>28.12</td>
      <td>134.7</td>
      <td>292.4</td>
      <td>0.08000</td>
      <td>0.00286</td>
      <td>-0.04651</td>
      <td>0.07367</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-01-01</th>
      <td>-0.045</td>
      <td>0.085</td>
      <td>-0.012</td>
      <td>0.049</td>
      <td>0.208</td>
      <td>0.225</td>
      <td>0.040</td>
      <td>-0.039</td>
      <td>0.015</td>
      <td>0.027</td>
      <td>...</td>
      <td>0.036</td>
      <td>0.023</td>
      <td>27.22</td>
      <td>137.4</td>
      <td>293.1</td>
      <td>0.18519</td>
      <td>0.11111</td>
      <td>0.11498</td>
      <td>0.07919</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-02-01</th>
      <td>0.037</td>
      <td>0.039</td>
      <td>0.060</td>
      <td>-0.035</td>
      <td>0.237</td>
      <td>-0.010</td>
      <td>0.027</td>
      <td>0.067</td>
      <td>0.024</td>
      <td>0.010</td>
      <td>...</td>
      <td>0.008</td>
      <td>0.065</td>
      <td>26.41</td>
      <td>138.1</td>
      <td>293.2</td>
      <td>-0.03125</td>
      <td>0.07826</td>
      <td>0.01500</td>
      <td>0.02199</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-03-01</th>
      <td>0.113</td>
      <td>0.132</td>
      <td>0.048</td>
      <td>0.097</td>
      <td>0.040</td>
      <td>0.034</td>
      <td>-0.016</td>
      <td>0.061</td>
      <td>0.084</td>
      <td>0.028</td>
      <td>...</td>
      <td>0.039</td>
      <td>-0.023</td>
      <td>26.08</td>
      <td>140.0</td>
      <td>293.4</td>
      <td>0.00000</td>
      <td>-0.09839</td>
      <td>0.04062</td>
      <td>-0.14419</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-04-01</th>
      <td>0.082</td>
      <td>0.104</td>
      <td>0.045</td>
      <td>0.073</td>
      <td>0.079</td>
      <td>-0.060</td>
      <td>-0.043</td>
      <td>0.066</td>
      <td>0.119</td>
      <td>0.150</td>
      <td>...</td>
      <td>0.098</td>
      <td>0.091</td>
      <td>25.85</td>
      <td>142.6</td>
      <td>295.5</td>
      <td>0.16129</td>
      <td>0.20455</td>
      <td>0.12613</td>
      <td>0.02952</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-05-01</th>
      <td>-0.014</td>
      <td>-0.102</td>
      <td>-0.012</td>
      <td>0.000</td>
      <td>-0.114</td>
      <td>-0.052</td>
      <td>-0.045</td>
      <td>0.023</td>
      <td>0.016</td>
      <td>-0.041</td>
      <td>...</td>
      <td>-0.038</td>
      <td>-0.067</td>
      <td>26.08</td>
      <td>144.4</td>
      <td>297.1</td>
      <td>0.02778</td>
      <td>0.00000</td>
      <td>0.04747</td>
      <td>0.01291</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-06-01</th>
      <td>-0.130</td>
      <td>-0.016</td>
      <td>0.000</td>
      <td>-0.068</td>
      <td>-0.042</td>
      <td>0.075</td>
      <td>0.012</td>
      <td>-0.026</td>
      <td>0.114</td>
      <td>0.081</td>
      <td>...</td>
      <td>0.018</td>
      <td>-0.013</td>
      <td>25.98</td>
      <td>146.4</td>
      <td>298.1</td>
      <td>-0.01351</td>
      <td>0.01736</td>
      <td>-0.01546</td>
      <td>-0.05673</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-07-01</th>
      <td>-0.087</td>
      <td>-0.079</td>
      <td>0.017</td>
      <td>0.046</td>
      <td>0.173</td>
      <td>-0.142</td>
      <td>-0.259</td>
      <td>-0.072</td>
      <td>-0.007</td>
      <td>0.001</td>
      <td>...</td>
      <td>0.036</td>
      <td>-0.071</td>
      <td>25.86</td>
      <td>149.7</td>
      <td>299.3</td>
      <td>-0.04110</td>
      <td>0.01128</td>
      <td>0.00524</td>
      <td>0.02560</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-08-01</th>
      <td>0.060</td>
      <td>-0.007</td>
      <td>-0.023</td>
      <td>0.055</td>
      <td>0.053</td>
      <td>0.007</td>
      <td>0.080</td>
      <td>-0.010</td>
      <td>0.062</td>
      <td>0.001</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.011</td>
      <td>26.03</td>
      <td>151.8</td>
      <td>300.3</td>
      <td>0.00000</td>
      <td>0.10037</td>
      <td>0.10625</td>
      <td>-0.01686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-09-01</th>
      <td>0.102</td>
      <td>0.006</td>
      <td>0.087</td>
      <td>-0.026</td>
      <td>0.090</td>
      <td>-0.005</td>
      <td>0.041</td>
      <td>-0.037</td>
      <td>0.049</td>
      <td>0.062</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.033</td>
      <td>26.08</td>
      <td>153.8</td>
      <td>301.8</td>
      <td>-0.07143</td>
      <td>-0.00135</td>
      <td>-0.01190</td>
      <td>-0.01158</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-10-01</th>
      <td>-0.052</td>
      <td>-0.118</td>
      <td>0.101</td>
      <td>-0.088</td>
      <td>-0.069</td>
      <td>-0.364</td>
      <td>0.039</td>
      <td>0.116</td>
      <td>0.000</td>
      <td>-0.001</td>
      <td>...</td>
      <td>-0.014</td>
      <td>-0.046</td>
      <td>26.04</td>
      <td>155.0</td>
      <td>302.6</td>
      <td>-0.07692</td>
      <td>-0.05479</td>
      <td>-0.00723</td>
      <td>-0.04246</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.147</td>
      <td>0.162</td>
      <td>-0.025</td>
      <td>0.096</td>
      <td>-0.014</td>
      <td>0.065</td>
      <td>0.120</td>
      <td>-0.014</td>
      <td>0.077</td>
      <td>-0.066</td>
      <td>...</td>
      <td>0.011</td>
      <td>0.151</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>303.1</td>
      <td>0.06667</td>
      <td>-0.05072</td>
      <td>0.06214</td>
      <td>-0.02957</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.012</td>
      <td>0.023</td>
      <td>0.005</td>
      <td>-0.016</td>
      <td>0.068</td>
      <td>0.034</td>
      <td>-0.028</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.069</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>303.5</td>
      <td>-0.03125</td>
      <td>0.03282</td>
      <td>-0.03704</td>
      <td>0.01488</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.054</td>
      <td>0.024</td>
      <td>0.005</td>
      <td>-0.034</td>
      <td>0.117</td>
      <td>0.208</td>
      <td>-0.013</td>
      <td>-0.009</td>
      <td>0.065</td>
      <td>-0.065</td>
      <td>...</td>
      <td>0.108</td>
      <td>-0.039</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>305.4</td>
      <td>-0.01613</td>
      <td>-0.05243</td>
      <td>-0.04327</td>
      <td>-0.04322</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.088</td>
      <td>-0.039</td>
      <td>-0.069</td>
      <td>-0.101</td>
      <td>0.027</td>
      <td>-0.024</td>
      <td>-0.117</td>
      <td>-0.073</td>
      <td>-0.091</td>
      <td>-0.026</td>
      <td>...</td>
      <td>0.151</td>
      <td>-0.093</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>306.6</td>
      <td>0.03279</td>
      <td>-0.11462</td>
      <td>-0.03367</td>
      <td>0.04059</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>0.079</td>
      <td>-0.054</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>0.056</td>
      <td>0.057</td>
      <td>0.065</td>
      <td>-0.018</td>
      <td>-0.003</td>
      <td>0.034</td>
      <td>...</td>
      <td>-0.122</td>
      <td>0.094</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>307.3</td>
      <td>0.04762</td>
      <td>0.15000</td>
      <td>0.03958</td>
      <td>0.02159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>0.012</td>
      <td>-0.004</td>
      <td>0.031</td>
      <td>-0.231</td>
      <td>0.089</td>
      <td>0.053</td>
      <td>-0.085</td>
      <td>0.065</td>
      <td>-0.025</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>308.8</td>
      <td>-0.01515</td>
      <td>0.01969</td>
      <td>0.02538</td>
      <td>-0.03213</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 27 columns</p>
</div>




```python
df1.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
      <th>Before_Special</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.032</td>
      <td>0.011</td>
      <td>0.066</td>
      <td>0.143</td>
      <td>0.107</td>
      <td>0.185</td>
      <td>0.075</td>
      <td>-0.012</td>
      <td>0.092</td>
      <td>...</td>
      <td>0.164</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>196.7</td>
      <td>0.04405</td>
      <td>0.07107</td>
      <td>0.07813</td>
      <td>0.02814</td>
      <td>-0.01422</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.088</td>
      <td>0.024</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>-0.017</td>
      <td>-0.021</td>
      <td>-0.051</td>
      <td>-0.079</td>
      <td>0.049</td>
      <td>...</td>
      <td>0.039</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>197.8</td>
      <td>-0.04636</td>
      <td>0.04265</td>
      <td>0.03727</td>
      <td>0.09005</td>
      <td>0.09519</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.011</td>
      <td>0.048</td>
      <td>-0.013</td>
      <td>-0.031</td>
      <td>-0.037</td>
      <td>-0.081</td>
      <td>-0.012</td>
      <td>0.104</td>
      <td>-0.051</td>
      <td>...</td>
      <td>-0.021</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>199.3</td>
      <td>0.03472</td>
      <td>0.04000</td>
      <td>0.03024</td>
      <td>0.02977</td>
      <td>0.04889</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.071</td>
      <td>-0.067</td>
      <td>-0.123</td>
      <td>-0.085</td>
      <td>-0.077</td>
      <td>-0.153</td>
      <td>-0.032</td>
      <td>-0.138</td>
      <td>-0.046</td>
      <td>...</td>
      <td>-0.090</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>200.9</td>
      <td>-0.07651</td>
      <td>-0.07522</td>
      <td>-0.06067</td>
      <td>0.07194</td>
      <td>-0.13136</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.005</td>
      <td>0.035</td>
      <td>-0.038</td>
      <td>0.044</td>
      <td>0.064</td>
      <td>0.055</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.033</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>202.0</td>
      <td>0.04478</td>
      <td>0.00478</td>
      <td>0.05000</td>
      <td>-0.09443</td>
      <td>0.02927</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.047</td>
      <td>0.034</td>
      <td>0.117</td>
      <td>-0.023</td>
      <td>0.022</td>
      <td>-0.086</td>
      <td>0.108</td>
      <td>...</td>
      <td>-0.034</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>203.3</td>
      <td>0.00000</td>
      <td>-0.03905</td>
      <td>0.02857</td>
      <td>0.00941</td>
      <td>0.08173</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>0.043</td>
      <td>0.076</td>
      <td>-0.024</td>
      <td>-0.008</td>
      <td>-0.012</td>
      <td>-0.054</td>
      <td>-0.032</td>
      <td>0.042</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.203</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>204.7</td>
      <td>0.05429</td>
      <td>0.07035</td>
      <td>0.06151</td>
      <td>0.09336</td>
      <td>0.06667</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.082</td>
      <td>-0.011</td>
      <td>-0.020</td>
      <td>-0.015</td>
      <td>-0.066</td>
      <td>-0.060</td>
      <td>-0.079</td>
      <td>-0.023</td>
      <td>-0.017</td>
      <td>...</td>
      <td>-0.038</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>207.1</td>
      <td>-0.05556</td>
      <td>-0.04225</td>
      <td>-0.02150</td>
      <td>0.08042</td>
      <td>0.02500</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.171</td>
      <td>0.088</td>
      <td>0.098</td>
      <td>-0.043</td>
      <td>0.065</td>
      <td>0.052</td>
      <td>...</td>
      <td>0.097</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>209.1</td>
      <td>-0.04412</td>
      <td>0.11176</td>
      <td>0.09082</td>
      <td>-0.01428</td>
      <td>0.12757</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.000</td>
      <td>-0.057</td>
      <td>0.064</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>-0.056</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.069</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>211.5</td>
      <td>-0.33077</td>
      <td>-0.07143</td>
      <td>-0.06200</td>
      <td>-0.01333</td>
      <td>0.00000</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.005</td>
      <td>-0.045</td>
      <td>-0.028</td>
      <td>0.063</td>
      <td>0.035</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.013</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>214.1</td>
      <td>-0.17241</td>
      <td>-0.01923</td>
      <td>-0.03305</td>
      <td>0.07745</td>
      <td>0.00511</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.095</td>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>-0.006</td>
      <td>-0.043</td>
      <td>0.095</td>
      <td>-0.035</td>
      <td>...</td>
      <td>0.053</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>216.6</td>
      <td>0.15714</td>
      <td>0.02843</td>
      <td>-0.02174</td>
      <td>0.08434</td>
      <td>0.11397</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.075</td>
      <td>0.015</td>
      <td>-0.034</td>
      <td>-0.059</td>
      <td>0.009</td>
      <td>0.075</td>
      <td>-0.013</td>
      <td>-0.096</td>
      <td>-0.049</td>
      <td>...</td>
      <td>0.000</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>218.9</td>
      <td>-0.01235</td>
      <td>0.08213</td>
      <td>-0.00909</td>
      <td>0.05802</td>
      <td>0.01980</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.147</td>
      <td>0.162</td>
      <td>-0.025</td>
      <td>0.096</td>
      <td>-0.014</td>
      <td>0.065</td>
      <td>0.120</td>
      <td>-0.014</td>
      <td>0.077</td>
      <td>-0.066</td>
      <td>...</td>
      <td>0.151</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>303.1</td>
      <td>0.06667</td>
      <td>-0.05072</td>
      <td>0.06214</td>
      <td>-0.02957</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.012</td>
      <td>0.023</td>
      <td>0.005</td>
      <td>-0.016</td>
      <td>0.068</td>
      <td>0.034</td>
      <td>-0.028</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.039</td>
      <td>...</td>
      <td>-0.069</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>303.5</td>
      <td>-0.03125</td>
      <td>0.03282</td>
      <td>-0.03704</td>
      <td>0.01488</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.054</td>
      <td>0.024</td>
      <td>0.005</td>
      <td>-0.034</td>
      <td>0.117</td>
      <td>0.208</td>
      <td>-0.013</td>
      <td>-0.009</td>
      <td>0.065</td>
      <td>-0.065</td>
      <td>...</td>
      <td>-0.039</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>305.4</td>
      <td>-0.01613</td>
      <td>-0.05243</td>
      <td>-0.04327</td>
      <td>-0.04322</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.088</td>
      <td>-0.039</td>
      <td>-0.069</td>
      <td>-0.101</td>
      <td>0.027</td>
      <td>-0.024</td>
      <td>-0.117</td>
      <td>-0.073</td>
      <td>-0.091</td>
      <td>-0.026</td>
      <td>...</td>
      <td>-0.093</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>306.6</td>
      <td>0.03279</td>
      <td>-0.11462</td>
      <td>-0.03367</td>
      <td>0.04059</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>0.079</td>
      <td>-0.054</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>0.056</td>
      <td>0.057</td>
      <td>0.065</td>
      <td>-0.018</td>
      <td>-0.003</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.094</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>307.3</td>
      <td>0.04762</td>
      <td>0.15000</td>
      <td>0.03958</td>
      <td>0.02159</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>0.012</td>
      <td>-0.004</td>
      <td>0.031</td>
      <td>-0.231</td>
      <td>0.089</td>
      <td>0.053</td>
      <td>-0.085</td>
      <td>0.065</td>
      <td>-0.025</td>
      <td>-0.002</td>
      <td>...</td>
      <td>-0.088</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>308.8</td>
      <td>-0.01515</td>
      <td>0.01969</td>
      <td>0.02538</td>
      <td>-0.03213</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-05-01</th>
      <td>-0.172</td>
      <td>-0.148</td>
      <td>0.021</td>
      <td>-0.600</td>
      <td>-0.094</td>
      <td>-0.071</td>
      <td>-0.070</td>
      <td>0.018</td>
      <td>-0.087</td>
      <td>-0.044</td>
      <td>...</td>
      <td>-0.087</td>
      <td>26.00</td>
      <td>162.8</td>
      <td>309.7</td>
      <td>0.07692</td>
      <td>-0.12355</td>
      <td>-0.05545</td>
      <td>-0.01126</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-06-01</th>
      <td>0.025</td>
      <td>0.078</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>-0.043</td>
      <td>-0.012</td>
      <td>0.055</td>
      <td>0.105</td>
      <td>-0.019</td>
      <td>...</td>
      <td>0.019</td>
      <td>26.09</td>
      <td>164.4</td>
      <td>310.7</td>
      <td>0.02857</td>
      <td>0.00264</td>
      <td>-0.02926</td>
      <td>0.00103</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-07-01</th>
      <td>0.015</td>
      <td>-0.029</td>
      <td>0.054</td>
      <td>-0.205</td>
      <td>-0.061</td>
      <td>-0.009</td>
      <td>0.045</td>
      <td>-0.018</td>
      <td>-0.112</td>
      <td>0.047</td>
      <td>...</td>
      <td>0.036</td>
      <td>26.11</td>
      <td>165.9</td>
      <td>311.7</td>
      <td>0.08333</td>
      <td>0.01339</td>
      <td>-0.03014</td>
      <td>-0.08266</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-08-01</th>
      <td>0.177</td>
      <td>0.164</td>
      <td>0.029</td>
      <td>0.086</td>
      <td>0.312</td>
      <td>0.159</td>
      <td>0.040</td>
      <td>0.061</td>
      <td>0.018</td>
      <td>0.127</td>
      <td>...</td>
      <td>0.055</td>
      <td>26.02</td>
      <td>166.0</td>
      <td>313.0</td>
      <td>0.02564</td>
      <td>0.10132</td>
      <td>0.14689</td>
      <td>0.00360</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-09-01</th>
      <td>-0.056</td>
      <td>0.076</td>
      <td>0.051</td>
      <td>0.974</td>
      <td>-0.132</td>
      <td>-0.025</td>
      <td>0.008</td>
      <td>0.011</td>
      <td>0.165</td>
      <td>0.004</td>
      <td>...</td>
      <td>-0.069</td>
      <td>25.97</td>
      <td>165.0</td>
      <td>314.5</td>
      <td>0.01250</td>
      <td>-0.08160</td>
      <td>-0.00750</td>
      <td>-0.01942</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-10-01</th>
      <td>0.053</td>
      <td>-0.027</td>
      <td>0.019</td>
      <td>-0.232</td>
      <td>0.047</td>
      <td>0.093</td>
      <td>0.161</td>
      <td>-0.010</td>
      <td>-0.160</td>
      <td>0.012</td>
      <td>...</td>
      <td>0.035</td>
      <td>25.92</td>
      <td>164.5</td>
      <td>315.3</td>
      <td>0.08642</td>
      <td>0.02655</td>
      <td>-0.05542</td>
      <td>-0.00217</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-11-01</th>
      <td>-0.038</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>-0.023</td>
      <td>0.019</td>
      <td>0.006</td>
      <td>-0.026</td>
      <td>-0.072</td>
      <td>0.094</td>
      <td>-0.023</td>
      <td>...</td>
      <td>0.032</td>
      <td>25.44</td>
      <td>165.2</td>
      <td>315.3</td>
      <td>0.02273</td>
      <td>-0.00862</td>
      <td>0.00800</td>
      <td>0.00285</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-12-01</th>
      <td>0.068</td>
      <td>0.098</td>
      <td>0.084</td>
      <td>0.095</td>
      <td>0.096</td>
      <td>0.070</td>
      <td>0.156</td>
      <td>0.017</td>
      <td>-0.005</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.026</td>
      <td>25.05</td>
      <td>166.2</td>
      <td>315.5</td>
      <td>0.02222</td>
      <td>-0.02783</td>
      <td>0.06452</td>
      <td>-0.06479</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-01-01</th>
      <td>0.046</td>
      <td>0.097</td>
      <td>-0.021</td>
      <td>0.587</td>
      <td>0.215</td>
      <td>0.084</td>
      <td>-0.010</td>
      <td>0.095</td>
      <td>0.091</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.084</td>
      <td>24.28</td>
      <td>165.6</td>
      <td>316.1</td>
      <td>0.00000</td>
      <td>0.07727</td>
      <td>0.04293</td>
      <td>-0.06265</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>-0.059</td>
      <td>-0.015</td>
      <td>0.034</td>
      <td>-0.096</td>
      <td>-0.210</td>
      <td>-0.067</td>
      <td>0.087</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>-0.009</td>
      <td>...</td>
      <td>-0.016</td>
      <td>23.63</td>
      <td>165.7</td>
      <td>317.4</td>
      <td>0.08696</td>
      <td>0.00844</td>
      <td>0.03148</td>
      <td>0.01720</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>-0.029</td>
      <td>0.046</td>
      <td>0.057</td>
      <td>0.030</td>
      <td>-0.195</td>
      <td>-0.071</td>
      <td>-0.003</td>
      <td>0.054</td>
      <td>0.130</td>
      <td>-0.052</td>
      <td>...</td>
      <td>-0.081</td>
      <td>23.88</td>
      <td>166.1</td>
      <td>318.8</td>
      <td>-0.04000</td>
      <td>-0.01423</td>
      <td>-0.01190</td>
      <td>-0.04400</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>0.010</td>
      <td>0.012</td>
      <td>0.019</td>
      <td>-0.029</td>
      <td>-0.157</td>
      <td>-0.050</td>
      <td>-0.123</td>
      <td>-0.083</td>
      <td>-0.037</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.003</td>
      <td>24.15</td>
      <td>166.2</td>
      <td>320.1</td>
      <td>-0.01042</td>
      <td>0.03448</td>
      <td>0.06988</td>
      <td>0.13447</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.158</td>
      <td>0.094</td>
      <td>0.098</td>
      <td>-0.091</td>
      <td>-0.078</td>
      <td>0.057</td>
      <td>0.179</td>
      <td>0.137</td>
      <td>0.234</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.031</td>
      <td>24.18</td>
      <td>166.2</td>
      <td>321.3</td>
      <td>0.17895</td>
      <td>0.13333</td>
      <td>0.10135</td>
      <td>-0.02165</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.086</td>
      <td>0.043</td>
      <td>0.046</td>
      <td>-0.050</td>
      <td>0.060</td>
      <td>-0.101</td>
      <td>0.021</td>
      <td>0.060</td>
      <td>-0.031</td>
      <td>-0.038</td>
      <td>...</td>
      <td>-0.004</td>
      <td>24.03</td>
      <td>166.5</td>
      <td>322.3</td>
      <td>0.00893</td>
      <td>0.06471</td>
      <td>-0.03727</td>
      <td>-0.01393</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>-0.026</td>
      <td>-0.030</td>
      <td>-0.084</td>
      <td>0.018</td>
      <td>0.043</td>
      <td>0.080</td>
      <td>0.008</td>
      <td>-0.099</td>
      <td>-0.036</td>
      <td>0.062</td>
      <td>...</td>
      <td>0.020</td>
      <td>24.00</td>
      <td>166.2</td>
      <td>322.8</td>
      <td>-0.06195</td>
      <td>0.03147</td>
      <td>0.03011</td>
      <td>-0.00816</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.011</td>
      <td>-0.063</td>
      <td>0.043</td>
      <td>-0.052</td>
      <td>-0.006</td>
      <td>0.032</td>
      <td>-0.066</td>
      <td>0.002</td>
      <td>0.025</td>
      <td>-0.028</td>
      <td>...</td>
      <td>-0.013</td>
      <td>23.92</td>
      <td>167.7</td>
      <td>323.5</td>
      <td>0.08491</td>
      <td>-0.02712</td>
      <td>-0.02296</td>
      <td>0.03275</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>-0.095</td>
      <td>-0.085</td>
      <td>-0.032</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.112</td>
      <td>0.081</td>
      <td>-0.048</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.074</td>
      <td>23.93</td>
      <td>167.6</td>
      <td>324.5</td>
      <td>-0.04348</td>
      <td>-0.02927</td>
      <td>-0.00649</td>
      <td>-0.01440</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>-0.035</td>
      <td>0.090</td>
      <td>0.066</td>
      <td>0.105</td>
      <td>0.032</td>
      <td>0.040</td>
      <td>-0.083</td>
      <td>0.013</td>
      <td>0.097</td>
      <td>0.048</td>
      <td>...</td>
      <td>0.008</td>
      <td>24.06</td>
      <td>166.6</td>
      <td>325.5</td>
      <td>0.14545</td>
      <td>0.06182</td>
      <td>0.08497</td>
      <td>0.01663</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.088</td>
      <td>0.062</td>
      <td>0.032</td>
      <td>0.048</td>
      <td>0.109</td>
      <td>0.073</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.137</td>
      <td>0.085</td>
      <td>...</td>
      <td>0.171</td>
      <td>24.31</td>
      <td>167.6</td>
      <td>326.6</td>
      <td>0.03175</td>
      <td>0.06507</td>
      <td>0.03012</td>
      <td>-0.00413</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.064</td>
      <td>0.065</td>
      <td>0.082</td>
      <td>0.197</td>
      <td>0.023</td>
      <td>0.095</td>
      <td>0.030</td>
      <td>0.027</td>
      <td>0.063</td>
      <td>0.113</td>
      <td>...</td>
      <td>-0.004</td>
      <td>24.53</td>
      <td>168.8</td>
      <td>327.4</td>
      <td>0.04615</td>
      <td>0.06624</td>
      <td>0.07101</td>
      <td>-0.02318</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.032</td>
      <td>0.005</td>
      <td>0.022</td>
      <td>0.000</td>
      <td>-0.055</td>
      <td>0.162</td>
      <td>0.122</td>
      <td>0.019</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>...</td>
      <td>0.072</td>
      <td>23.12</td>
      <td>169.6</td>
      <td>328.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>0.093</td>
      <td>0.101</td>
      <td>0.048</td>
      <td>-0.051</td>
      <td>-0.044</td>
      <td>0.093</td>
      <td>-0.055</td>
      <td>0.121</td>
      <td>0.034</td>
      <td>0.003</td>
      <td>...</td>
      <td>0.123</td>
      <td>17.65</td>
      <td>168.4</td>
      <td>327.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>0.066</td>
      <td>0.153</td>
      <td>0.021</td>
      <td>-0.040</td>
      <td>-0.043</td>
      <td>-0.063</td>
      <td>0.076</td>
      <td>0.072</td>
      <td>0.174</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.051</td>
      <td>12.62</td>
      <td>166.1</td>
      <td>326.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.013</td>
      <td>-0.042</td>
      <td>-0.006</td>
      <td>-0.097</td>
      <td>0.061</td>
      <td>0.119</td>
      <td>0.059</td>
      <td>-0.051</td>
      <td>0.113</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.037</td>
      <td>10.68</td>
      <td>167.6</td>
      <td>325.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 28 columns</p>
</div>




```python
formula_with_intercept = 'BOISE ~ MARKET + CPI + FRBIND'

results_with_intercept = smf.ols(formula_with_intercept, df1).fit()
print(results_with_intercept.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  BOISE   R-squared:                       0.430
    Model:                            OLS   Adj. R-squared:                  0.415
    Method:                 Least Squares   F-statistic:                     28.91
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           5.27e-14
    Time:                        14:39:12   Log-Likelihood:                 142.46
    No. Observations:                 119   AIC:                            -276.9
    Df Residuals:                     115   BIC:                            -265.8
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0426      0.096     -0.445      0.657      -0.232       0.147
    MARKET         0.9416      0.102      9.274      0.000       0.740       1.143
    CPI            0.0002      0.000      1.016      0.312      -0.000       0.001
    FRBIND     -6.588e-05      0.001     -0.085      0.933      -0.002       0.001
    ==============================================================================
    Omnibus:                        4.443   Durbin-Watson:                   2.215
    Prob(Omnibus):                  0.108   Jarque-Bera (JB):                4.956
    Skew:                           0.196   Prob(JB):                       0.0839
    Kurtosis:                       3.920   Cond. No.                     5.07e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.07e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1976-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>1976-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>1976-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>1976-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>1976-06-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>1976-07-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>1976-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>1976-09-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>1976-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>1976-11-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>1976-12-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>1977-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>1977-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>1977-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>1977-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>1977-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>1977-06-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>1977-07-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>1977-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>1977-09-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>1977-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>1977-11-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>1977-12-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1981-11-01</th>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>0.075</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1981-12-01</th>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-01-01</th>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>0.119</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-02-01</th>
      <td>-0.159</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.175</td>
      <td>-0.152</td>
      <td>-0.049</td>
      <td>0.145</td>
      <td>0.098</td>
      <td>-0.111</td>
      <td>-0.014</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>29.76</td>
      <td>142.9</td>
      <td>283.4</td>
      <td>-0.17021</td>
      <td>-0.10825</td>
      <td>0.00276</td>
      <td>-0.02619</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-03-01</th>
      <td>0.108</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>-0.017</td>
      <td>-0.302</td>
      <td>-0.104</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.136</td>
      <td>-0.034</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.009</td>
      <td>28.31</td>
      <td>141.7</td>
      <td>283.1</td>
      <td>-0.05128</td>
      <td>0.09595</td>
      <td>-0.06643</td>
      <td>-0.11714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-04-01</th>
      <td>-0.009</td>
      <td>0.101</td>
      <td>0.050</td>
      <td>-0.013</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>-0.025</td>
      <td>0.076</td>
      <td>0.044</td>
      <td>0.075</td>
      <td>...</td>
      <td>-0.008</td>
      <td>0.059</td>
      <td>27.65</td>
      <td>140.2</td>
      <td>284.3</td>
      <td>0.13514</td>
      <td>-0.02151</td>
      <td>0.05993</td>
      <td>0.06141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-05-01</th>
      <td>-0.189</td>
      <td>-0.101</td>
      <td>0.016</td>
      <td>-0.091</td>
      <td>-0.180</td>
      <td>-0.056</td>
      <td>0.042</td>
      <td>-0.027</td>
      <td>0.043</td>
      <td>-0.029</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.086</td>
      <td>27.67</td>
      <td>139.2</td>
      <td>287.1</td>
      <td>-0.04762</td>
      <td>-0.05495</td>
      <td>-0.02898</td>
      <td>-0.04610</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-06-01</th>
      <td>-0.044</td>
      <td>-0.003</td>
      <td>-0.024</td>
      <td>-0.096</td>
      <td>-0.060</td>
      <td>-0.073</td>
      <td>0.106</td>
      <td>0.050</td>
      <td>-0.033</td>
      <td>-0.014</td>
      <td>...</td>
      <td>-0.017</td>
      <td>-0.015</td>
      <td>28.11</td>
      <td>138.7</td>
      <td>290.6</td>
      <td>-0.02500</td>
      <td>-0.01395</td>
      <td>-0.02222</td>
      <td>-0.05799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-07-01</th>
      <td>0.006</td>
      <td>-0.025</td>
      <td>-0.032</td>
      <td>-0.303</td>
      <td>-0.054</td>
      <td>-0.055</td>
      <td>-0.118</td>
      <td>0.038</td>
      <td>0.019</td>
      <td>0.082</td>
      <td>...</td>
      <td>-0.060</td>
      <td>-0.012</td>
      <td>28.33</td>
      <td>138.8</td>
      <td>292.6</td>
      <td>0.10256</td>
      <td>-0.02410</td>
      <td>-0.08333</td>
      <td>0.07975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-08-01</th>
      <td>0.379</td>
      <td>0.077</td>
      <td>0.133</td>
      <td>0.070</td>
      <td>0.216</td>
      <td>0.273</td>
      <td>0.055</td>
      <td>0.032</td>
      <td>0.130</td>
      <td>0.087</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.221</td>
      <td>28.18</td>
      <td>138.4</td>
      <td>292.8</td>
      <td>0.09302</td>
      <td>0.22222</td>
      <td>0.17273</td>
      <td>0.07607</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-09-01</th>
      <td>-0.109</td>
      <td>0.059</td>
      <td>0.039</td>
      <td>0.058</td>
      <td>-0.165</td>
      <td>-0.061</td>
      <td>-0.139</td>
      <td>0.000</td>
      <td>0.209</td>
      <td>0.041</td>
      <td>...</td>
      <td>0.027</td>
      <td>-0.029</td>
      <td>27.99</td>
      <td>137.3</td>
      <td>293.3</td>
      <td>-0.02128</td>
      <td>-0.06566</td>
      <td>-0.01075</td>
      <td>0.19015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-10-01</th>
      <td>0.314</td>
      <td>0.318</td>
      <td>-0.050</td>
      <td>0.268</td>
      <td>0.528</td>
      <td>0.133</td>
      <td>0.171</td>
      <td>0.160</td>
      <td>-0.009</td>
      <td>0.089</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.150</td>
      <td>28.74</td>
      <td>135.8</td>
      <td>294.1</td>
      <td>0.10870</td>
      <td>0.13297</td>
      <td>0.11594</td>
      <td>-0.03079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-11-01</th>
      <td>0.145</td>
      <td>0.007</td>
      <td>-0.011</td>
      <td>-0.106</td>
      <td>0.003</td>
      <td>0.175</td>
      <td>0.289</td>
      <td>-0.025</td>
      <td>-0.072</td>
      <td>0.094</td>
      <td>...</td>
      <td>0.012</td>
      <td>0.141</td>
      <td>28.70</td>
      <td>134.8</td>
      <td>293.6</td>
      <td>-0.01961</td>
      <td>0.01942</td>
      <td>-0.00714</td>
      <td>-0.01897</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1982-12-01</th>
      <td>-0.001</td>
      <td>-0.098</td>
      <td>0.123</td>
      <td>0.037</td>
      <td>0.053</td>
      <td>-0.052</td>
      <td>0.093</td>
      <td>-0.020</td>
      <td>0.015</td>
      <td>0.113</td>
      <td>...</td>
      <td>0.029</td>
      <td>-0.040</td>
      <td>28.12</td>
      <td>134.7</td>
      <td>292.4</td>
      <td>0.08000</td>
      <td>0.00286</td>
      <td>-0.04651</td>
      <td>0.07367</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-01-01</th>
      <td>-0.045</td>
      <td>0.085</td>
      <td>-0.012</td>
      <td>0.049</td>
      <td>0.208</td>
      <td>0.225</td>
      <td>0.040</td>
      <td>-0.039</td>
      <td>0.015</td>
      <td>0.027</td>
      <td>...</td>
      <td>0.036</td>
      <td>0.023</td>
      <td>27.22</td>
      <td>137.4</td>
      <td>293.1</td>
      <td>0.18519</td>
      <td>0.11111</td>
      <td>0.11498</td>
      <td>0.07919</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-02-01</th>
      <td>0.037</td>
      <td>0.039</td>
      <td>0.060</td>
      <td>-0.035</td>
      <td>0.237</td>
      <td>-0.010</td>
      <td>0.027</td>
      <td>0.067</td>
      <td>0.024</td>
      <td>0.010</td>
      <td>...</td>
      <td>0.008</td>
      <td>0.065</td>
      <td>26.41</td>
      <td>138.1</td>
      <td>293.2</td>
      <td>-0.03125</td>
      <td>0.07826</td>
      <td>0.01500</td>
      <td>0.02199</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-03-01</th>
      <td>0.113</td>
      <td>0.132</td>
      <td>0.048</td>
      <td>0.097</td>
      <td>0.040</td>
      <td>0.034</td>
      <td>-0.016</td>
      <td>0.061</td>
      <td>0.084</td>
      <td>0.028</td>
      <td>...</td>
      <td>0.039</td>
      <td>-0.023</td>
      <td>26.08</td>
      <td>140.0</td>
      <td>293.4</td>
      <td>0.00000</td>
      <td>-0.09839</td>
      <td>0.04062</td>
      <td>-0.14419</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-04-01</th>
      <td>0.082</td>
      <td>0.104</td>
      <td>0.045</td>
      <td>0.073</td>
      <td>0.079</td>
      <td>-0.060</td>
      <td>-0.043</td>
      <td>0.066</td>
      <td>0.119</td>
      <td>0.150</td>
      <td>...</td>
      <td>0.098</td>
      <td>0.091</td>
      <td>25.85</td>
      <td>142.6</td>
      <td>295.5</td>
      <td>0.16129</td>
      <td>0.20455</td>
      <td>0.12613</td>
      <td>0.02952</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-05-01</th>
      <td>-0.014</td>
      <td>-0.102</td>
      <td>-0.012</td>
      <td>0.000</td>
      <td>-0.114</td>
      <td>-0.052</td>
      <td>-0.045</td>
      <td>0.023</td>
      <td>0.016</td>
      <td>-0.041</td>
      <td>...</td>
      <td>-0.038</td>
      <td>-0.067</td>
      <td>26.08</td>
      <td>144.4</td>
      <td>297.1</td>
      <td>0.02778</td>
      <td>0.00000</td>
      <td>0.04747</td>
      <td>0.01291</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-06-01</th>
      <td>-0.130</td>
      <td>-0.016</td>
      <td>0.000</td>
      <td>-0.068</td>
      <td>-0.042</td>
      <td>0.075</td>
      <td>0.012</td>
      <td>-0.026</td>
      <td>0.114</td>
      <td>0.081</td>
      <td>...</td>
      <td>0.018</td>
      <td>-0.013</td>
      <td>25.98</td>
      <td>146.4</td>
      <td>298.1</td>
      <td>-0.01351</td>
      <td>0.01736</td>
      <td>-0.01546</td>
      <td>-0.05673</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-07-01</th>
      <td>-0.087</td>
      <td>-0.079</td>
      <td>0.017</td>
      <td>0.046</td>
      <td>0.173</td>
      <td>-0.142</td>
      <td>-0.259</td>
      <td>-0.072</td>
      <td>-0.007</td>
      <td>0.001</td>
      <td>...</td>
      <td>0.036</td>
      <td>-0.071</td>
      <td>25.86</td>
      <td>149.7</td>
      <td>299.3</td>
      <td>-0.04110</td>
      <td>0.01128</td>
      <td>0.00524</td>
      <td>0.02560</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-08-01</th>
      <td>0.060</td>
      <td>-0.007</td>
      <td>-0.023</td>
      <td>0.055</td>
      <td>0.053</td>
      <td>0.007</td>
      <td>0.080</td>
      <td>-0.010</td>
      <td>0.062</td>
      <td>0.001</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.011</td>
      <td>26.03</td>
      <td>151.8</td>
      <td>300.3</td>
      <td>0.00000</td>
      <td>0.10037</td>
      <td>0.10625</td>
      <td>-0.01686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-09-01</th>
      <td>0.102</td>
      <td>0.006</td>
      <td>0.087</td>
      <td>-0.026</td>
      <td>0.090</td>
      <td>-0.005</td>
      <td>0.041</td>
      <td>-0.037</td>
      <td>0.049</td>
      <td>0.062</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.033</td>
      <td>26.08</td>
      <td>153.8</td>
      <td>301.8</td>
      <td>-0.07143</td>
      <td>-0.00135</td>
      <td>-0.01190</td>
      <td>-0.01158</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-10-01</th>
      <td>-0.052</td>
      <td>-0.118</td>
      <td>0.101</td>
      <td>-0.088</td>
      <td>-0.069</td>
      <td>-0.364</td>
      <td>0.039</td>
      <td>0.116</td>
      <td>0.000</td>
      <td>-0.001</td>
      <td>...</td>
      <td>-0.014</td>
      <td>-0.046</td>
      <td>26.04</td>
      <td>155.0</td>
      <td>302.6</td>
      <td>-0.07692</td>
      <td>-0.05479</td>
      <td>-0.00723</td>
      <td>-0.04246</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.147</td>
      <td>0.162</td>
      <td>-0.025</td>
      <td>0.096</td>
      <td>-0.014</td>
      <td>0.065</td>
      <td>0.120</td>
      <td>-0.014</td>
      <td>0.077</td>
      <td>-0.066</td>
      <td>...</td>
      <td>0.011</td>
      <td>0.151</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>303.1</td>
      <td>0.06667</td>
      <td>-0.05072</td>
      <td>0.06214</td>
      <td>-0.02957</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.012</td>
      <td>0.023</td>
      <td>0.005</td>
      <td>-0.016</td>
      <td>0.068</td>
      <td>0.034</td>
      <td>-0.028</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.069</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>303.5</td>
      <td>-0.03125</td>
      <td>0.03282</td>
      <td>-0.03704</td>
      <td>0.01488</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.054</td>
      <td>0.024</td>
      <td>0.005</td>
      <td>-0.034</td>
      <td>0.117</td>
      <td>0.208</td>
      <td>-0.013</td>
      <td>-0.009</td>
      <td>0.065</td>
      <td>-0.065</td>
      <td>...</td>
      <td>0.108</td>
      <td>-0.039</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>305.4</td>
      <td>-0.01613</td>
      <td>-0.05243</td>
      <td>-0.04327</td>
      <td>-0.04322</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.088</td>
      <td>-0.039</td>
      <td>-0.069</td>
      <td>-0.101</td>
      <td>0.027</td>
      <td>-0.024</td>
      <td>-0.117</td>
      <td>-0.073</td>
      <td>-0.091</td>
      <td>-0.026</td>
      <td>...</td>
      <td>0.151</td>
      <td>-0.093</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>306.6</td>
      <td>0.03279</td>
      <td>-0.11462</td>
      <td>-0.03367</td>
      <td>0.04059</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>0.079</td>
      <td>-0.054</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>0.056</td>
      <td>0.057</td>
      <td>0.065</td>
      <td>-0.018</td>
      <td>-0.003</td>
      <td>0.034</td>
      <td>...</td>
      <td>-0.122</td>
      <td>0.094</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>307.3</td>
      <td>0.04762</td>
      <td>0.15000</td>
      <td>0.03958</td>
      <td>0.02159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>0.012</td>
      <td>-0.004</td>
      <td>0.031</td>
      <td>-0.231</td>
      <td>0.089</td>
      <td>0.053</td>
      <td>-0.085</td>
      <td>0.065</td>
      <td>-0.025</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>308.8</td>
      <td>-0.01515</td>
      <td>0.01969</td>
      <td>0.02538</td>
      <td>-0.03213</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 27 columns</p>
</div>




```python
df4=df1[['BOISE','CONTIL','MARKET', 'CPI', 'POIL', 'FRBIND']].dropna()
df4['RINF']=df4['CPI'].pct_change(1)
df4['GIND']=df4['FRBIND'].pct_change(1)
df4['real_POIL']=df4['POIL']/df4['CPI']
df4['ROIL']=df4['real_POIL'].pct_change(1)
df4.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.045</td>
      <td>187.2</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.046368</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>0.037</td>
      <td>0.010</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>0.006410</td>
      <td>0.002882</td>
      <td>0.046921</td>
      <td>0.011946</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.003</td>
      <td>0.050</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>0.007431</td>
      <td>0.012213</td>
      <td>0.046365</td>
      <td>-0.011868</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.180</td>
      <td>0.063</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>0.008957</td>
      <td>0.016324</td>
      <td>0.046057</td>
      <td>-0.006625</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.061</td>
      <td>0.067</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>0.009399</td>
      <td>0.004888</td>
      <td>0.045577</td>
      <td>-0.010435</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>-0.059</td>
      <td>0.007</td>
      <td>195.3</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>0.010347</td>
      <td>0.006949</td>
      <td>0.046339</td>
      <td>0.016722</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.066</td>
      <td>0.071</td>
      <td>196.7</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>0.007168</td>
      <td>0.008282</td>
      <td>0.045552</td>
      <td>-0.016991</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.033</td>
      <td>0.079</td>
      <td>197.8</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>0.005592</td>
      <td>0.006845</td>
      <td>0.040698</td>
      <td>-0.106559</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.002</td>
      <td>199.3</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>0.007583</td>
      <td>0.004759</td>
      <td>0.045911</td>
      <td>0.128091</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.123</td>
      <td>-0.189</td>
      <td>200.9</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>0.008028</td>
      <td>0.005413</td>
      <td>0.045645</td>
      <td>-0.005796</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.038</td>
      <td>0.084</td>
      <td>202.0</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>0.005475</td>
      <td>0.006057</td>
      <td>0.045545</td>
      <td>-0.002192</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>0.047</td>
      <td>0.015</td>
      <td>203.3</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>0.006436</td>
      <td>0.006020</td>
      <td>0.046581</td>
      <td>0.022766</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>-0.024</td>
      <td>0.058</td>
      <td>204.7</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>0.006886</td>
      <td>0.010638</td>
      <td>0.046214</td>
      <td>-0.007888</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>207.1</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>0.011724</td>
      <td>0.003289</td>
      <td>0.046789</td>
      <td>0.012443</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.043</td>
      <td>0.123</td>
      <td>209.1</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>0.009657</td>
      <td>0.006557</td>
      <td>0.047011</td>
      <td>0.004745</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.064</td>
      <td>0.026</td>
      <td>211.5</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>0.011478</td>
      <td>-0.015635</td>
      <td>0.048842</td>
      <td>0.038940</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.005</td>
      <td>0.014</td>
      <td>214.1</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>0.012293</td>
      <td>0.010589</td>
      <td>0.050023</td>
      <td>0.024195</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.092</td>
      <td>0.075</td>
      <td>216.6</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>0.011677</td>
      <td>0.001965</td>
      <td>0.054017</td>
      <td>0.079828</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.034</td>
      <td>-0.013</td>
      <td>218.9</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>0.010619</td>
      <td>0.000000</td>
      <td>0.061169</td>
      <td>0.132420</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.058</td>
      <td>0.095</td>
      <td>221.1</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>0.010050</td>
      <td>-0.005882</td>
      <td>0.063320</td>
      <td>0.035153</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.033</td>
      <td>0.039</td>
      <td>223.4</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>0.010403</td>
      <td>0.003945</td>
      <td>0.065219</td>
      <td>0.030000</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.136</td>
      <td>-0.097</td>
      <td>225.4</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>0.008953</td>
      <td>0.000000</td>
      <td>0.067036</td>
      <td>0.027860</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.081</td>
      <td>0.116</td>
      <td>227.5</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>0.009317</td>
      <td>-0.002620</td>
      <td>0.068220</td>
      <td>0.017653</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.104</td>
      <td>0.086</td>
      <td>229.9</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>0.010549</td>
      <td>0.001313</td>
      <td>0.074076</td>
      <td>0.085839</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.103</td>
      <td>0.124</td>
      <td>233.2</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>0.014354</td>
      <td>0.001311</td>
      <td>0.076587</td>
      <td>0.033897</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.087</td>
      <td>0.112</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>0.085</td>
      <td>-0.243</td>
      <td>239.8</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>0.014382</td>
      <td>-0.003277</td>
      <td>0.080651</td>
      <td>0.013599</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.074</td>
      <td>0.080</td>
      <td>242.5</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>0.011259</td>
      <td>-0.024984</td>
      <td>0.083670</td>
      <td>0.037440</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.023</td>
      <td>0.062</td>
      <td>244.9</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>0.009897</td>
      <td>-0.028995</td>
      <td>0.085790</td>
      <td>0.025338</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.064</td>
      <td>0.086</td>
      <td>247.6</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>0.011025</td>
      <td>-0.017361</td>
      <td>0.086955</td>
      <td>0.013576</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.147</td>
      <td>0.096</td>
      <td>0.066</td>
      <td>303.1</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>0.001652</td>
      <td>0.001935</td>
      <td>0.086077</td>
      <td>0.000267</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.012</td>
      <td>-0.016</td>
      <td>-0.012</td>
      <td>303.5</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>0.001320</td>
      <td>0.005795</td>
      <td>0.085272</td>
      <td>-0.009356</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.054</td>
      <td>-0.034</td>
      <td>-0.029</td>
      <td>305.4</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>0.006260</td>
      <td>0.014725</td>
      <td>0.084905</td>
      <td>-0.004301</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.088</td>
      <td>-0.101</td>
      <td>-0.030</td>
      <td>306.6</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>0.003929</td>
      <td>0.009464</td>
      <td>0.084997</td>
      <td>0.001080</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>0.079</td>
      <td>-0.033</td>
      <td>0.003</td>
      <td>307.3</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>0.002283</td>
      <td>0.005000</td>
      <td>0.084771</td>
      <td>-0.002661</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>0.012</td>
      <td>-0.231</td>
      <td>-0.003</td>
      <td>308.8</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>0.004881</td>
      <td>0.008085</td>
      <td>0.083970</td>
      <td>-0.009442</td>
    </tr>
    <tr>
      <th>1984-05-01</th>
      <td>-0.172</td>
      <td>-0.600</td>
      <td>-0.058</td>
      <td>309.7</td>
      <td>26.00</td>
      <td>162.8</td>
      <td>0.002915</td>
      <td>0.004318</td>
      <td>0.083952</td>
      <td>-0.000214</td>
    </tr>
    <tr>
      <th>1984-06-01</th>
      <td>0.025</td>
      <td>0.000</td>
      <td>0.005</td>
      <td>310.7</td>
      <td>26.09</td>
      <td>164.4</td>
      <td>0.003229</td>
      <td>0.009828</td>
      <td>0.083972</td>
      <td>0.000232</td>
    </tr>
    <tr>
      <th>1984-07-01</th>
      <td>0.015</td>
      <td>-0.205</td>
      <td>-0.058</td>
      <td>311.7</td>
      <td>26.11</td>
      <td>165.9</td>
      <td>0.003219</td>
      <td>0.009124</td>
      <td>0.083766</td>
      <td>-0.002444</td>
    </tr>
    <tr>
      <th>1984-08-01</th>
      <td>0.177</td>
      <td>0.086</td>
      <td>0.146</td>
      <td>313.0</td>
      <td>26.02</td>
      <td>166.0</td>
      <td>0.004171</td>
      <td>0.000603</td>
      <td>0.083131</td>
      <td>-0.007586</td>
    </tr>
    <tr>
      <th>1984-09-01</th>
      <td>-0.056</td>
      <td>0.974</td>
      <td>0.000</td>
      <td>314.5</td>
      <td>25.97</td>
      <td>165.0</td>
      <td>0.004792</td>
      <td>-0.006024</td>
      <td>0.082576</td>
      <td>-0.006682</td>
    </tr>
    <tr>
      <th>1984-10-01</th>
      <td>0.053</td>
      <td>-0.232</td>
      <td>-0.035</td>
      <td>315.3</td>
      <td>25.92</td>
      <td>164.5</td>
      <td>0.002544</td>
      <td>-0.003030</td>
      <td>0.082207</td>
      <td>-0.004458</td>
    </tr>
    <tr>
      <th>1984-11-01</th>
      <td>-0.038</td>
      <td>-0.023</td>
      <td>-0.019</td>
      <td>315.3</td>
      <td>25.44</td>
      <td>165.2</td>
      <td>0.000000</td>
      <td>0.004255</td>
      <td>0.080685</td>
      <td>-0.018519</td>
    </tr>
    <tr>
      <th>1984-12-01</th>
      <td>0.068</td>
      <td>0.095</td>
      <td>-0.001</td>
      <td>315.5</td>
      <td>25.05</td>
      <td>166.2</td>
      <td>0.000634</td>
      <td>0.006053</td>
      <td>0.079398</td>
      <td>-0.015954</td>
    </tr>
    <tr>
      <th>1985-01-01</th>
      <td>0.046</td>
      <td>0.587</td>
      <td>0.097</td>
      <td>316.1</td>
      <td>24.28</td>
      <td>165.6</td>
      <td>0.001902</td>
      <td>-0.003610</td>
      <td>0.076811</td>
      <td>-0.032578</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>-0.059</td>
      <td>-0.096</td>
      <td>0.012</td>
      <td>317.4</td>
      <td>23.63</td>
      <td>165.7</td>
      <td>0.004113</td>
      <td>0.000604</td>
      <td>0.074449</td>
      <td>-0.030757</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>-0.029</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>318.8</td>
      <td>23.88</td>
      <td>166.1</td>
      <td>0.004411</td>
      <td>0.002414</td>
      <td>0.074906</td>
      <td>0.006142</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>0.010</td>
      <td>-0.029</td>
      <td>-0.010</td>
      <td>320.1</td>
      <td>24.15</td>
      <td>166.2</td>
      <td>0.004078</td>
      <td>0.000602</td>
      <td>0.075445</td>
      <td>0.007199</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.158</td>
      <td>-0.091</td>
      <td>0.019</td>
      <td>321.3</td>
      <td>24.18</td>
      <td>166.2</td>
      <td>0.003749</td>
      <td>0.000000</td>
      <td>0.075257</td>
      <td>-0.002497</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.086</td>
      <td>-0.050</td>
      <td>-0.003</td>
      <td>322.3</td>
      <td>24.03</td>
      <td>166.5</td>
      <td>0.003112</td>
      <td>0.001805</td>
      <td>0.074558</td>
      <td>-0.009287</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>-0.026</td>
      <td>0.018</td>
      <td>0.012</td>
      <td>322.8</td>
      <td>24.00</td>
      <td>166.2</td>
      <td>0.001551</td>
      <td>-0.001802</td>
      <td>0.074349</td>
      <td>-0.002795</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.011</td>
      <td>-0.052</td>
      <td>0.005</td>
      <td>323.5</td>
      <td>23.92</td>
      <td>167.7</td>
      <td>0.002169</td>
      <td>0.009025</td>
      <td>0.073941</td>
      <td>-0.005490</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>-0.095</td>
      <td>0.036</td>
      <td>-0.055</td>
      <td>324.5</td>
      <td>23.93</td>
      <td>167.6</td>
      <td>0.003091</td>
      <td>-0.000596</td>
      <td>0.073744</td>
      <td>-0.002665</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>-0.035</td>
      <td>0.105</td>
      <td>0.026</td>
      <td>325.5</td>
      <td>24.06</td>
      <td>166.6</td>
      <td>0.003082</td>
      <td>-0.005967</td>
      <td>0.073917</td>
      <td>0.002344</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.088</td>
      <td>0.048</td>
      <td>0.059</td>
      <td>326.6</td>
      <td>24.31</td>
      <td>167.6</td>
      <td>0.003379</td>
      <td>0.006002</td>
      <td>0.074434</td>
      <td>0.006988</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.064</td>
      <td>0.197</td>
      <td>0.013</td>
      <td>327.4</td>
      <td>24.53</td>
      <td>168.8</td>
      <td>0.002449</td>
      <td>0.007160</td>
      <td>0.074924</td>
      <td>0.006584</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.032</td>
      <td>0.000</td>
      <td>-0.009</td>
      <td>328.4</td>
      <td>23.12</td>
      <td>169.6</td>
      <td>0.003054</td>
      <td>0.004739</td>
      <td>0.070402</td>
      <td>-0.060351</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>0.093</td>
      <td>-0.051</td>
      <td>0.049</td>
      <td>327.5</td>
      <td>17.65</td>
      <td>168.4</td>
      <td>-0.002741</td>
      <td>-0.007075</td>
      <td>0.053893</td>
      <td>-0.234494</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>0.066</td>
      <td>-0.040</td>
      <td>0.048</td>
      <td>326.0</td>
      <td>12.62</td>
      <td>166.1</td>
      <td>-0.004580</td>
      <td>-0.013658</td>
      <td>0.038712</td>
      <td>-0.281696</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.013</td>
      <td>-0.097</td>
      <td>-0.009</td>
      <td>325.3</td>
      <td>10.68</td>
      <td>167.6</td>
      <td>-0.002147</td>
      <td>0.009031</td>
      <td>0.032831</td>
      <td>-0.151903</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 10 columns</p>
</div>




```python
formula_without_intercept = 'CONTIL ~ MARKET + RINF + GIND + ROIL'

results_without_intercept = smf.ols(formula_without_intercept, df4).fit()
print(results_without_intercept.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 CONTIL   R-squared:                       0.124
    Model:                            OLS   Adj. R-squared:                  0.093
    Method:                 Least Squares   F-statistic:                     3.988
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):            0.00460
    Time:                        14:39:21   Log-Likelihood:                 63.586
    No. Observations:                 118   AIC:                            -117.2
    Df Residuals:                     113   BIC:                            -103.3
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0096      0.024     -0.399      0.691      -0.057       0.038
    MARKET         0.6962      0.196      3.550      0.001       0.308       1.085
    RINF           0.5381      3.769      0.143      0.887      -6.929       8.005
    GIND          -1.6486      1.440     -1.145      0.255      -4.502       1.205
    ROIL           0.1740      0.265      0.657      0.512      -0.351       0.699
    ==============================================================================
    Omnibus:                      103.711   Durbin-Watson:                   2.058
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2129.891
    Skew:                           2.634   Prob(JB):                         0.00
    Kurtosis:                      23.136   Cond. No.                         284.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
formula_without_intercept = 'BOISE ~ MARKET + RINF + GIND + ROIL'

results_without_intercept = smf.ols(formula_without_intercept, df4).fit()
print(results_without_intercept.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  BOISE   R-squared:                       0.459
    Model:                            OLS   Adj. R-squared:                  0.440
    Method:                 Least Squares   F-statistic:                     23.97
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           2.26e-14
    Time:                        14:39:22   Log-Likelihood:                 144.34
    No. Observations:                 118   AIC:                            -278.7
    Df Residuals:                     113   BIC:                            -264.8
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.0320      0.012      2.630      0.010       0.008       0.056
    MARKET         0.9047      0.099      9.145      0.000       0.709       1.101
    RINF          -5.1257      1.901     -2.696      0.008      -8.892      -1.359
    GIND          -0.7855      0.727     -1.081      0.282      -2.225       0.654
    ROIL           0.2110      0.134      1.580      0.117      -0.054       0.476
    ==============================================================================
    Omnibus:                        1.808   Durbin-Watson:                   2.325
    Prob(Omnibus):                  0.405   Jarque-Bera (JB):                1.343
    Skew:                           0.110   Prob(JB):                        0.511
    Kurtosis:                       3.474   Cond. No.                         284.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
df3.head(100)
df.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.032</td>
      <td>0.011</td>
      <td>0.066</td>
      <td>0.143</td>
      <td>0.107</td>
      <td>0.185</td>
      <td>0.075</td>
      <td>-0.012</td>
      <td>0.092</td>
      <td>...</td>
      <td>0.042</td>
      <td>0.164</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>196.7</td>
      <td>0.04405</td>
      <td>0.07107</td>
      <td>0.07813</td>
      <td>0.02814</td>
      <td>-0.01422</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.088</td>
      <td>0.024</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>-0.017</td>
      <td>-0.021</td>
      <td>-0.051</td>
      <td>-0.079</td>
      <td>0.049</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.039</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>197.8</td>
      <td>-0.04636</td>
      <td>0.04265</td>
      <td>0.03727</td>
      <td>0.09005</td>
      <td>0.09519</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.011</td>
      <td>0.048</td>
      <td>-0.013</td>
      <td>-0.031</td>
      <td>-0.037</td>
      <td>-0.081</td>
      <td>-0.012</td>
      <td>0.104</td>
      <td>-0.051</td>
      <td>...</td>
      <td>0.010</td>
      <td>-0.021</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>199.3</td>
      <td>0.03472</td>
      <td>0.04000</td>
      <td>0.03024</td>
      <td>0.02977</td>
      <td>0.04889</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.071</td>
      <td>-0.067</td>
      <td>-0.123</td>
      <td>-0.085</td>
      <td>-0.077</td>
      <td>-0.153</td>
      <td>-0.032</td>
      <td>-0.138</td>
      <td>-0.046</td>
      <td>...</td>
      <td>-0.066</td>
      <td>-0.090</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>200.9</td>
      <td>-0.07651</td>
      <td>-0.07522</td>
      <td>-0.06067</td>
      <td>0.07194</td>
      <td>-0.13136</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.005</td>
      <td>0.035</td>
      <td>-0.038</td>
      <td>0.044</td>
      <td>0.064</td>
      <td>0.055</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>202.0</td>
      <td>0.04478</td>
      <td>0.00478</td>
      <td>0.05000</td>
      <td>-0.09443</td>
      <td>0.02927</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.047</td>
      <td>0.034</td>
      <td>0.117</td>
      <td>-0.023</td>
      <td>0.022</td>
      <td>-0.086</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.000</td>
      <td>-0.034</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>203.3</td>
      <td>0.00000</td>
      <td>-0.03905</td>
      <td>0.02857</td>
      <td>0.00941</td>
      <td>0.08173</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>0.043</td>
      <td>0.076</td>
      <td>-0.024</td>
      <td>-0.008</td>
      <td>-0.012</td>
      <td>-0.054</td>
      <td>-0.032</td>
      <td>0.042</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.037</td>
      <td>0.203</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>204.7</td>
      <td>0.05429</td>
      <td>0.07035</td>
      <td>0.06151</td>
      <td>0.09336</td>
      <td>0.06667</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.082</td>
      <td>-0.011</td>
      <td>-0.020</td>
      <td>-0.015</td>
      <td>-0.066</td>
      <td>-0.060</td>
      <td>-0.079</td>
      <td>-0.023</td>
      <td>-0.017</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.038</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>207.1</td>
      <td>-0.05556</td>
      <td>-0.04225</td>
      <td>-0.02150</td>
      <td>0.08042</td>
      <td>0.02500</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.171</td>
      <td>0.088</td>
      <td>0.098</td>
      <td>-0.043</td>
      <td>0.065</td>
      <td>0.052</td>
      <td>...</td>
      <td>0.068</td>
      <td>0.097</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>209.1</td>
      <td>-0.04412</td>
      <td>0.11176</td>
      <td>0.09082</td>
      <td>-0.01428</td>
      <td>0.12757</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.000</td>
      <td>-0.057</td>
      <td>0.064</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>-0.056</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.069</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>211.5</td>
      <td>-0.33077</td>
      <td>-0.07143</td>
      <td>-0.06200</td>
      <td>-0.01333</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.005</td>
      <td>-0.045</td>
      <td>-0.028</td>
      <td>0.063</td>
      <td>0.035</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.040</td>
      <td>-0.013</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>214.1</td>
      <td>-0.17241</td>
      <td>-0.01923</td>
      <td>-0.03305</td>
      <td>0.07745</td>
      <td>0.00511</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.095</td>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>-0.006</td>
      <td>-0.043</td>
      <td>0.095</td>
      <td>-0.035</td>
      <td>...</td>
      <td>0.083</td>
      <td>0.053</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>216.6</td>
      <td>0.15714</td>
      <td>0.02843</td>
      <td>-0.02174</td>
      <td>0.08434</td>
      <td>0.11397</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.075</td>
      <td>0.015</td>
      <td>-0.034</td>
      <td>-0.059</td>
      <td>0.009</td>
      <td>0.075</td>
      <td>-0.013</td>
      <td>-0.096</td>
      <td>-0.049</td>
      <td>...</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>218.9</td>
      <td>-0.01235</td>
      <td>0.08213</td>
      <td>-0.00909</td>
      <td>0.05802</td>
      <td>0.01980</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.041</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>0.030</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.053</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>0.067</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>-0.029</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.229</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.161</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>0.111</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>-0.021</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>0.035</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>-0.052</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>0.011</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>-0.060</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>0.017</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>-0.015</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
  </tbody>
</table>
<p>45 rows × 27 columns</p>
</div>




```python
np.percentile(df3['CONTIL'],25)
```




    -0.069975




```python

```


```python
np.percentile(df3['BOISE'],25)
```




    -0.05036




```python
print("25th percentile of CONTIL stock ", np.percentile(df3['CONTIL'],25))
```

    25th percentile of CONTIL stock  -0.069975



```python
print("25th percentile of CONTIL stock ", np.percentile(df3['CONTIL'],25))
print("75th percentile of CONTIL stock ", np.percentile(df3['CONTIL'],75))
```

    25th percentile of CONTIL stock  -0.069975
    75th percentile of CONTIL stock  0.06158



```python
IQR=np.percentile(df3['CONTIL'],75)-np.percentile(df3['CONTIL'],25)
IQR
```




    0.131555




```python
fig1, ax1 = plt.subplots()
ax1.set_title('Baisc Plot')
ax1.boxplot(df3['CONTIL'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1c1fa13550>,
      <matplotlib.lines.Line2D at 0x1c1fa13668>],
     'caps': [<matplotlib.lines.Line2D at 0x1c1fb60400>,
      <matplotlib.lines.Line2D at 0x1c1fb60780>],
     'boxes': [<matplotlib.lines.Line2D at 0x1c1fa130f0>],
     'medians': [<matplotlib.lines.Line2D at 0x1c1fb60b00>],
     'fliers': [<matplotlib.lines.Line2D at 0x1c1fb60e80>],
     'means': []}




![png](output_90_1.png)



```python
df =pd.read_excel('Data_For_Analysis.xlsx')
df.head(80)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1976-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1976-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1976-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1976-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1976-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1976-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1976-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1976-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1977-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1977-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1977-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1977-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1977-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1977-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1977-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1977-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1977-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1977-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1977-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1977-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1978-01-01</td>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1978-02-01</td>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1978-03-01</td>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1978-04-01</td>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1978-05-01</td>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1978-06-01</td>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1980-03-01</td>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1980-04-01</td>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1980-05-01</td>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1980-06-01</td>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1980-07-01</td>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1980-08-01</td>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1980-09-01</td>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1980-10-01</td>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1980-11-01</td>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1980-12-01</td>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1981-01-01</td>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1981-02-01</td>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1981-03-01</td>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1981-04-01</td>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1981-05-01</td>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1981-06-01</td>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1981-07-01</td>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>67</th>
      <td>1981-08-01</td>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1981-09-01</td>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1981-10-01</td>
      <td>0.016</td>
      <td>0.052</td>
      <td>0.112</td>
      <td>0.049</td>
      <td>0.094</td>
      <td>0.102</td>
      <td>-0.072</td>
      <td>0.067</td>
      <td>-0.012</td>
      <td>...</td>
      <td>0.008</td>
      <td>-0.020</td>
      <td>31.00</td>
      <td>149.1</td>
      <td>279.9</td>
      <td>0.10256</td>
      <td>0.06701</td>
      <td>0.06230</td>
      <td>-0.01430</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1981-11-01</td>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1981-12-01</td>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1982-01-01</td>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1982-02-01</td>
      <td>-0.159</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.175</td>
      <td>-0.152</td>
      <td>-0.049</td>
      <td>0.145</td>
      <td>0.098</td>
      <td>-0.111</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>29.76</td>
      <td>142.9</td>
      <td>283.4</td>
      <td>-0.17021</td>
      <td>-0.10825</td>
      <td>0.00276</td>
      <td>-0.02619</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1982-03-01</td>
      <td>0.108</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>-0.017</td>
      <td>-0.302</td>
      <td>-0.104</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.136</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.009</td>
      <td>28.31</td>
      <td>141.7</td>
      <td>283.1</td>
      <td>-0.05128</td>
      <td>0.09595</td>
      <td>-0.06643</td>
      <td>-0.11714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1982-04-01</td>
      <td>-0.009</td>
      <td>0.101</td>
      <td>0.050</td>
      <td>-0.013</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>-0.025</td>
      <td>0.076</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.008</td>
      <td>0.059</td>
      <td>27.65</td>
      <td>140.2</td>
      <td>284.3</td>
      <td>0.13514</td>
      <td>-0.02151</td>
      <td>0.05993</td>
      <td>0.06141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1982-05-01</td>
      <td>-0.189</td>
      <td>-0.101</td>
      <td>0.016</td>
      <td>-0.091</td>
      <td>-0.180</td>
      <td>-0.056</td>
      <td>0.042</td>
      <td>-0.027</td>
      <td>0.043</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.086</td>
      <td>27.67</td>
      <td>139.2</td>
      <td>287.1</td>
      <td>-0.04762</td>
      <td>-0.05495</td>
      <td>-0.02898</td>
      <td>-0.04610</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1982-06-01</td>
      <td>-0.044</td>
      <td>-0.003</td>
      <td>-0.024</td>
      <td>-0.096</td>
      <td>-0.060</td>
      <td>-0.073</td>
      <td>0.106</td>
      <td>0.050</td>
      <td>-0.033</td>
      <td>...</td>
      <td>-0.017</td>
      <td>-0.015</td>
      <td>28.11</td>
      <td>138.7</td>
      <td>290.6</td>
      <td>-0.02500</td>
      <td>-0.01395</td>
      <td>-0.02222</td>
      <td>-0.05799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1982-07-01</td>
      <td>0.006</td>
      <td>-0.025</td>
      <td>-0.032</td>
      <td>-0.303</td>
      <td>-0.054</td>
      <td>-0.055</td>
      <td>-0.118</td>
      <td>0.038</td>
      <td>0.019</td>
      <td>...</td>
      <td>-0.060</td>
      <td>-0.012</td>
      <td>28.33</td>
      <td>138.8</td>
      <td>292.6</td>
      <td>0.10256</td>
      <td>-0.02410</td>
      <td>-0.08333</td>
      <td>0.07975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1982-08-01</td>
      <td>0.379</td>
      <td>0.077</td>
      <td>0.133</td>
      <td>0.070</td>
      <td>0.216</td>
      <td>0.273</td>
      <td>0.055</td>
      <td>0.032</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.221</td>
      <td>28.18</td>
      <td>138.4</td>
      <td>292.8</td>
      <td>0.09302</td>
      <td>0.22222</td>
      <td>0.17273</td>
      <td>0.07607</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 28 columns</p>
</div>




```python
df =pd.read_excel('Data_For_Analysis.xlsx')
df.head(80)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1976-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1976-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1976-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1976-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1976-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1976-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1976-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1976-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1977-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1977-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1977-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1977-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1977-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1977-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1977-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1977-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1977-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1977-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1977-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1977-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1978-01-01</td>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1978-02-01</td>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1978-03-01</td>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1978-04-01</td>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1978-05-01</td>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1978-06-01</td>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1980-03-01</td>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>...</td>
      <td>-0.179</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1980-04-01</td>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>...</td>
      <td>0.082</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1980-05-01</td>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1980-06-01</td>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.032</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1980-07-01</td>
      <td>0.073</td>
      <td>-0.023</td>
      <td>-0.027</td>
      <td>-0.034</td>
      <td>0.212</td>
      <td>0.183</td>
      <td>0.283</td>
      <td>0.012</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.003</td>
      <td>0.140</td>
      <td>22.26</td>
      <td>140.4</td>
      <td>247.8</td>
      <td>0.08511</td>
      <td>0.08550</td>
      <td>0.02687</td>
      <td>0.07083</td>
      <td>0.02138</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1980-08-01</td>
      <td>-0.045</td>
      <td>0.029</td>
      <td>-0.005</td>
      <td>-0.018</td>
      <td>0.058</td>
      <td>0.081</td>
      <td>-0.056</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>...</td>
      <td>0.031</td>
      <td>-0.041</td>
      <td>22.63</td>
      <td>141.8</td>
      <td>249.4</td>
      <td>-0.19608</td>
      <td>-0.04452</td>
      <td>0.05233</td>
      <td>-0.02459</td>
      <td>-0.02233</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1980-09-01</td>
      <td>0.019</td>
      <td>-0.068</td>
      <td>-0.010</td>
      <td>0.034</td>
      <td>-0.136</td>
      <td>0.045</td>
      <td>-0.053</td>
      <td>-0.013</td>
      <td>0.066</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.064</td>
      <td>22.59</td>
      <td>143.9</td>
      <td>251.7</td>
      <td>0.02439</td>
      <td>-0.00645</td>
      <td>0.00838</td>
      <td>0.07699</td>
      <td>0.05288</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1980-10-01</td>
      <td>-0.054</td>
      <td>-0.049</td>
      <td>-0.021</td>
      <td>0.035</td>
      <td>0.007</td>
      <td>-0.028</td>
      <td>0.046</td>
      <td>-0.073</td>
      <td>0.026</td>
      <td>...</td>
      <td>0.087</td>
      <td>0.017</td>
      <td>23.23</td>
      <td>146.5</td>
      <td>253.9</td>
      <td>-0.09524</td>
      <td>-0.05109</td>
      <td>-0.11911</td>
      <td>-0.02162</td>
      <td>0.08082</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1980-11-01</td>
      <td>0.028</td>
      <td>0.123</td>
      <td>-0.035</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>0.220</td>
      <td>-0.030</td>
      <td>0.023</td>
      <td>...</td>
      <td>0.399</td>
      <td>0.015</td>
      <td>23.92</td>
      <td>148.5</td>
      <td>256.2</td>
      <td>-0.05263</td>
      <td>0.06154</td>
      <td>0.07547</td>
      <td>-0.05855</td>
      <td>0.23881</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1980-12-01</td>
      <td>-0.047</td>
      <td>0.131</td>
      <td>0.131</td>
      <td>0.103</td>
      <td>-0.098</td>
      <td>0.035</td>
      <td>0.040</td>
      <td>0.102</td>
      <td>0.070</td>
      <td>...</td>
      <td>-0.109</td>
      <td>0.007</td>
      <td>25.80</td>
      <td>150.0</td>
      <td>258.4</td>
      <td>0.11111</td>
      <td>-0.05580</td>
      <td>0.01205</td>
      <td>-0.04421</td>
      <td>-0.09983</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1981-01-01</td>
      <td>0.011</td>
      <td>-0.062</td>
      <td>-0.015</td>
      <td>0.040</td>
      <td>-0.231</td>
      <td>-0.089</td>
      <td>0.112</td>
      <td>0.079</td>
      <td>0.056</td>
      <td>...</td>
      <td>-0.145</td>
      <td>0.028</td>
      <td>28.85</td>
      <td>151.4</td>
      <td>260.5</td>
      <td>-0.15000</td>
      <td>0.06615</td>
      <td>0.08036</td>
      <td>-0.06308</td>
      <td>-0.08222</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1981-02-01</td>
      <td>0.152</td>
      <td>-0.005</td>
      <td>-0.021</td>
      <td>0.069</td>
      <td>-0.072</td>
      <td>0.006</td>
      <td>0.031</td>
      <td>0.013</td>
      <td>-0.020</td>
      <td>...</td>
      <td>-0.012</td>
      <td>0.025</td>
      <td>34.10</td>
      <td>151.8</td>
      <td>263.2</td>
      <td>0.05882</td>
      <td>0.07664</td>
      <td>0.08760</td>
      <td>-0.10250</td>
      <td>-0.00792</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1981-03-01</td>
      <td>0.056</td>
      <td>0.045</td>
      <td>0.151</td>
      <td>0.024</td>
      <td>0.184</td>
      <td>0.075</td>
      <td>0.024</td>
      <td>0.146</td>
      <td>0.023</td>
      <td>...</td>
      <td>-0.063</td>
      <td>0.088</td>
      <td>34.70</td>
      <td>152.1</td>
      <td>265.1</td>
      <td>-0.08333</td>
      <td>0.04949</td>
      <td>0.01538</td>
      <td>-0.00300</td>
      <td>-0.03822</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1981-04-01</td>
      <td>0.045</td>
      <td>0.086</td>
      <td>0.061</td>
      <td>-0.025</td>
      <td>0.088</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.019</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.003</td>
      <td>-0.050</td>
      <td>34.05</td>
      <td>151.9</td>
      <td>266.8</td>
      <td>0.15152</td>
      <td>-0.09150</td>
      <td>0.00505</td>
      <td>-0.00774</td>
      <td>-0.12583</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1981-05-01</td>
      <td>0.032</td>
      <td>0.099</td>
      <td>0.017</td>
      <td>0.117</td>
      <td>0.112</td>
      <td>0.107</td>
      <td>0.105</td>
      <td>0.030</td>
      <td>0.008</td>
      <td>...</td>
      <td>-0.055</td>
      <td>-0.031</td>
      <td>32.71</td>
      <td>152.7</td>
      <td>269.0</td>
      <td>0.00000</td>
      <td>-0.07194</td>
      <td>-0.00050</td>
      <td>-0.03053</td>
      <td>0.05606</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1981-06-01</td>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.022</td>
      <td>0.077</td>
      <td>-0.178</td>
      <td>-0.112</td>
      <td>-0.114</td>
      <td>0.094</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.021</td>
      <td>31.71</td>
      <td>152.9</td>
      <td>271.3</td>
      <td>0.10526</td>
      <td>0.04109</td>
      <td>0.08142</td>
      <td>-0.03966</td>
      <td>0.26877</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1981-07-01</td>
      <td>-0.065</td>
      <td>-0.019</td>
      <td>0.026</td>
      <td>-0.092</td>
      <td>0.007</td>
      <td>-0.014</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.021</td>
      <td>...</td>
      <td>0.045</td>
      <td>-0.081</td>
      <td>31.13</td>
      <td>153.9</td>
      <td>274.4</td>
      <td>-0.07143</td>
      <td>-0.05660</td>
      <td>-0.14353</td>
      <td>-0.11260</td>
      <td>0.39313</td>
    </tr>
    <tr>
      <th>67</th>
      <td>1981-08-01</td>
      <td>-0.125</td>
      <td>-0.108</td>
      <td>0.021</td>
      <td>-0.030</td>
      <td>-0.191</td>
      <td>-0.065</td>
      <td>-0.072</td>
      <td>-0.031</td>
      <td>0.031</td>
      <td>...</td>
      <td>0.003</td>
      <td>-0.061</td>
      <td>31.13</td>
      <td>153.6</td>
      <td>276.5</td>
      <td>-0.05128</td>
      <td>-0.12000</td>
      <td>-0.09396</td>
      <td>0.00494</td>
      <td>-0.08603</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1981-09-01</td>
      <td>-0.062</td>
      <td>0.032</td>
      <td>-0.013</td>
      <td>0.003</td>
      <td>0.089</td>
      <td>-0.019</td>
      <td>-0.013</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>...</td>
      <td>-0.093</td>
      <td>-0.113</td>
      <td>31.13</td>
      <td>151.6</td>
      <td>279.3</td>
      <td>0.05405</td>
      <td>-0.10182</td>
      <td>-0.06154</td>
      <td>0.08080</td>
      <td>-0.22356</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1981-10-01</td>
      <td>0.016</td>
      <td>0.052</td>
      <td>0.112</td>
      <td>0.049</td>
      <td>0.094</td>
      <td>0.102</td>
      <td>-0.072</td>
      <td>0.067</td>
      <td>-0.012</td>
      <td>...</td>
      <td>0.008</td>
      <td>-0.020</td>
      <td>31.00</td>
      <td>149.1</td>
      <td>279.9</td>
      <td>0.10256</td>
      <td>0.06701</td>
      <td>0.06230</td>
      <td>-0.01430</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1981-11-01</td>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1981-12-01</td>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1982-01-01</td>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1982-02-01</td>
      <td>-0.159</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.175</td>
      <td>-0.152</td>
      <td>-0.049</td>
      <td>0.145</td>
      <td>0.098</td>
      <td>-0.111</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>29.76</td>
      <td>142.9</td>
      <td>283.4</td>
      <td>-0.17021</td>
      <td>-0.10825</td>
      <td>0.00276</td>
      <td>-0.02619</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1982-03-01</td>
      <td>0.108</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>-0.017</td>
      <td>-0.302</td>
      <td>-0.104</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.136</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.009</td>
      <td>28.31</td>
      <td>141.7</td>
      <td>283.1</td>
      <td>-0.05128</td>
      <td>0.09595</td>
      <td>-0.06643</td>
      <td>-0.11714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1982-04-01</td>
      <td>-0.009</td>
      <td>0.101</td>
      <td>0.050</td>
      <td>-0.013</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>-0.025</td>
      <td>0.076</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.008</td>
      <td>0.059</td>
      <td>27.65</td>
      <td>140.2</td>
      <td>284.3</td>
      <td>0.13514</td>
      <td>-0.02151</td>
      <td>0.05993</td>
      <td>0.06141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1982-05-01</td>
      <td>-0.189</td>
      <td>-0.101</td>
      <td>0.016</td>
      <td>-0.091</td>
      <td>-0.180</td>
      <td>-0.056</td>
      <td>0.042</td>
      <td>-0.027</td>
      <td>0.043</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.086</td>
      <td>27.67</td>
      <td>139.2</td>
      <td>287.1</td>
      <td>-0.04762</td>
      <td>-0.05495</td>
      <td>-0.02898</td>
      <td>-0.04610</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1982-06-01</td>
      <td>-0.044</td>
      <td>-0.003</td>
      <td>-0.024</td>
      <td>-0.096</td>
      <td>-0.060</td>
      <td>-0.073</td>
      <td>0.106</td>
      <td>0.050</td>
      <td>-0.033</td>
      <td>...</td>
      <td>-0.017</td>
      <td>-0.015</td>
      <td>28.11</td>
      <td>138.7</td>
      <td>290.6</td>
      <td>-0.02500</td>
      <td>-0.01395</td>
      <td>-0.02222</td>
      <td>-0.05799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1982-07-01</td>
      <td>0.006</td>
      <td>-0.025</td>
      <td>-0.032</td>
      <td>-0.303</td>
      <td>-0.054</td>
      <td>-0.055</td>
      <td>-0.118</td>
      <td>0.038</td>
      <td>0.019</td>
      <td>...</td>
      <td>-0.060</td>
      <td>-0.012</td>
      <td>28.33</td>
      <td>138.8</td>
      <td>292.6</td>
      <td>0.10256</td>
      <td>-0.02410</td>
      <td>-0.08333</td>
      <td>0.07975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1982-08-01</td>
      <td>0.379</td>
      <td>0.077</td>
      <td>0.133</td>
      <td>0.070</td>
      <td>0.216</td>
      <td>0.273</td>
      <td>0.055</td>
      <td>0.032</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.221</td>
      <td>28.18</td>
      <td>138.4</td>
      <td>292.8</td>
      <td>0.09302</td>
      <td>0.22222</td>
      <td>0.17273</td>
      <td>0.07607</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 28 columns</p>
</div>




```python
sns.boxplot(x="MARKET", y='CONTIL', data=df3, palette="Set1")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1fb7fa90>




![png](output_93_1.png)



```python
boxplot = df3.boxplot()
```


![png](output_94_0.png)



```python
df3.var()
```




    BOISE     0.016534
    CONTIL    0.021317
    MARKET    0.011777
    dtype: float64




```python
df3.std
df3.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CONTIL</th>
      <th>MARKET</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-03-01</th>
      <td>0.05043</td>
      <td>-0.08457</td>
      <td>-0.00457</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.04300</td>
      <td>0.06400</td>
      <td>0.02600</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.03111</td>
      <td>-0.00011</td>
      <td>0.00889</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>-0.05697</td>
      <td>-0.02197</td>
      <td>-0.03897</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.02720</td>
      <td>-0.05380</td>
      <td>-0.03280</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.00860</td>
      <td>0.01140</td>
      <td>0.04840</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.04375</td>
      <td>-0.12675</td>
      <td>-0.05475</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.11671</td>
      <td>-0.10171</td>
      <td>-0.06271</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.07441</td>
      <td>0.01059</td>
      <td>0.04559</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>-0.01387</td>
      <td>0.04813</td>
      <td>0.03013</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.01734</td>
      <td>-0.22734</td>
      <td>-0.00034</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.08600</td>
      <td>-0.10300</td>
      <td>0.09600</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.02789</td>
      <td>0.25089</td>
      <td>-0.07711</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.01402</td>
      <td>0.01798</td>
      <td>0.02398</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.05754</td>
      <td>-0.02846</td>
      <td>0.01054</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.00598</td>
      <td>-0.00998</td>
      <td>0.01202</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.05162</td>
      <td>-0.05538</td>
      <td>0.04362</td>
    </tr>
    <tr>
      <th>1980-08-01</th>
      <td>-0.02267</td>
      <td>0.00433</td>
      <td>0.04733</td>
    </tr>
    <tr>
      <th>1980-09-01</th>
      <td>-0.03388</td>
      <td>-0.01888</td>
      <td>-0.03788</td>
    </tr>
    <tr>
      <th>1980-10-01</th>
      <td>-0.13482</td>
      <td>-0.04582</td>
      <td>-0.07482</td>
    </tr>
    <tr>
      <th>1980-11-01</th>
      <td>-0.21081</td>
      <td>-0.25581</td>
      <td>-0.14681</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>0.05283</td>
      <td>0.20283</td>
      <td>0.04383</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>0.09322</td>
      <td>0.12222</td>
      <td>0.06822</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.15992</td>
      <td>0.07692</td>
      <td>-0.00108</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.09422</td>
      <td>0.06222</td>
      <td>0.10522</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.17083</td>
      <td>0.10083</td>
      <td>0.11783</td>
    </tr>
    <tr>
      <th>1981-05-01</th>
      <td>-0.02406</td>
      <td>0.06094</td>
      <td>0.00794</td>
    </tr>
    <tr>
      <th>1981-06-01</th>
      <td>-0.30577</td>
      <td>-0.19177</td>
      <td>-0.27177</td>
    </tr>
    <tr>
      <th>1981-07-01</th>
      <td>-0.45813</td>
      <td>-0.48513</td>
      <td>-0.42613</td>
    </tr>
    <tr>
      <th>1981-08-01</th>
      <td>-0.03897</td>
      <td>0.05603</td>
      <td>0.05503</td>
    </tr>
    <tr>
      <th>1981-09-01</th>
      <td>0.16156</td>
      <td>0.22656</td>
      <td>0.05956</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(df3['CONTIL'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1f162ba8>




![png](output_97_1.png)



```python
m, s = stats.norm.fit(df3['CONTIL'])
```


```python
m
```




    -0.015955161290322577




```python
s
```




    0.14362765395219923




```python
m, s = stats.norm.fit(df3['BOISE'])
```


```python
m
```




    -0.025600322580645162




```python
s
```




    0.12649389666479696




```python
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)

sns.distplot(df3['BOISE'], color="r", ax=axes[0,0])
sns.distplot(df3['CONTIL'], color="b", ax=axes[0,1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1f76b128>




![png](output_104_1.png)



```python
df1.var()/df1.mean()
```




    BOISE      0.604939
    CITCRP     0.576942
    CONED      0.135915
    CONTIL   -14.960264
    DATGEN     2.679905
    DEC        0.521622
    DELTA      0.852008
    GENMIL     0.265865
    GERBER     0.485193
    IBM        0.375298
    MARKET     0.346923
    MOBIL      0.422214
    MOTOR      0.442681
    PANAM      2.697942
    PSNH      -9.911186
    RKFREE     0.000681
    TANDY      0.659466
    TEXACO     0.585124
    WEYER      0.816287
    POIL       2.842325
    FRBIND     0.873288
    CPI        7.680525
    GPU        1.956240
    DOW        0.453628
    DUPONT     0.317565
    GOLD       0.660820
    CONOCO     0.455283
    dtype: float64




```python
formula ='CONTIL ~ MARKET + RINF + GIND + ROIL'
results = smf.ols(formula, df4).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 CONTIL   R-squared:                       0.124
    Model:                            OLS   Adj. R-squared:                  0.093
    Method:                 Least Squares   F-statistic:                     3.988
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):            0.00460
    Time:                        14:39:43   Log-Likelihood:                 63.586
    No. Observations:                 118   AIC:                            -117.2
    Df Residuals:                     113   BIC:                            -103.3
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0096      0.024     -0.399      0.691      -0.057       0.038
    MARKET         0.6962      0.196      3.550      0.001       0.308       1.085
    RINF           0.5381      3.769      0.143      0.887      -6.929       8.005
    GIND          -1.6486      1.440     -1.145      0.255      -4.502       1.205
    ROIL           0.1740      0.265      0.657      0.512      -0.351       0.699
    ==============================================================================
    Omnibus:                      103.711   Durbin-Watson:                   2.058
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2129.891
    Skew:                           2.634   Prob(JB):                         0.00
    Kurtosis:                      23.136   Cond. No.                         284.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
hypotheses = 'GIND=0, ROIL=0'
f_test=results.f_test(hypotheses)
print(f_test)
```

    <F test: F=array([[0.84148198]]), p=0.43375422200985014, df_denom=113, df_num=2>



```python
formula ='CONTIL ~ MARKET + RINF + ROIL'
results = smf.ols(formula, df4).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 CONTIL   R-squared:                       0.114
    Model:                            OLS   Adj. R-squared:                  0.090
    Method:                 Least Squares   F-statistic:                     4.867
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):            0.00319
    Time:                        14:39:43   Log-Likelihood:                 62.906
    No. Observations:                 118   AIC:                            -117.8
    Df Residuals:                     114   BIC:                            -106.7
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0142      0.024     -0.595      0.553      -0.061       0.033
    MARKET         0.7163      0.196      3.662      0.000       0.329       1.104
    RINF           0.6614      3.772      0.175      0.861      -6.812       8.135
    ROIL           0.1616      0.265      0.610      0.543      -0.363       0.686
    ==============================================================================
    Omnibus:                      106.538   Durbin-Watson:                   2.069
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2279.411
    Skew:                           2.729   Prob(JB):                         0.00
    Kurtosis:                      23.828   Cond. No.                         284.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
formula ='CONTIL ~ MARKET + RINF + ROIL + GIND'
results = smf.ols(formula, df4).fit()
hypotheses = 'GIND'
wald_0 = results.wald_test(hypotheses)
print('H0:', hypotheses)
print(wald_0)
```

    H0: GIND
    <F test: F=array([[1.31001863]]), p=0.25480985797858324, df_denom=113, df_num=1>



```python
df1.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
      <th>Before_Special</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>-0.029</td>
      <td>...</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>-0.043</td>
      <td>...</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>-0.063</td>
      <td>...</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>-0.018</td>
      <td>...</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.140</td>
      <td>0.032</td>
      <td>0.011</td>
      <td>0.066</td>
      <td>0.143</td>
      <td>0.107</td>
      <td>0.185</td>
      <td>0.075</td>
      <td>-0.012</td>
      <td>0.092</td>
      <td>...</td>
      <td>0.164</td>
      <td>8.96</td>
      <td>146.1</td>
      <td>196.7</td>
      <td>0.04405</td>
      <td>0.07107</td>
      <td>0.07813</td>
      <td>0.02814</td>
      <td>-0.01422</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.078</td>
      <td>0.088</td>
      <td>0.024</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>-0.017</td>
      <td>-0.021</td>
      <td>-0.051</td>
      <td>-0.079</td>
      <td>0.049</td>
      <td>...</td>
      <td>0.039</td>
      <td>8.05</td>
      <td>147.1</td>
      <td>197.8</td>
      <td>-0.04636</td>
      <td>0.04265</td>
      <td>0.03727</td>
      <td>0.09005</td>
      <td>0.09519</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.011</td>
      <td>0.048</td>
      <td>-0.013</td>
      <td>-0.031</td>
      <td>-0.037</td>
      <td>-0.081</td>
      <td>-0.012</td>
      <td>0.104</td>
      <td>-0.051</td>
      <td>...</td>
      <td>-0.021</td>
      <td>9.15</td>
      <td>147.8</td>
      <td>199.3</td>
      <td>0.03472</td>
      <td>0.04000</td>
      <td>0.03024</td>
      <td>0.02977</td>
      <td>0.04889</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.118</td>
      <td>-0.071</td>
      <td>-0.067</td>
      <td>-0.123</td>
      <td>-0.085</td>
      <td>-0.077</td>
      <td>-0.153</td>
      <td>-0.032</td>
      <td>-0.138</td>
      <td>-0.046</td>
      <td>...</td>
      <td>-0.090</td>
      <td>9.17</td>
      <td>148.6</td>
      <td>200.9</td>
      <td>-0.07651</td>
      <td>-0.07522</td>
      <td>-0.06067</td>
      <td>0.07194</td>
      <td>-0.13136</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.060</td>
      <td>-0.005</td>
      <td>0.035</td>
      <td>-0.038</td>
      <td>0.044</td>
      <td>0.064</td>
      <td>0.055</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.033</td>
      <td>9.20</td>
      <td>149.5</td>
      <td>202.0</td>
      <td>0.04478</td>
      <td>0.00478</td>
      <td>0.05000</td>
      <td>-0.09443</td>
      <td>0.02927</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.067</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.047</td>
      <td>0.034</td>
      <td>0.117</td>
      <td>-0.023</td>
      <td>0.022</td>
      <td>-0.086</td>
      <td>0.108</td>
      <td>...</td>
      <td>-0.034</td>
      <td>9.47</td>
      <td>150.4</td>
      <td>203.3</td>
      <td>0.00000</td>
      <td>-0.03905</td>
      <td>0.02857</td>
      <td>0.00941</td>
      <td>0.08173</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>0.168</td>
      <td>0.043</td>
      <td>0.076</td>
      <td>-0.024</td>
      <td>-0.008</td>
      <td>-0.012</td>
      <td>-0.054</td>
      <td>-0.032</td>
      <td>0.042</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.203</td>
      <td>9.46</td>
      <td>152.0</td>
      <td>204.7</td>
      <td>0.05429</td>
      <td>0.07035</td>
      <td>0.06151</td>
      <td>0.09336</td>
      <td>0.06667</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.032</td>
      <td>-0.082</td>
      <td>-0.011</td>
      <td>-0.020</td>
      <td>-0.015</td>
      <td>-0.066</td>
      <td>-0.060</td>
      <td>-0.079</td>
      <td>-0.023</td>
      <td>-0.017</td>
      <td>...</td>
      <td>-0.038</td>
      <td>9.69</td>
      <td>152.5</td>
      <td>207.1</td>
      <td>-0.05556</td>
      <td>-0.04225</td>
      <td>-0.02150</td>
      <td>0.08042</td>
      <td>0.02500</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.178</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.171</td>
      <td>0.088</td>
      <td>0.098</td>
      <td>-0.043</td>
      <td>0.065</td>
      <td>0.052</td>
      <td>...</td>
      <td>0.097</td>
      <td>9.83</td>
      <td>153.5</td>
      <td>209.1</td>
      <td>-0.04412</td>
      <td>0.11176</td>
      <td>0.09082</td>
      <td>-0.01428</td>
      <td>0.12757</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>-0.043</td>
      <td>0.000</td>
      <td>-0.057</td>
      <td>0.064</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>-0.056</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.069</td>
      <td>10.33</td>
      <td>151.1</td>
      <td>211.5</td>
      <td>-0.33077</td>
      <td>-0.07143</td>
      <td>-0.06200</td>
      <td>-0.01333</td>
      <td>0.00000</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.026</td>
      <td>0.022</td>
      <td>0.032</td>
      <td>0.005</td>
      <td>-0.045</td>
      <td>-0.028</td>
      <td>0.063</td>
      <td>0.035</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.013</td>
      <td>10.71</td>
      <td>152.7</td>
      <td>214.1</td>
      <td>-0.17241</td>
      <td>-0.01923</td>
      <td>-0.03305</td>
      <td>0.07745</td>
      <td>0.00511</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.057</td>
      <td>0.095</td>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>-0.006</td>
      <td>-0.043</td>
      <td>0.095</td>
      <td>-0.035</td>
      <td>...</td>
      <td>0.053</td>
      <td>11.70</td>
      <td>153.0</td>
      <td>216.6</td>
      <td>0.15714</td>
      <td>0.02843</td>
      <td>-0.02174</td>
      <td>0.08434</td>
      <td>0.11397</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>0.047</td>
      <td>-0.075</td>
      <td>0.015</td>
      <td>-0.034</td>
      <td>-0.059</td>
      <td>0.009</td>
      <td>0.075</td>
      <td>-0.013</td>
      <td>-0.096</td>
      <td>-0.049</td>
      <td>...</td>
      <td>0.000</td>
      <td>13.39</td>
      <td>153.0</td>
      <td>218.9</td>
      <td>-0.01235</td>
      <td>0.08213</td>
      <td>-0.00909</td>
      <td>0.05802</td>
      <td>0.01980</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.038</td>
      <td>0.065</td>
      <td>-0.021</td>
      <td>0.058</td>
      <td>0.078</td>
      <td>0.140</td>
      <td>0.021</td>
      <td>0.138</td>
      <td>0.148</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.165</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>221.1</td>
      <td>0.00000</td>
      <td>0.09375</td>
      <td>0.07034</td>
      <td>0.02064</td>
      <td>0.04660</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>0.050</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.033</td>
      <td>-0.031</td>
      <td>-0.027</td>
      <td>-0.026</td>
      <td>-0.032</td>
      <td>-0.009</td>
      <td>-0.032</td>
      <td>...</td>
      <td>-0.015</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>223.4</td>
      <td>-0.07692</td>
      <td>0.07837</td>
      <td>-0.02312</td>
      <td>0.18394</td>
      <td>0.09375</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.151</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>-0.136</td>
      <td>-0.246</td>
      <td>-0.010</td>
      <td>-0.147</td>
      <td>-0.067</td>
      <td>-0.090</td>
      <td>-0.079</td>
      <td>...</td>
      <td>-0.083</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>225.4</td>
      <td>-0.08333</td>
      <td>-0.08812</td>
      <td>-0.08284</td>
      <td>0.09749</td>
      <td>-0.03429</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>-0.004</td>
      <td>0.030</td>
      <td>0.109</td>
      <td>0.081</td>
      <td>0.062</td>
      <td>0.095</td>
      <td>0.063</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.060</td>
      <td>...</td>
      <td>-0.065</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>227.5</td>
      <td>0.00000</td>
      <td>0.07563</td>
      <td>0.06452</td>
      <td>0.00163</td>
      <td>0.07041</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.042</td>
      <td>0.113</td>
      <td>0.005</td>
      <td>0.104</td>
      <td>0.021</td>
      <td>0.018</td>
      <td>0.020</td>
      <td>0.005</td>
      <td>-0.036</td>
      <td>-0.013</td>
      <td>...</td>
      <td>0.104</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>229.9</td>
      <td>0.07813</td>
      <td>0.01641</td>
      <td>0.00937</td>
      <td>0.16912</td>
      <td>0.05587</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>0.107</td>
      <td>-0.079</td>
      <td>-0.039</td>
      <td>-0.103</td>
      <td>0.157</td>
      <td>0.058</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.048</td>
      <td>0.066</td>
      <td>...</td>
      <td>0.069</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>233.2</td>
      <td>-0.05797</td>
      <td>0.06226</td>
      <td>0.00929</td>
      <td>0.47437</td>
      <td>0.12434</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.070</td>
      <td>-0.080</td>
      <td>-0.061</td>
      <td>-0.087</td>
      <td>0.043</td>
      <td>0.034</td>
      <td>-0.093</td>
      <td>-0.096</td>
      <td>-0.004</td>
      <td>-0.062</td>
      <td>...</td>
      <td>0.033</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>236.4</td>
      <td>-0.24615</td>
      <td>0.02564</td>
      <td>-0.05521</td>
      <td>-0.01418</td>
      <td>0.01600</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>-0.138</td>
      <td>-0.069</td>
      <td>0.006</td>
      <td>0.085</td>
      <td>-0.094</td>
      <td>-0.182</td>
      <td>-0.031</td>
      <td>0.011</td>
      <td>-0.237</td>
      <td>-0.122</td>
      <td>...</td>
      <td>-0.129</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>239.8</td>
      <td>-0.30612</td>
      <td>-0.09571</td>
      <td>-0.08882</td>
      <td>-0.16820</td>
      <td>-0.16589</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.047</td>
      <td>-0.018</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.016</td>
      <td>...</td>
      <td>0.027</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>242.5</td>
      <td>0.38235</td>
      <td>0.04000</td>
      <td>0.05415</td>
      <td>-0.06696</td>
      <td>0.05602</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.109</td>
      <td>0.104</td>
      <td>0.043</td>
      <td>0.023</td>
      <td>-0.043</td>
      <td>0.016</td>
      <td>0.144</td>
      <td>0.180</td>
      <td>0.233</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.089</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>244.9</td>
      <td>0.06383</td>
      <td>0.05385</td>
      <td>0.09247</td>
      <td>-0.00740</td>
      <td>0.05146</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.068</td>
      <td>0.058</td>
      <td>0.040</td>
      <td>0.064</td>
      <td>0.108</td>
      <td>0.021</td>
      <td>0.010</td>
      <td>-0.013</td>
      <td>0.011</td>
      <td>0.061</td>
      <td>...</td>
      <td>-0.026</td>
      <td>21.53</td>
      <td>141.5</td>
      <td>247.6</td>
      <td>-0.06000</td>
      <td>-0.00657</td>
      <td>0.06349</td>
      <td>0.16878</td>
      <td>0.07398</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.147</td>
      <td>0.162</td>
      <td>-0.025</td>
      <td>0.096</td>
      <td>-0.014</td>
      <td>0.065</td>
      <td>0.120</td>
      <td>-0.014</td>
      <td>0.077</td>
      <td>-0.066</td>
      <td>...</td>
      <td>0.151</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>303.1</td>
      <td>0.06667</td>
      <td>-0.05072</td>
      <td>0.06214</td>
      <td>-0.02957</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.012</td>
      <td>0.023</td>
      <td>0.005</td>
      <td>-0.016</td>
      <td>0.068</td>
      <td>0.034</td>
      <td>-0.028</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.039</td>
      <td>...</td>
      <td>-0.069</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>303.5</td>
      <td>-0.03125</td>
      <td>0.03282</td>
      <td>-0.03704</td>
      <td>0.01488</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.054</td>
      <td>0.024</td>
      <td>0.005</td>
      <td>-0.034</td>
      <td>0.117</td>
      <td>0.208</td>
      <td>-0.013</td>
      <td>-0.009</td>
      <td>0.065</td>
      <td>-0.065</td>
      <td>...</td>
      <td>-0.039</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>305.4</td>
      <td>-0.01613</td>
      <td>-0.05243</td>
      <td>-0.04327</td>
      <td>-0.04322</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.088</td>
      <td>-0.039</td>
      <td>-0.069</td>
      <td>-0.101</td>
      <td>0.027</td>
      <td>-0.024</td>
      <td>-0.117</td>
      <td>-0.073</td>
      <td>-0.091</td>
      <td>-0.026</td>
      <td>...</td>
      <td>-0.093</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>306.6</td>
      <td>0.03279</td>
      <td>-0.11462</td>
      <td>-0.03367</td>
      <td>0.04059</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>0.079</td>
      <td>-0.054</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>0.056</td>
      <td>0.057</td>
      <td>0.065</td>
      <td>-0.018</td>
      <td>-0.003</td>
      <td>0.034</td>
      <td>...</td>
      <td>0.094</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>307.3</td>
      <td>0.04762</td>
      <td>0.15000</td>
      <td>0.03958</td>
      <td>0.02159</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>0.012</td>
      <td>-0.004</td>
      <td>0.031</td>
      <td>-0.231</td>
      <td>0.089</td>
      <td>0.053</td>
      <td>-0.085</td>
      <td>0.065</td>
      <td>-0.025</td>
      <td>-0.002</td>
      <td>...</td>
      <td>-0.088</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>308.8</td>
      <td>-0.01515</td>
      <td>0.01969</td>
      <td>0.02538</td>
      <td>-0.03213</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-05-01</th>
      <td>-0.172</td>
      <td>-0.148</td>
      <td>0.021</td>
      <td>-0.600</td>
      <td>-0.094</td>
      <td>-0.071</td>
      <td>-0.070</td>
      <td>0.018</td>
      <td>-0.087</td>
      <td>-0.044</td>
      <td>...</td>
      <td>-0.087</td>
      <td>26.00</td>
      <td>162.8</td>
      <td>309.7</td>
      <td>0.07692</td>
      <td>-0.12355</td>
      <td>-0.05545</td>
      <td>-0.01126</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-06-01</th>
      <td>0.025</td>
      <td>0.078</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>-0.043</td>
      <td>-0.012</td>
      <td>0.055</td>
      <td>0.105</td>
      <td>-0.019</td>
      <td>...</td>
      <td>0.019</td>
      <td>26.09</td>
      <td>164.4</td>
      <td>310.7</td>
      <td>0.02857</td>
      <td>0.00264</td>
      <td>-0.02926</td>
      <td>0.00103</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-07-01</th>
      <td>0.015</td>
      <td>-0.029</td>
      <td>0.054</td>
      <td>-0.205</td>
      <td>-0.061</td>
      <td>-0.009</td>
      <td>0.045</td>
      <td>-0.018</td>
      <td>-0.112</td>
      <td>0.047</td>
      <td>...</td>
      <td>0.036</td>
      <td>26.11</td>
      <td>165.9</td>
      <td>311.7</td>
      <td>0.08333</td>
      <td>0.01339</td>
      <td>-0.03014</td>
      <td>-0.08266</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-08-01</th>
      <td>0.177</td>
      <td>0.164</td>
      <td>0.029</td>
      <td>0.086</td>
      <td>0.312</td>
      <td>0.159</td>
      <td>0.040</td>
      <td>0.061</td>
      <td>0.018</td>
      <td>0.127</td>
      <td>...</td>
      <td>0.055</td>
      <td>26.02</td>
      <td>166.0</td>
      <td>313.0</td>
      <td>0.02564</td>
      <td>0.10132</td>
      <td>0.14689</td>
      <td>0.00360</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-09-01</th>
      <td>-0.056</td>
      <td>0.076</td>
      <td>0.051</td>
      <td>0.974</td>
      <td>-0.132</td>
      <td>-0.025</td>
      <td>0.008</td>
      <td>0.011</td>
      <td>0.165</td>
      <td>0.004</td>
      <td>...</td>
      <td>-0.069</td>
      <td>25.97</td>
      <td>165.0</td>
      <td>314.5</td>
      <td>0.01250</td>
      <td>-0.08160</td>
      <td>-0.00750</td>
      <td>-0.01942</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-10-01</th>
      <td>0.053</td>
      <td>-0.027</td>
      <td>0.019</td>
      <td>-0.232</td>
      <td>0.047</td>
      <td>0.093</td>
      <td>0.161</td>
      <td>-0.010</td>
      <td>-0.160</td>
      <td>0.012</td>
      <td>...</td>
      <td>0.035</td>
      <td>25.92</td>
      <td>164.5</td>
      <td>315.3</td>
      <td>0.08642</td>
      <td>0.02655</td>
      <td>-0.05542</td>
      <td>-0.00217</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-11-01</th>
      <td>-0.038</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>-0.023</td>
      <td>0.019</td>
      <td>0.006</td>
      <td>-0.026</td>
      <td>-0.072</td>
      <td>0.094</td>
      <td>-0.023</td>
      <td>...</td>
      <td>0.032</td>
      <td>25.44</td>
      <td>165.2</td>
      <td>315.3</td>
      <td>0.02273</td>
      <td>-0.00862</td>
      <td>0.00800</td>
      <td>0.00285</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1984-12-01</th>
      <td>0.068</td>
      <td>0.098</td>
      <td>0.084</td>
      <td>0.095</td>
      <td>0.096</td>
      <td>0.070</td>
      <td>0.156</td>
      <td>0.017</td>
      <td>-0.005</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.026</td>
      <td>25.05</td>
      <td>166.2</td>
      <td>315.5</td>
      <td>0.02222</td>
      <td>-0.02783</td>
      <td>0.06452</td>
      <td>-0.06479</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-01-01</th>
      <td>0.046</td>
      <td>0.097</td>
      <td>-0.021</td>
      <td>0.587</td>
      <td>0.215</td>
      <td>0.084</td>
      <td>-0.010</td>
      <td>0.095</td>
      <td>0.091</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.084</td>
      <td>24.28</td>
      <td>165.6</td>
      <td>316.1</td>
      <td>0.00000</td>
      <td>0.07727</td>
      <td>0.04293</td>
      <td>-0.06265</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>-0.059</td>
      <td>-0.015</td>
      <td>0.034</td>
      <td>-0.096</td>
      <td>-0.210</td>
      <td>-0.067</td>
      <td>0.087</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>-0.009</td>
      <td>...</td>
      <td>-0.016</td>
      <td>23.63</td>
      <td>165.7</td>
      <td>317.4</td>
      <td>0.08696</td>
      <td>0.00844</td>
      <td>0.03148</td>
      <td>0.01720</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>-0.029</td>
      <td>0.046</td>
      <td>0.057</td>
      <td>0.030</td>
      <td>-0.195</td>
      <td>-0.071</td>
      <td>-0.003</td>
      <td>0.054</td>
      <td>0.130</td>
      <td>-0.052</td>
      <td>...</td>
      <td>-0.081</td>
      <td>23.88</td>
      <td>166.1</td>
      <td>318.8</td>
      <td>-0.04000</td>
      <td>-0.01423</td>
      <td>-0.01190</td>
      <td>-0.04400</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>0.010</td>
      <td>0.012</td>
      <td>0.019</td>
      <td>-0.029</td>
      <td>-0.157</td>
      <td>-0.050</td>
      <td>-0.123</td>
      <td>-0.083</td>
      <td>-0.037</td>
      <td>-0.004</td>
      <td>...</td>
      <td>0.003</td>
      <td>24.15</td>
      <td>166.2</td>
      <td>320.1</td>
      <td>-0.01042</td>
      <td>0.03448</td>
      <td>0.06988</td>
      <td>0.13447</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.158</td>
      <td>0.094</td>
      <td>0.098</td>
      <td>-0.091</td>
      <td>-0.078</td>
      <td>0.057</td>
      <td>0.179</td>
      <td>0.137</td>
      <td>0.234</td>
      <td>0.025</td>
      <td>...</td>
      <td>0.031</td>
      <td>24.18</td>
      <td>166.2</td>
      <td>321.3</td>
      <td>0.17895</td>
      <td>0.13333</td>
      <td>0.10135</td>
      <td>-0.02165</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.086</td>
      <td>0.043</td>
      <td>0.046</td>
      <td>-0.050</td>
      <td>0.060</td>
      <td>-0.101</td>
      <td>0.021</td>
      <td>0.060</td>
      <td>-0.031</td>
      <td>-0.038</td>
      <td>...</td>
      <td>-0.004</td>
      <td>24.03</td>
      <td>166.5</td>
      <td>322.3</td>
      <td>0.00893</td>
      <td>0.06471</td>
      <td>-0.03727</td>
      <td>-0.01393</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>-0.026</td>
      <td>-0.030</td>
      <td>-0.084</td>
      <td>0.018</td>
      <td>0.043</td>
      <td>0.080</td>
      <td>0.008</td>
      <td>-0.099</td>
      <td>-0.036</td>
      <td>0.062</td>
      <td>...</td>
      <td>0.020</td>
      <td>24.00</td>
      <td>166.2</td>
      <td>322.8</td>
      <td>-0.06195</td>
      <td>0.03147</td>
      <td>0.03011</td>
      <td>-0.00816</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.011</td>
      <td>-0.063</td>
      <td>0.043</td>
      <td>-0.052</td>
      <td>-0.006</td>
      <td>0.032</td>
      <td>-0.066</td>
      <td>0.002</td>
      <td>0.025</td>
      <td>-0.028</td>
      <td>...</td>
      <td>-0.013</td>
      <td>23.92</td>
      <td>167.7</td>
      <td>323.5</td>
      <td>0.08491</td>
      <td>-0.02712</td>
      <td>-0.02296</td>
      <td>0.03275</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>-0.095</td>
      <td>-0.085</td>
      <td>-0.032</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.112</td>
      <td>0.081</td>
      <td>-0.048</td>
      <td>-0.022</td>
      <td>...</td>
      <td>-0.074</td>
      <td>23.93</td>
      <td>167.6</td>
      <td>324.5</td>
      <td>-0.04348</td>
      <td>-0.02927</td>
      <td>-0.00649</td>
      <td>-0.01440</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>-0.035</td>
      <td>0.090</td>
      <td>0.066</td>
      <td>0.105</td>
      <td>0.032</td>
      <td>0.040</td>
      <td>-0.083</td>
      <td>0.013</td>
      <td>0.097</td>
      <td>0.048</td>
      <td>...</td>
      <td>0.008</td>
      <td>24.06</td>
      <td>166.6</td>
      <td>325.5</td>
      <td>0.14545</td>
      <td>0.06182</td>
      <td>0.08497</td>
      <td>0.01663</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.088</td>
      <td>0.062</td>
      <td>0.032</td>
      <td>0.048</td>
      <td>0.109</td>
      <td>0.073</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.137</td>
      <td>0.085</td>
      <td>...</td>
      <td>0.171</td>
      <td>24.31</td>
      <td>167.6</td>
      <td>326.6</td>
      <td>0.03175</td>
      <td>0.06507</td>
      <td>0.03012</td>
      <td>-0.00413</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.064</td>
      <td>0.065</td>
      <td>0.082</td>
      <td>0.197</td>
      <td>0.023</td>
      <td>0.095</td>
      <td>0.030</td>
      <td>0.027</td>
      <td>0.063</td>
      <td>0.113</td>
      <td>...</td>
      <td>-0.004</td>
      <td>24.53</td>
      <td>168.8</td>
      <td>327.4</td>
      <td>0.04615</td>
      <td>0.06624</td>
      <td>0.07101</td>
      <td>-0.02318</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.032</td>
      <td>0.005</td>
      <td>0.022</td>
      <td>0.000</td>
      <td>-0.055</td>
      <td>0.162</td>
      <td>0.122</td>
      <td>0.019</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>...</td>
      <td>0.072</td>
      <td>23.12</td>
      <td>169.6</td>
      <td>328.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>0.093</td>
      <td>0.101</td>
      <td>0.048</td>
      <td>-0.051</td>
      <td>-0.044</td>
      <td>0.093</td>
      <td>-0.055</td>
      <td>0.121</td>
      <td>0.034</td>
      <td>0.003</td>
      <td>...</td>
      <td>0.123</td>
      <td>17.65</td>
      <td>168.4</td>
      <td>327.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>0.066</td>
      <td>0.153</td>
      <td>0.021</td>
      <td>-0.040</td>
      <td>-0.043</td>
      <td>-0.063</td>
      <td>0.076</td>
      <td>0.072</td>
      <td>0.174</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.051</td>
      <td>12.62</td>
      <td>166.1</td>
      <td>326.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.013</td>
      <td>-0.042</td>
      <td>-0.006</td>
      <td>-0.097</td>
      <td>0.061</td>
      <td>0.119</td>
      <td>0.059</td>
      <td>-0.051</td>
      <td>0.113</td>
      <td>0.031</td>
      <td>...</td>
      <td>-0.037</td>
      <td>10.68</td>
      <td>167.6</td>
      <td>325.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Before</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 28 columns</p>
</div>




```python
df1[['CONTIL','CITCRP']].plot(figsize=(15,8),linewidth=1.5,fontsize=20)
plt.xlabel('Date',fontsize=20);
```


![png](output_111_0.png)



```python
df1[['BOISE','WEYER']].plot(figsize=(15,8),linewidth=1.5,fontsize=20)
plt.xlabel('Date',fontsize=20);
```


![png](output_112_0.png)



```python
formula = 'Q("CONTIL") ~ Q("MARKET")'
results = smf.ols(formula, df3).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CONTIL")   R-squared:                       0.500
    Model:                            OLS   Adj. R-squared:                  0.483
    Method:                 Least Squares   F-statistic:                     29.04
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           8.60e-06
    Time:                        14:39:49   Log-Likelihood:                 26.925
    No. Observations:                  31   AIC:                            -49.85
    Df Residuals:                      29   BIC:                            -46.98
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0043      0.019     -0.227      0.822      -0.043       0.035
    Q("MARKET")     0.9517      0.177      5.389      0.000       0.590       1.313
    ==============================================================================
    Omnibus:                        9.455   Durbin-Watson:                   2.064
    Prob(Omnibus):                  0.009   Jarque-Bera (JB):               10.072
    Skew:                           0.777   Prob(JB):                      0.00650
    Kurtosis:                       5.320   Cond. No.                         9.37
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
hypotheses = 'Intercept=0'
Simple_ttest_Ols(results, hypotheses, alternative='larger', level_of_sig = 0.05)
```

    We accept the null hypothesis: Intercept=0 with a 5.0 % significance level



```python
hypotheses = 'Q("MARKET")=0'
Simple_ttest_Ols(results, hypotheses, alternative='smaller', level_of_sig = 0.05)
```

    We accept the null hypothesis: Q("MARKET")=0 with a 5.0 % significance level



```python
formula = 'Q("CITCRP") ~ Q("MARKET")'
results = smf.ols(formula, df).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CITCRP")   R-squared:                       0.316
    Model:                            OLS   Adj. R-squared:                  0.310
    Method:                 Least Squares   F-statistic:                     53.95
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           2.99e-11
    Time:                        14:39:49   Log-Likelihood:                 153.04
    No. Observations:                 119   AIC:                            -302.1
    Df Residuals:                     117   BIC:                            -296.5
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       0.0024      0.006      0.385      0.701      -0.010       0.015
    Q("MARKET")     0.6664      0.091      7.345      0.000       0.487       0.846
    ==============================================================================
    Omnibus:                        1.688   Durbin-Watson:                   1.820
    Prob(Omnibus):                  0.430   Jarque-Bera (JB):                1.213
    Skew:                           0.110   Prob(JB):                        0.545
    Kurtosis:                       3.443   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
hypotheses = 'Q("MARKET")=0'
Simple_ttest_Ols(results, hypotheses, alternative='smaller', level_of_sig = 0.05)
```

    We accept the null hypothesis: Q("MARKET")=0 with a 5.0 % significance level



```python
hypotheses = 'Intercept=0'
Simple_ttest_Ols(results, hypotheses, alternative='larger', level_of_sig = 0.05)
```

    We accept the null hypothesis: Intercept=0 with a 5.0 % significance level



```python
formula = 'Q("CONTIL") ~ Q("MARKET")'

results = smf.ols(formula, df1).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CONTIL")   R-squared:                       0.111
    Model:                            OLS   Adj. R-squared:                  0.104
    Method:                 Least Squares   F-statistic:                     14.65
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           0.000209
    Time:                        14:39:49   Log-Likelihood:                 63.430
    No. Observations:                 119   AIC:                            -122.9
    Df Residuals:                     117   BIC:                            -117.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0115      0.013     -0.858      0.393      -0.038       0.015
    Q("MARKET")     0.7375      0.193      3.828      0.000       0.356       1.119
    ==============================================================================
    Omnibus:                      106.122   Durbin-Watson:                   2.068
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2224.184
    Skew:                           2.697   Prob(JB):                         0.00
    Kurtosis:                      23.481   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
RF.Chow_Test(df1, y='CONTIL', x='MARKET', special_date='1984-11-01')
```




    (2.5869412758378107, 0.07951319460600349)




![png](output_120_1.png)



```python
formula = 'Q("BOISE") ~ Q("MARKET")'

results = smf.ols(formula, df1).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             Q("BOISE")   R-squared:                       0.422
    Model:                            OLS   Adj. R-squared:                  0.417
    Method:                 Least Squares   F-statistic:                     85.45
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           1.31e-15
    Time:                        14:39:50   Log-Likelihood:                 141.64
    No. Observations:                 119   AIC:                            -279.3
    Df Residuals:                     117   BIC:                            -273.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       0.0032      0.007      0.456      0.649      -0.011       0.017
    Q("MARKET")     0.9230      0.100      9.244      0.000       0.725       1.121
    ==============================================================================
    Omnibus:                        4.937   Durbin-Watson:                   2.183
    Prob(Omnibus):                  0.085   Jarque-Bera (JB):                5.734
    Skew:                           0.215   Prob(JB):                       0.0569
    Kurtosis:                       3.986   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
RF.Chow_Test(df1, y='BOISE', x='MARKET', special_date='1986-11-01')
```




    (1.0439225070169136, 0.35529933592969426)




![png](output_122_1.png)



```python
formula = 'Q("CONTIL") ~ Q("MARKET")'
results = smf.ols(formula, df1).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            Q("CONTIL")   R-squared:                       0.111
    Model:                            OLS   Adj. R-squared:                  0.104
    Method:                 Least Squares   F-statistic:                     14.65
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           0.000209
    Time:                        14:39:51   Log-Likelihood:                 63.430
    No. Observations:                 119   AIC:                            -122.9
    Df Residuals:                     117   BIC:                            -117.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept      -0.0115      0.013     -0.858      0.393      -0.038       0.015
    Q("MARKET")     0.7375      0.193      3.828      0.000       0.356       1.119
    ==============================================================================
    Omnibus:                      106.122   Durbin-Watson:                   2.068
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2224.184
    Skew:                           2.697   Prob(JB):                         0.00
    Kurtosis:                      23.481   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
formula = 'Q("BOISE") ~ Q("MARKET")'
results = smf.ols(formula, df1).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             Q("BOISE")   R-squared:                       0.422
    Model:                            OLS   Adj. R-squared:                  0.417
    Method:                 Least Squares   F-statistic:                     85.45
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           1.31e-15
    Time:                        14:39:51   Log-Likelihood:                 141.64
    No. Observations:                 119   AIC:                            -279.3
    Df Residuals:                     117   BIC:                            -273.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       0.0032      0.007      0.456      0.649      -0.011       0.017
    Q("MARKET")     0.9230      0.100      9.244      0.000       0.725       1.121
    ==============================================================================
    Omnibus:                        4.937   Durbin-Watson:                   2.183
    Prob(Omnibus):                  0.085   Jarque-Bera (JB):                5.734
    Skew:                           0.215   Prob(JB):                       0.0569
    Kurtosis:                       3.986   Cond. No.                         14.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1976-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1976-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.99</td>
      <td>129.8</td>
      <td>170.1</td>
      <td>0.01515</td>
      <td>-0.01772</td>
      <td>-0.07692</td>
      <td>-0.00965</td>
      <td>0.10432</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1976-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>130.7</td>
      <td>171.1</td>
      <td>0.05493</td>
      <td>-0.02591</td>
      <td>-0.01254</td>
      <td>-0.06505</td>
      <td>-0.04235</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1976-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.03</td>
      <td>131.3</td>
      <td>171.9</td>
      <td>0.05797</td>
      <td>-0.04255</td>
      <td>-0.05626</td>
      <td>-0.06703</td>
      <td>0.01497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1976-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>130.6</td>
      <td>172.6</td>
      <td>0.04110</td>
      <td>-0.00556</td>
      <td>-0.01748</td>
      <td>0.04142</td>
      <td>0.04054</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1976-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.46</td>
      <td>130.2</td>
      <td>173.3</td>
      <td>-0.01737</td>
      <td>-0.01966</td>
      <td>0.02174</td>
      <td>0.01736</td>
      <td>-0.05065</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1976-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>131.5</td>
      <td>173.8</td>
      <td>0.00685</td>
      <td>-0.10602</td>
      <td>-0.03578</td>
      <td>0.12637</td>
      <td>0.00690</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1976-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>133.0</td>
      <td>174.5</td>
      <td>0.06122</td>
      <td>0.11859</td>
      <td>0.09969</td>
      <td>0.02306</td>
      <td>0.02740</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1977-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>132.3</td>
      <td>175.3</td>
      <td>0.02154</td>
      <td>-0.11816</td>
      <td>-0.04255</td>
      <td>-0.01109</td>
      <td>-0.03667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1977-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
      <td>133.3</td>
      <td>177.1</td>
      <td>-0.05769</td>
      <td>-0.03595</td>
      <td>-0.00676</td>
      <td>0.02874</td>
      <td>-0.01246</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1977-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.45</td>
      <td>135.3</td>
      <td>178.2</td>
      <td>0.02041</td>
      <td>0.02712</td>
      <td>-0.01179</td>
      <td>0.08703</td>
      <td>-0.01413</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1977-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>136.1</td>
      <td>179.6</td>
      <td>0.02240</td>
      <td>-0.01329</td>
      <td>-0.00099</td>
      <td>0.00700</td>
      <td>0.03943</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1977-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.49</td>
      <td>137.0</td>
      <td>180.6</td>
      <td>0.04000</td>
      <td>-0.05387</td>
      <td>-0.04577</td>
      <td>-0.01637</td>
      <td>-0.09034</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1977-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.44</td>
      <td>137.8</td>
      <td>181.8</td>
      <td>0.02564</td>
      <td>-0.01993</td>
      <td>-0.02213</td>
      <td>-0.04394</td>
      <td>0.03831</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1977-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.48</td>
      <td>138.7</td>
      <td>182.6</td>
      <td>0.02824</td>
      <td>-0.07692</td>
      <td>0.02263</td>
      <td>0.04845</td>
      <td>-0.06273</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1977-08-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.62</td>
      <td>138.1</td>
      <td>183.3</td>
      <td>0.02659</td>
      <td>-0.01984</td>
      <td>-0.04215</td>
      <td>-0.01301</td>
      <td>-0.05197</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1977-09-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.63</td>
      <td>138.5</td>
      <td>184.0</td>
      <td>0.02424</td>
      <td>0.01781</td>
      <td>-0.02225</td>
      <td>0.03048</td>
      <td>-0.00420</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1977-10-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>138.9</td>
      <td>184.5</td>
      <td>-0.03243</td>
      <td>-0.07229</td>
      <td>0.02389</td>
      <td>0.06156</td>
      <td>-0.04219</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1977-11-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>139.3</td>
      <td>185.4</td>
      <td>0.04375</td>
      <td>-0.06494</td>
      <td>0.06444</td>
      <td>-0.02912</td>
      <td>0.05198</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1977-12-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.77</td>
      <td>139.7</td>
      <td>186.1</td>
      <td>0.00000</td>
      <td>0.00185</td>
      <td>0.02229</td>
      <td>0.04163</td>
      <td>0.01695</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1978-01-01</td>
      <td>-0.079</td>
      <td>-0.115</td>
      <td>-0.079</td>
      <td>-0.129</td>
      <td>-0.084</td>
      <td>-0.100</td>
      <td>-0.028</td>
      <td>-0.099</td>
      <td>-0.048</td>
      <td>...</td>
      <td>-0.054</td>
      <td>-0.116</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>187.2</td>
      <td>-0.06874</td>
      <td>-0.04673</td>
      <td>-0.11319</td>
      <td>0.07788</td>
      <td>-0.11250</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1978-02-01</td>
      <td>0.013</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>0.037</td>
      <td>-0.097</td>
      <td>-0.063</td>
      <td>-0.033</td>
      <td>0.018</td>
      <td>0.160</td>
      <td>...</td>
      <td>-0.010</td>
      <td>-0.135</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>188.4</td>
      <td>0.01974</td>
      <td>-0.11765</td>
      <td>-0.07260</td>
      <td>0.02821</td>
      <td>0.01315</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1978-03-01</td>
      <td>0.070</td>
      <td>0.059</td>
      <td>0.022</td>
      <td>0.003</td>
      <td>0.063</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>...</td>
      <td>0.015</td>
      <td>0.084</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>189.8</td>
      <td>0.02581</td>
      <td>0.05778</td>
      <td>0.03453</td>
      <td>0.03057</td>
      <td>0.00469</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1978-04-01</td>
      <td>0.120</td>
      <td>0.127</td>
      <td>-0.005</td>
      <td>0.180</td>
      <td>0.179</td>
      <td>0.165</td>
      <td>0.150</td>
      <td>0.046</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.144</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>191.5</td>
      <td>-0.00931</td>
      <td>0.14362</td>
      <td>0.15451</td>
      <td>-0.04548</td>
      <td>0.05607</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1978-05-01</td>
      <td>0.071</td>
      <td>0.005</td>
      <td>-0.014</td>
      <td>0.061</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>-0.031</td>
      <td>0.063</td>
      <td>0.046</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>193.3</td>
      <td>-0.03247</td>
      <td>-0.04651</td>
      <td>0.00000</td>
      <td>0.00450</td>
      <td>0.04779</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1978-06-01</td>
      <td>-0.098</td>
      <td>0.007</td>
      <td>0.034</td>
      <td>-0.059</td>
      <td>-0.023</td>
      <td>-0.021</td>
      <td>0.023</td>
      <td>0.008</td>
      <td>0.028</td>
      <td>...</td>
      <td>-0.025</td>
      <td>0.005</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>195.3</td>
      <td>-0.00671</td>
      <td>-0.02732</td>
      <td>-0.03030</td>
      <td>0.04295</td>
      <td>-0.09829</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1981-11-01</td>
      <td>0.092</td>
      <td>0.045</td>
      <td>0.038</td>
      <td>0.010</td>
      <td>0.093</td>
      <td>-0.065</td>
      <td>-0.032</td>
      <td>-0.030</td>
      <td>0.011</td>
      <td>...</td>
      <td>0.065</td>
      <td>0.179</td>
      <td>30.98</td>
      <td>146.3</td>
      <td>280.7</td>
      <td>0.18605</td>
      <td>0.00966</td>
      <td>0.01420</td>
      <td>-0.05686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1981-12-01</td>
      <td>-0.029</td>
      <td>-0.028</td>
      <td>-0.008</td>
      <td>-0.106</td>
      <td>-0.083</td>
      <td>-0.060</td>
      <td>-0.062</td>
      <td>-0.024</td>
      <td>-0.077</td>
      <td>...</td>
      <td>-0.047</td>
      <td>-0.072</td>
      <td>30.72</td>
      <td>143.4</td>
      <td>281.5</td>
      <td>0.05882</td>
      <td>0.02201</td>
      <td>-0.07165</td>
      <td>-0.00855</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1982-01-01</td>
      <td>-0.084</td>
      <td>0.035</td>
      <td>0.042</td>
      <td>0.102</td>
      <td>-0.002</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>-0.030</td>
      <td>-0.004</td>
      <td>...</td>
      <td>-0.045</td>
      <td>-0.079</td>
      <td>30.87</td>
      <td>140.7</td>
      <td>282.5</td>
      <td>-0.12963</td>
      <td>-0.07619</td>
      <td>-0.02685</td>
      <td>-0.06156</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1982-02-01</td>
      <td>-0.159</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.175</td>
      <td>-0.152</td>
      <td>-0.049</td>
      <td>0.145</td>
      <td>0.098</td>
      <td>-0.111</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>29.76</td>
      <td>142.9</td>
      <td>283.4</td>
      <td>-0.17021</td>
      <td>-0.10825</td>
      <td>0.00276</td>
      <td>-0.02619</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1982-03-01</td>
      <td>0.108</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>-0.017</td>
      <td>-0.302</td>
      <td>-0.104</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.136</td>
      <td>...</td>
      <td>-0.029</td>
      <td>-0.009</td>
      <td>28.31</td>
      <td>141.7</td>
      <td>283.1</td>
      <td>-0.05128</td>
      <td>0.09595</td>
      <td>-0.06643</td>
      <td>-0.11714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1982-04-01</td>
      <td>-0.009</td>
      <td>0.101</td>
      <td>0.050</td>
      <td>-0.013</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>-0.025</td>
      <td>0.076</td>
      <td>0.044</td>
      <td>...</td>
      <td>-0.008</td>
      <td>0.059</td>
      <td>27.65</td>
      <td>140.2</td>
      <td>284.3</td>
      <td>0.13514</td>
      <td>-0.02151</td>
      <td>0.05993</td>
      <td>0.06141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1982-05-01</td>
      <td>-0.189</td>
      <td>-0.101</td>
      <td>0.016</td>
      <td>-0.091</td>
      <td>-0.180</td>
      <td>-0.056</td>
      <td>0.042</td>
      <td>-0.027</td>
      <td>0.043</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.086</td>
      <td>27.67</td>
      <td>139.2</td>
      <td>287.1</td>
      <td>-0.04762</td>
      <td>-0.05495</td>
      <td>-0.02898</td>
      <td>-0.04610</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1982-06-01</td>
      <td>-0.044</td>
      <td>-0.003</td>
      <td>-0.024</td>
      <td>-0.096</td>
      <td>-0.060</td>
      <td>-0.073</td>
      <td>0.106</td>
      <td>0.050</td>
      <td>-0.033</td>
      <td>...</td>
      <td>-0.017</td>
      <td>-0.015</td>
      <td>28.11</td>
      <td>138.7</td>
      <td>290.6</td>
      <td>-0.02500</td>
      <td>-0.01395</td>
      <td>-0.02222</td>
      <td>-0.05799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1982-07-01</td>
      <td>0.006</td>
      <td>-0.025</td>
      <td>-0.032</td>
      <td>-0.303</td>
      <td>-0.054</td>
      <td>-0.055</td>
      <td>-0.118</td>
      <td>0.038</td>
      <td>0.019</td>
      <td>...</td>
      <td>-0.060</td>
      <td>-0.012</td>
      <td>28.33</td>
      <td>138.8</td>
      <td>292.6</td>
      <td>0.10256</td>
      <td>-0.02410</td>
      <td>-0.08333</td>
      <td>0.07975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1982-08-01</td>
      <td>0.379</td>
      <td>0.077</td>
      <td>0.133</td>
      <td>0.070</td>
      <td>0.216</td>
      <td>0.273</td>
      <td>0.055</td>
      <td>0.032</td>
      <td>0.130</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.221</td>
      <td>28.18</td>
      <td>138.4</td>
      <td>292.8</td>
      <td>0.09302</td>
      <td>0.22222</td>
      <td>0.17273</td>
      <td>0.07607</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1982-09-01</td>
      <td>-0.109</td>
      <td>0.059</td>
      <td>0.039</td>
      <td>0.058</td>
      <td>-0.165</td>
      <td>-0.061</td>
      <td>-0.139</td>
      <td>0.000</td>
      <td>0.209</td>
      <td>...</td>
      <td>0.027</td>
      <td>-0.029</td>
      <td>27.99</td>
      <td>137.3</td>
      <td>293.3</td>
      <td>-0.02128</td>
      <td>-0.06566</td>
      <td>-0.01075</td>
      <td>0.19015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81</th>
      <td>1982-10-01</td>
      <td>0.314</td>
      <td>0.318</td>
      <td>-0.050</td>
      <td>0.268</td>
      <td>0.528</td>
      <td>0.133</td>
      <td>0.171</td>
      <td>0.160</td>
      <td>-0.009</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.150</td>
      <td>28.74</td>
      <td>135.8</td>
      <td>294.1</td>
      <td>0.10870</td>
      <td>0.13297</td>
      <td>0.11594</td>
      <td>-0.03079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>82</th>
      <td>1982-11-01</td>
      <td>0.145</td>
      <td>0.007</td>
      <td>-0.011</td>
      <td>-0.106</td>
      <td>0.003</td>
      <td>0.175</td>
      <td>0.289</td>
      <td>-0.025</td>
      <td>-0.072</td>
      <td>...</td>
      <td>0.012</td>
      <td>0.141</td>
      <td>28.70</td>
      <td>134.8</td>
      <td>293.6</td>
      <td>-0.01961</td>
      <td>0.01942</td>
      <td>-0.00714</td>
      <td>-0.01897</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>83</th>
      <td>1982-12-01</td>
      <td>-0.001</td>
      <td>-0.098</td>
      <td>0.123</td>
      <td>0.037</td>
      <td>0.053</td>
      <td>-0.052</td>
      <td>0.093</td>
      <td>-0.020</td>
      <td>0.015</td>
      <td>...</td>
      <td>0.029</td>
      <td>-0.040</td>
      <td>28.12</td>
      <td>134.7</td>
      <td>292.4</td>
      <td>0.08000</td>
      <td>0.00286</td>
      <td>-0.04651</td>
      <td>0.07367</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>1983-01-01</td>
      <td>-0.045</td>
      <td>0.085</td>
      <td>-0.012</td>
      <td>0.049</td>
      <td>0.208</td>
      <td>0.225</td>
      <td>0.040</td>
      <td>-0.039</td>
      <td>0.015</td>
      <td>...</td>
      <td>0.036</td>
      <td>0.023</td>
      <td>27.22</td>
      <td>137.4</td>
      <td>293.1</td>
      <td>0.18519</td>
      <td>0.11111</td>
      <td>0.11498</td>
      <td>0.07919</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1983-02-01</td>
      <td>0.037</td>
      <td>0.039</td>
      <td>0.060</td>
      <td>-0.035</td>
      <td>0.237</td>
      <td>-0.010</td>
      <td>0.027</td>
      <td>0.067</td>
      <td>0.024</td>
      <td>...</td>
      <td>0.008</td>
      <td>0.065</td>
      <td>26.41</td>
      <td>138.1</td>
      <td>293.2</td>
      <td>-0.03125</td>
      <td>0.07826</td>
      <td>0.01500</td>
      <td>0.02199</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1983-03-01</td>
      <td>0.113</td>
      <td>0.132</td>
      <td>0.048</td>
      <td>0.097</td>
      <td>0.040</td>
      <td>0.034</td>
      <td>-0.016</td>
      <td>0.061</td>
      <td>0.084</td>
      <td>...</td>
      <td>0.039</td>
      <td>-0.023</td>
      <td>26.08</td>
      <td>140.0</td>
      <td>293.4</td>
      <td>0.00000</td>
      <td>-0.09839</td>
      <td>0.04062</td>
      <td>-0.14419</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>87</th>
      <td>1983-04-01</td>
      <td>0.082</td>
      <td>0.104</td>
      <td>0.045</td>
      <td>0.073</td>
      <td>0.079</td>
      <td>-0.060</td>
      <td>-0.043</td>
      <td>0.066</td>
      <td>0.119</td>
      <td>...</td>
      <td>0.098</td>
      <td>0.091</td>
      <td>25.85</td>
      <td>142.6</td>
      <td>295.5</td>
      <td>0.16129</td>
      <td>0.20455</td>
      <td>0.12613</td>
      <td>0.02952</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1983-05-01</td>
      <td>-0.014</td>
      <td>-0.102</td>
      <td>-0.012</td>
      <td>0.000</td>
      <td>-0.114</td>
      <td>-0.052</td>
      <td>-0.045</td>
      <td>0.023</td>
      <td>0.016</td>
      <td>...</td>
      <td>-0.038</td>
      <td>-0.067</td>
      <td>26.08</td>
      <td>144.4</td>
      <td>297.1</td>
      <td>0.02778</td>
      <td>0.00000</td>
      <td>0.04747</td>
      <td>0.01291</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1983-06-01</td>
      <td>-0.130</td>
      <td>-0.016</td>
      <td>0.000</td>
      <td>-0.068</td>
      <td>-0.042</td>
      <td>0.075</td>
      <td>0.012</td>
      <td>-0.026</td>
      <td>0.114</td>
      <td>...</td>
      <td>0.018</td>
      <td>-0.013</td>
      <td>25.98</td>
      <td>146.4</td>
      <td>298.1</td>
      <td>-0.01351</td>
      <td>0.01736</td>
      <td>-0.01546</td>
      <td>-0.05673</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1983-07-01</td>
      <td>-0.087</td>
      <td>-0.079</td>
      <td>0.017</td>
      <td>0.046</td>
      <td>0.173</td>
      <td>-0.142</td>
      <td>-0.259</td>
      <td>-0.072</td>
      <td>-0.007</td>
      <td>...</td>
      <td>0.036</td>
      <td>-0.071</td>
      <td>25.86</td>
      <td>149.7</td>
      <td>299.3</td>
      <td>-0.04110</td>
      <td>0.01128</td>
      <td>0.00524</td>
      <td>0.02560</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1983-08-01</td>
      <td>0.060</td>
      <td>-0.007</td>
      <td>-0.023</td>
      <td>0.055</td>
      <td>0.053</td>
      <td>0.007</td>
      <td>0.080</td>
      <td>-0.010</td>
      <td>0.062</td>
      <td>...</td>
      <td>0.059</td>
      <td>-0.011</td>
      <td>26.03</td>
      <td>151.8</td>
      <td>300.3</td>
      <td>0.00000</td>
      <td>0.10037</td>
      <td>0.10625</td>
      <td>-0.01686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1983-09-01</td>
      <td>0.102</td>
      <td>0.006</td>
      <td>0.087</td>
      <td>-0.026</td>
      <td>0.090</td>
      <td>-0.005</td>
      <td>0.041</td>
      <td>-0.037</td>
      <td>0.049</td>
      <td>...</td>
      <td>-0.037</td>
      <td>-0.033</td>
      <td>26.08</td>
      <td>153.8</td>
      <td>301.8</td>
      <td>-0.07143</td>
      <td>-0.00135</td>
      <td>-0.01190</td>
      <td>-0.01158</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1983-10-01</td>
      <td>-0.052</td>
      <td>-0.118</td>
      <td>0.101</td>
      <td>-0.088</td>
      <td>-0.069</td>
      <td>-0.364</td>
      <td>0.039</td>
      <td>0.116</td>
      <td>0.000</td>
      <td>...</td>
      <td>-0.014</td>
      <td>-0.046</td>
      <td>26.04</td>
      <td>155.0</td>
      <td>302.6</td>
      <td>-0.07692</td>
      <td>-0.05479</td>
      <td>-0.00723</td>
      <td>-0.04246</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1983-11-01</td>
      <td>0.147</td>
      <td>0.162</td>
      <td>-0.025</td>
      <td>0.096</td>
      <td>-0.014</td>
      <td>0.065</td>
      <td>0.120</td>
      <td>-0.014</td>
      <td>0.077</td>
      <td>...</td>
      <td>0.011</td>
      <td>0.151</td>
      <td>26.09</td>
      <td>155.3</td>
      <td>303.1</td>
      <td>0.06667</td>
      <td>-0.05072</td>
      <td>0.06214</td>
      <td>-0.02957</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1983-12-01</td>
      <td>-0.012</td>
      <td>0.023</td>
      <td>0.005</td>
      <td>-0.016</td>
      <td>0.068</td>
      <td>0.034</td>
      <td>-0.028</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>...</td>
      <td>0.021</td>
      <td>-0.069</td>
      <td>25.88</td>
      <td>156.2</td>
      <td>303.5</td>
      <td>-0.03125</td>
      <td>0.03282</td>
      <td>-0.03704</td>
      <td>0.01488</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1984-01-01</td>
      <td>-0.054</td>
      <td>0.024</td>
      <td>0.005</td>
      <td>-0.034</td>
      <td>0.117</td>
      <td>0.208</td>
      <td>-0.013</td>
      <td>-0.009</td>
      <td>0.065</td>
      <td>...</td>
      <td>0.108</td>
      <td>-0.039</td>
      <td>25.93</td>
      <td>158.5</td>
      <td>305.4</td>
      <td>-0.01613</td>
      <td>-0.05243</td>
      <td>-0.04327</td>
      <td>-0.04322</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1984-02-01</td>
      <td>-0.088</td>
      <td>-0.039</td>
      <td>-0.069</td>
      <td>-0.101</td>
      <td>0.027</td>
      <td>-0.024</td>
      <td>-0.117</td>
      <td>-0.073</td>
      <td>-0.091</td>
      <td>...</td>
      <td>0.151</td>
      <td>-0.093</td>
      <td>26.06</td>
      <td>160.0</td>
      <td>306.6</td>
      <td>0.03279</td>
      <td>-0.11462</td>
      <td>-0.03367</td>
      <td>0.04059</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1984-03-01</td>
      <td>0.079</td>
      <td>-0.054</td>
      <td>0.055</td>
      <td>-0.033</td>
      <td>0.056</td>
      <td>0.057</td>
      <td>0.065</td>
      <td>-0.018</td>
      <td>-0.003</td>
      <td>...</td>
      <td>-0.122</td>
      <td>0.094</td>
      <td>26.05</td>
      <td>160.8</td>
      <td>307.3</td>
      <td>0.04762</td>
      <td>0.15000</td>
      <td>0.03958</td>
      <td>0.02159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1984-04-01</td>
      <td>0.012</td>
      <td>-0.004</td>
      <td>0.031</td>
      <td>-0.231</td>
      <td>0.089</td>
      <td>0.053</td>
      <td>-0.085</td>
      <td>0.065</td>
      <td>-0.025</td>
      <td>...</td>
      <td>0.022</td>
      <td>-0.088</td>
      <td>25.93</td>
      <td>162.1</td>
      <td>308.8</td>
      <td>-0.01515</td>
      <td>0.01969</td>
      <td>0.02538</td>
      <td>-0.03213</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 28 columns</p>
</div>




```python
static,pvalue=ss.jarque_bera(df3['CONTIL'])
```


```python
static
```




    9.75995360623815




```python
pvalue
```




    0.007597190256651287




```python
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 9.75995360623815 and the P-value is given by 0.007597190256651287



```python
static,pvalue=ss.kstest(df3['CONTIL'], 'norm')
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 0.4009495780206833 and the P-value is given by 5.191736905658404e-05



```python
static,pvalue=statsmodels.stats._adnorm.normal_ad(df3['CONTIL'])
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 0.6431851643635831 and the P-value is given by 0.08479062672413287



```python
static,pvalue=ss.shapiro(df3['CONTIL'])
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 0.9337706565856934 and the P-value is given by 0.05558066442608833



```python
static,pvalue=ss.jarque_bera(df3['BOISE'])
```


```python
static
```




    20.667853044913137




```python
pvalue
```




    3.251118017322252e-05




```python
static,pvalue=ss.kstest(df3['BOISE'], 'norm')
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 0.43217871960473353 and the P-value is given by 8.857278685089354e-06



```python
static,pvalue=statsmodels.stats._adnorm.normal_ad(df3['BOISE'])
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 1.1523255888499122 and the P-value is given by 0.004367941102311427



```python
static,pvalue=ss.shapiro(df3['BOISE'])
print('The test statistic is given by {} and the P-value is given by {}'.format(static, pvalue))
```

    The test statistic is given by 0.8807262182235718 and the P-value is given by 0.002462482312694192



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1976-01-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python

df = pd.read_excel('Data_For_Analysis.xlsx')
df.set_index('Date', inplace=True)

df.columns
# df=df.dropna()
```




    Index(['BOISE', 'CITCRP', 'CONED', 'CONTIL', 'DATGEN', 'DEC', 'DELTA',
           'GENMIL', 'GERBER', 'IBM', 'MARKET', 'MOBIL', 'MOTOR', 'PANAM', 'PSNH',
           'RKFREE', 'TANDY', 'TEXACO', 'WEYER', 'POIL', 'FRBIND', 'CPI', 'GPU',
           'DOW', 'DUPONT', 'GOLD', 'CONOCO'],
          dtype='object')




```python
MainDF=df[['CONTIL','IBM','MARKET','RKFREE','CPI','POIL','FRBIND']]
MainDF=MainDF.dropna()
MainDF.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.129</td>
      <td>-0.029</td>
      <td>-0.045</td>
      <td>0.00487</td>
      <td>187.2</td>
      <td>8.68</td>
      <td>138.8</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>0.037</td>
      <td>-0.043</td>
      <td>0.010</td>
      <td>0.00494</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.003</td>
      <td>-0.063</td>
      <td>0.050</td>
      <td>0.00526</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.180</td>
      <td>0.130</td>
      <td>0.063</td>
      <td>0.00491</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.061</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>0.00513</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.059</td>
      <td>-0.004</td>
      <td>0.007</td>
      <td>0.00527</td>
      <td>195.3</td>
      <td>9.05</td>
      <td>144.9</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.066</td>
      <td>0.092</td>
      <td>0.071</td>
      <td>0.00528</td>
      <td>196.7</td>
      <td>8.96</td>
      <td>146.1</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.033</td>
      <td>0.049</td>
      <td>0.079</td>
      <td>0.00607</td>
      <td>197.8</td>
      <td>8.05</td>
      <td>147.1</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.013</td>
      <td>-0.051</td>
      <td>0.002</td>
      <td>0.00645</td>
      <td>199.3</td>
      <td>9.15</td>
      <td>147.8</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>-0.123</td>
      <td>-0.046</td>
      <td>-0.189</td>
      <td>0.00685</td>
      <td>200.9</td>
      <td>9.17</td>
      <td>148.6</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.038</td>
      <td>0.031</td>
      <td>0.084</td>
      <td>0.00719</td>
      <td>202.0</td>
      <td>9.20</td>
      <td>149.5</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.047</td>
      <td>0.108</td>
      <td>0.015</td>
      <td>0.00690</td>
      <td>203.3</td>
      <td>9.47</td>
      <td>150.4</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>-0.024</td>
      <td>0.034</td>
      <td>0.058</td>
      <td>0.00761</td>
      <td>204.7</td>
      <td>9.46</td>
      <td>152.0</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.020</td>
      <td>-0.017</td>
      <td>0.011</td>
      <td>0.00761</td>
      <td>207.1</td>
      <td>9.69</td>
      <td>152.5</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>0.043</td>
      <td>0.052</td>
      <td>0.123</td>
      <td>0.00769</td>
      <td>209.1</td>
      <td>9.83</td>
      <td>153.5</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>0.064</td>
      <td>-0.004</td>
      <td>0.026</td>
      <td>0.00764</td>
      <td>211.5</td>
      <td>10.33</td>
      <td>151.1</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>0.005</td>
      <td>-0.022</td>
      <td>0.014</td>
      <td>0.00772</td>
      <td>214.1</td>
      <td>10.71</td>
      <td>152.7</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.092</td>
      <td>-0.035</td>
      <td>0.075</td>
      <td>0.00715</td>
      <td>216.6</td>
      <td>11.70</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>-0.034</td>
      <td>-0.049</td>
      <td>-0.013</td>
      <td>0.00728</td>
      <td>218.9</td>
      <td>13.39</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>0.058</td>
      <td>0.016</td>
      <td>0.095</td>
      <td>0.00789</td>
      <td>221.1</td>
      <td>14.00</td>
      <td>152.1</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.033</td>
      <td>-0.032</td>
      <td>0.039</td>
      <td>0.00802</td>
      <td>223.4</td>
      <td>14.57</td>
      <td>152.7</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.136</td>
      <td>-0.079</td>
      <td>-0.097</td>
      <td>0.00913</td>
      <td>225.4</td>
      <td>15.11</td>
      <td>152.7</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>0.081</td>
      <td>0.060</td>
      <td>0.116</td>
      <td>0.00819</td>
      <td>227.5</td>
      <td>15.52</td>
      <td>152.3</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.104</td>
      <td>-0.013</td>
      <td>0.086</td>
      <td>0.00747</td>
      <td>229.9</td>
      <td>17.03</td>
      <td>152.5</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>-0.103</td>
      <td>0.066</td>
      <td>0.124</td>
      <td>0.00883</td>
      <td>233.2</td>
      <td>17.86</td>
      <td>152.7</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.085</td>
      <td>-0.122</td>
      <td>-0.243</td>
      <td>0.01181</td>
      <td>239.8</td>
      <td>19.34</td>
      <td>152.1</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.074</td>
      <td>-0.016</td>
      <td>0.080</td>
      <td>0.00753</td>
      <td>242.5</td>
      <td>20.29</td>
      <td>148.3</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.023</td>
      <td>0.025</td>
      <td>0.062</td>
      <td>0.00630</td>
      <td>244.9</td>
      <td>21.01</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.064</td>
      <td>0.061</td>
      <td>0.086</td>
      <td>0.00503</td>
      <td>247.6</td>
      <td>21.53</td>
      <td>141.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1983-11-01</th>
      <td>0.096</td>
      <td>-0.066</td>
      <td>0.066</td>
      <td>0.00683</td>
      <td>303.1</td>
      <td>26.09</td>
      <td>155.3</td>
    </tr>
    <tr>
      <th>1983-12-01</th>
      <td>-0.016</td>
      <td>0.039</td>
      <td>-0.012</td>
      <td>0.00693</td>
      <td>303.5</td>
      <td>25.88</td>
      <td>156.2</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>-0.034</td>
      <td>-0.065</td>
      <td>-0.029</td>
      <td>0.00712</td>
      <td>305.4</td>
      <td>25.93</td>
      <td>158.5</td>
    </tr>
    <tr>
      <th>1984-02-01</th>
      <td>-0.101</td>
      <td>-0.026</td>
      <td>-0.030</td>
      <td>0.00672</td>
      <td>306.6</td>
      <td>26.06</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>1984-03-01</th>
      <td>-0.033</td>
      <td>0.034</td>
      <td>0.003</td>
      <td>0.00763</td>
      <td>307.3</td>
      <td>26.05</td>
      <td>160.8</td>
    </tr>
    <tr>
      <th>1984-04-01</th>
      <td>-0.231</td>
      <td>-0.002</td>
      <td>-0.003</td>
      <td>0.00741</td>
      <td>308.8</td>
      <td>25.93</td>
      <td>162.1</td>
    </tr>
    <tr>
      <th>1984-05-01</th>
      <td>-0.600</td>
      <td>-0.044</td>
      <td>-0.058</td>
      <td>0.00627</td>
      <td>309.7</td>
      <td>26.00</td>
      <td>162.8</td>
    </tr>
    <tr>
      <th>1984-06-01</th>
      <td>0.000</td>
      <td>-0.019</td>
      <td>0.005</td>
      <td>0.00748</td>
      <td>310.7</td>
      <td>26.09</td>
      <td>164.4</td>
    </tr>
    <tr>
      <th>1984-07-01</th>
      <td>-0.205</td>
      <td>0.047</td>
      <td>-0.058</td>
      <td>0.00771</td>
      <td>311.7</td>
      <td>26.11</td>
      <td>165.9</td>
    </tr>
    <tr>
      <th>1984-08-01</th>
      <td>0.086</td>
      <td>0.127</td>
      <td>0.146</td>
      <td>0.00852</td>
      <td>313.0</td>
      <td>26.02</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>1984-09-01</th>
      <td>0.974</td>
      <td>0.004</td>
      <td>0.000</td>
      <td>0.00830</td>
      <td>314.5</td>
      <td>25.97</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>1984-10-01</th>
      <td>-0.232</td>
      <td>0.012</td>
      <td>-0.035</td>
      <td>0.00688</td>
      <td>315.3</td>
      <td>25.92</td>
      <td>164.5</td>
    </tr>
    <tr>
      <th>1984-11-01</th>
      <td>-0.023</td>
      <td>-0.023</td>
      <td>-0.019</td>
      <td>0.00602</td>
      <td>315.3</td>
      <td>25.44</td>
      <td>165.2</td>
    </tr>
    <tr>
      <th>1984-12-01</th>
      <td>0.095</td>
      <td>0.011</td>
      <td>-0.001</td>
      <td>0.00612</td>
      <td>315.5</td>
      <td>25.05</td>
      <td>166.2</td>
    </tr>
    <tr>
      <th>1985-01-01</th>
      <td>0.587</td>
      <td>0.108</td>
      <td>0.097</td>
      <td>0.00606</td>
      <td>316.1</td>
      <td>24.28</td>
      <td>165.6</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>-0.096</td>
      <td>-0.009</td>
      <td>0.012</td>
      <td>0.00586</td>
      <td>317.4</td>
      <td>23.63</td>
      <td>165.7</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>0.030</td>
      <td>-0.052</td>
      <td>0.008</td>
      <td>0.00650</td>
      <td>318.8</td>
      <td>23.88</td>
      <td>166.1</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>-0.029</td>
      <td>-0.004</td>
      <td>-0.010</td>
      <td>0.00601</td>
      <td>320.1</td>
      <td>24.15</td>
      <td>166.2</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>-0.091</td>
      <td>0.025</td>
      <td>0.019</td>
      <td>0.00512</td>
      <td>321.3</td>
      <td>24.18</td>
      <td>166.2</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>-0.050</td>
      <td>-0.038</td>
      <td>-0.003</td>
      <td>0.00536</td>
      <td>322.3</td>
      <td>24.03</td>
      <td>166.5</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>0.018</td>
      <td>0.062</td>
      <td>0.012</td>
      <td>0.00562</td>
      <td>322.8</td>
      <td>24.00</td>
      <td>166.2</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>-0.052</td>
      <td>-0.028</td>
      <td>0.005</td>
      <td>0.00545</td>
      <td>323.5</td>
      <td>23.92</td>
      <td>167.7</td>
    </tr>
    <tr>
      <th>1985-09-01</th>
      <td>0.036</td>
      <td>-0.022</td>
      <td>-0.055</td>
      <td>0.00571</td>
      <td>324.5</td>
      <td>23.93</td>
      <td>167.6</td>
    </tr>
    <tr>
      <th>1985-10-01</th>
      <td>0.105</td>
      <td>0.048</td>
      <td>0.026</td>
      <td>0.00577</td>
      <td>325.5</td>
      <td>24.06</td>
      <td>166.6</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>0.048</td>
      <td>0.085</td>
      <td>0.059</td>
      <td>0.00540</td>
      <td>326.6</td>
      <td>24.31</td>
      <td>167.6</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>0.197</td>
      <td>0.113</td>
      <td>0.013</td>
      <td>0.00479</td>
      <td>327.4</td>
      <td>24.53</td>
      <td>168.8</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>0.000</td>
      <td>-0.026</td>
      <td>-0.009</td>
      <td>0.00548</td>
      <td>328.4</td>
      <td>23.12</td>
      <td>169.6</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>-0.051</td>
      <td>0.003</td>
      <td>0.049</td>
      <td>0.00523</td>
      <td>327.5</td>
      <td>17.65</td>
      <td>168.4</td>
    </tr>
    <tr>
      <th>1986-03-01</th>
      <td>-0.040</td>
      <td>0.004</td>
      <td>0.048</td>
      <td>0.00508</td>
      <td>326.0</td>
      <td>12.62</td>
      <td>166.1</td>
    </tr>
    <tr>
      <th>1986-04-01</th>
      <td>-0.097</td>
      <td>0.031</td>
      <td>-0.009</td>
      <td>0.00444</td>
      <td>325.3</td>
      <td>10.68</td>
      <td>167.6</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 7 columns</p>
</div>




```python
MainDF.index
```




    DatetimeIndex(['1978-01-01', '1978-02-01', '1978-03-01', '1978-04-01',
                   '1978-05-01', '1978-06-01', '1978-07-01', '1978-08-01',
                   '1978-09-01', '1978-10-01',
                   ...
                   '1987-02-01', '1987-03-01', '1987-04-01', '1987-05-01',
                   '1987-06-01', '1987-07-01', '1987-08-01', '1987-09-01',
                   '1987-10-01', '1987-11-01'],
                  dtype='datetime64[ns]', name='Date', length=119, freq=None)




```python
MainDF['RINF']=MainDF['CPI'].pct_change(1)
MainDF['GIND']=MainDF['FRBIND'].pct_change(1)
MainDF['real_POIL']=MainDF['POIL']/MainDF['CPI']
MainDF['ROIL']=MainDF['real_POIL'].pct_change(1)

MainDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOBIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-01-01</th>
      <td>-0.046</td>
      <td>-0.029</td>
      <td>-0.045</td>
      <td>0.00487</td>
      <td>187.2</td>
      <td>8.68</td>
      <td>138.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.046368</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1978-02-01</th>
      <td>-0.017</td>
      <td>-0.043</td>
      <td>0.010</td>
      <td>0.00494</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>0.006410</td>
      <td>0.002882</td>
      <td>0.046921</td>
      <td>0.011946</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.049</td>
      <td>-0.063</td>
      <td>0.050</td>
      <td>0.00526</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>0.007431</td>
      <td>0.012213</td>
      <td>0.046365</td>
      <td>-0.011868</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.077</td>
      <td>0.130</td>
      <td>0.063</td>
      <td>0.00491</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>0.008957</td>
      <td>0.016324</td>
      <td>0.046057</td>
      <td>-0.006625</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>-0.011</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>0.00513</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>0.009399</td>
      <td>0.004888</td>
      <td>0.045577</td>
      <td>-0.010435</td>
    </tr>
  </tbody>
</table>
</div>




```python
MainDF=MainDF.dropna()
MainDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOBIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-02-01</th>
      <td>-0.017</td>
      <td>-0.043</td>
      <td>0.010</td>
      <td>0.00494</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>0.006410</td>
      <td>0.002882</td>
      <td>0.046921</td>
      <td>0.011946</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.049</td>
      <td>-0.063</td>
      <td>0.050</td>
      <td>0.00526</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>0.007431</td>
      <td>0.012213</td>
      <td>0.046365</td>
      <td>-0.011868</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.077</td>
      <td>0.130</td>
      <td>0.063</td>
      <td>0.00491</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>0.008957</td>
      <td>0.016324</td>
      <td>0.046057</td>
      <td>-0.006625</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>-0.011</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>0.00513</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>0.009399</td>
      <td>0.004888</td>
      <td>0.045577</td>
      <td>-0.010435</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.043</td>
      <td>-0.004</td>
      <td>0.007</td>
      <td>0.00527</td>
      <td>195.3</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>0.010347</td>
      <td>0.006949</td>
      <td>0.046339</td>
      <td>0.016722</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.rcParams["figure.figsize"] = [10,5]
MainDF[['MARKET', 'CONTIL']].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c20d7e320>




![png](output_145_1.png)



```python
    MainDF[MainDF['CONTIL']== MainDF['CONTIL'].max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984-09-01</th>
      <td>0.974</td>
      <td>0.004</td>
      <td>0.0</td>
      <td>0.0083</td>
      <td>314.5</td>
      <td>25.97</td>
      <td>165.0</td>
      <td>0.004792</td>
      <td>-0.006024</td>
      <td>0.082576</td>
      <td>-0.006682</td>
    </tr>
  </tbody>
</table>
</div>




```python
start_date = dt.datetime(1978,1,1)
end_date = dt.datetime(1980,1,1)

start_date2 = dt.datetime(1980,1,1)
end_date2 = dt.datetime(1985,1,1)

select = (MainDF.index>=start_date)*(MainDF.index<end_date)

select2 = (MainDF.index>=start_date2)*(MainDF.index<end_date2)

MainDF_first_period=MainDF[select]
MainDF_second_period=MainDF[select2]

```


```python
MainDF_first_period.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-02-01</th>
      <td>0.037</td>
      <td>-0.043</td>
      <td>0.010</td>
      <td>0.00494</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>0.006410</td>
      <td>0.002882</td>
      <td>0.046921</td>
      <td>0.011946</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.003</td>
      <td>-0.063</td>
      <td>0.050</td>
      <td>0.00526</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>0.007431</td>
      <td>0.012213</td>
      <td>0.046365</td>
      <td>-0.011868</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.180</td>
      <td>0.130</td>
      <td>0.063</td>
      <td>0.00491</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>0.008957</td>
      <td>0.016324</td>
      <td>0.046057</td>
      <td>-0.006625</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.061</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>0.00513</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>0.009399</td>
      <td>0.004888</td>
      <td>0.045577</td>
      <td>-0.010435</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.059</td>
      <td>-0.004</td>
      <td>0.007</td>
      <td>0.00527</td>
      <td>195.3</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>0.010347</td>
      <td>0.006949</td>
      <td>0.046339</td>
      <td>0.016722</td>
    </tr>
  </tbody>
</table>
</div>




```python
MainDF_first_period.tail(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-11-01</th>
      <td>0.081</td>
      <td>0.060</td>
      <td>0.116</td>
      <td>0.00819</td>
      <td>227.5</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>0.009317</td>
      <td>-0.002620</td>
      <td>0.068220</td>
      <td>0.017653</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.104</td>
      <td>-0.013</td>
      <td>0.086</td>
      <td>0.00747</td>
      <td>229.9</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>0.010549</td>
      <td>0.001313</td>
      <td>0.074076</td>
      <td>0.085839</td>
    </tr>
  </tbody>
</table>
</div>




```python
MainDF_second_period.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOBIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-01</th>
      <td>0.075</td>
      <td>0.066</td>
      <td>0.124</td>
      <td>0.00883</td>
      <td>233.2</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>0.014354</td>
      <td>0.001311</td>
      <td>0.076587</td>
      <td>0.033897</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>0.366</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
  </tbody>
</table>
</div>




```python
MainDF_second_period.tail(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOBIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984-11-01</th>
      <td>-0.042</td>
      <td>-0.023</td>
      <td>-0.019</td>
      <td>0.00602</td>
      <td>315.3</td>
      <td>25.44</td>
      <td>165.2</td>
      <td>0.000000</td>
      <td>0.004255</td>
      <td>0.080685</td>
      <td>-0.018519</td>
    </tr>
    <tr>
      <th>1984-12-01</th>
      <td>-0.052</td>
      <td>0.011</td>
      <td>-0.001</td>
      <td>0.00612</td>
      <td>315.5</td>
      <td>25.05</td>
      <td>166.2</td>
      <td>0.000634</td>
      <td>0.006053</td>
      <td>0.079398</td>
      <td>-0.015954</td>
    </tr>
  </tbody>
</table>
</div>




```python
formula = 'CONTIL ~ MARKET'
results_Mobil_period1 = smf.ols(formula, MainDF_first_period).fit()
print(results_Mobil_period1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 CONTIL   R-squared:                       0.512
    Model:                            OLS   Adj. R-squared:                  0.489
    Method:                 Least Squares   F-statistic:                     22.03
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):           0.000124
    Time:                        16:20:38   Log-Likelihood:                 36.492
    No. Observations:                  23   AIC:                            -68.98
    Df Residuals:                      21   BIC:                            -66.71
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0088      0.012     -0.726      0.476      -0.034       0.016
    MARKET         0.7533      0.161      4.693      0.000       0.420       1.087
    ==============================================================================
    Omnibus:                        4.224   Durbin-Watson:                   2.335
    Prob(Omnibus):                  0.121   Jarque-Bera (JB):                2.320
    Skew:                           0.636   Prob(JB):                        0.313
    Kurtosis:                       3.895   Cond. No.                         14.9
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
formula = 'CONTIL ~ MARKET + RINF + GIND + ROIL'
results_Mobil_Secondmodel_period1 = smf.ols(formula, MainDF_first_period).fit()
print(results_Mobil_Secondmodel_period1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 CONTIL   R-squared:                       0.526
    Model:                            OLS   Adj. R-squared:                  0.420
    Method:                 Least Squares   F-statistic:                     4.988
    Date:                Wed, 18 Dec 2019   Prob (F-statistic):            0.00696
    Time:                        16:20:49   Log-Likelihood:                 36.821
    No. Observations:                  23   AIC:                            -63.64
    Df Residuals:                      18   BIC:                            -57.97
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0475      0.062     -0.769      0.452      -0.177       0.082
    MARKET         0.7556      0.173      4.374      0.000       0.393       1.119
    RINF           3.9987      6.696      0.597      0.558     -10.069      18.066
    GIND           0.4681      1.999      0.234      0.817      -3.732       4.668
    ROIL           0.0382      0.276      0.138      0.891      -0.542       0.618
    ==============================================================================
    Omnibus:                        3.598   Durbin-Watson:                   2.368
    Prob(Omnibus):                  0.165   Jarque-Bera (JB):                1.973
    Skew:                           0.664   Prob(JB):                        0.373
    Kurtosis:                       3.543   Cond. No.                         584.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
hypotheses = 'RINF=0, GIND=0, ROIL=0'
f_test=results_Mobil_Secondmodel_period1.f_test(hypotheses)
print(f_test)
```

    <F test: F=array([[0.17434189]]), p=0.9123562285533326, df_denom=18, df_num=3>



```python
resi_Model_1=results_Mobil_period1.resid
resi_Model_2=results_Mobil_Secondmodel_period1.resid
```


```python
resi_Model_1.std()
```




    0.050624668794528864




```python
resi_Model_2.std()
```




    0.04990481881444332




```python
wald_0 = results_Mobil_Secondmodel_period1.wald_test(hypotheses)
print('H0:', hypotheses)
print(wald_0)
```

    H0: RINF=0, GIND=0, ROIL=0
    <F test: F=array([[0.17434189]]), p=0.9123562285533326, df_denom=18, df_num=3>



```python
residuals=pd.DataFrame(resi_Model_1)
residuals.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c217008d0>




![png](output_159_1.png)



```python

```


```python
Plot_resi_corr_time(results_Mobil_Secondmodel_period1,MainDF_first_period)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Residuals</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-02-01</th>
      <td>0.049545</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>-0.022220</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.136730</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.018437</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.062015</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.027997</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>-0.000652</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.004416</td>
    </tr>
    <tr>
      <th>1978-10-01</th>
      <td>0.032938</td>
    </tr>
    <tr>
      <th>1978-11-01</th>
      <td>-0.078579</td>
    </tr>
    <tr>
      <th>1978-12-01</th>
      <td>0.053783</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>-0.052503</td>
    </tr>
    <tr>
      <th>1979-02-01</th>
      <td>-0.029670</td>
    </tr>
    <tr>
      <th>1979-03-01</th>
      <td>-0.044270</td>
    </tr>
    <tr>
      <th>1979-04-01</th>
      <td>0.051829</td>
    </tr>
    <tr>
      <th>1979-05-01</th>
      <td>-0.013077</td>
    </tr>
    <tr>
      <th>1979-06-01</th>
      <td>0.032207</td>
    </tr>
    <tr>
      <th>1979-07-01</th>
      <td>-0.024156</td>
    </tr>
    <tr>
      <th>1979-08-01</th>
      <td>-0.005022</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.059519</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.052028</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>0.004184</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.040478</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_161_1.png)



```python
print(resi_Model_1.describe())
```

    count    2.300000e+01
    mean     3.921984e-18
    std      5.062467e-02
    min     -9.246818e-02
    25%     -3.335145e-02
    50%     -4.754541e-03
    75%      3.323031e-02
    max      1.413512e-01
    dtype: float64



```python
Figure=resi_Model_1.plot.hist(grid=False, bins=20, rwidth=0.9,
                                    color='#607c8e')
plt.title('Histogram of Residuals of Model One')
plt.xlabel('Residuals of Model One')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
```


![png](output_163_0.png)



```python
Adj_df= Create_lags_of_variable(MainDF_first_period, lags=[1,2], column='CONTIL')
Adj_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL for time t-2</th>
      <th>CONTIL for time t-1</th>
      <th>CONTIL for time t-0</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-04-01</th>
      <td>0.037</td>
      <td>0.003</td>
      <td>0.180</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.003</td>
      <td>0.180</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.180</td>
      <td>0.061</td>
      <td>-0.059</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.061</td>
      <td>-0.059</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>-0.059</td>
      <td>0.066</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>




```python
Adj_df1=Create_lags_of_variable(MainDF_first_period, lags=3, column='CONTIL')
Adj_df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL at time t-3</th>
      <th>CONTIL at time t</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-05-01</th>
      <td>0.037</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.003</td>
      <td>-0.059</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.180</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.061</td>
      <td>0.033</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>-0.013</td>
    </tr>
  </tbody>
</table>
</div>




```python
Adj_df= Create_lags_of_variable(MainDF_first_period, lags=[1,2,3], column='CONTIL')
Adj_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL for time t-3</th>
      <th>CONTIL for time t-2</th>
      <th>CONTIL for time t-1</th>
      <th>CONTIL for time t-0</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-05-01</th>
      <td>0.037</td>
      <td>0.003</td>
      <td>0.180</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>0.003</td>
      <td>0.180</td>
      <td>0.061</td>
      <td>-0.059</td>
    </tr>
    <tr>
      <th>1978-07-01</th>
      <td>0.180</td>
      <td>0.061</td>
      <td>-0.059</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>1978-08-01</th>
      <td>0.061</td>
      <td>-0.059</td>
      <td>0.066</td>
      <td>0.033</td>
    </tr>
    <tr>
      <th>1978-09-01</th>
      <td>-0.059</td>
      <td>0.066</td>
      <td>0.033</td>
      <td>-0.013</td>
    </tr>
  </tbody>
</table>
</div>




```python
Adj_df.cov()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL for time t-3</th>
      <th>CONTIL for time t-2</th>
      <th>CONTIL for time t-1</th>
      <th>CONTIL for time t-0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CONTIL for time t-3</th>
      <td>0.004234</td>
      <td>0.000363</td>
      <td>-0.000880</td>
      <td>0.000672</td>
    </tr>
    <tr>
      <th>CONTIL for time t-2</th>
      <td>0.000363</td>
      <td>0.005371</td>
      <td>-0.000179</td>
      <td>-0.001827</td>
    </tr>
    <tr>
      <th>CONTIL for time t-1</th>
      <td>-0.000880</td>
      <td>-0.000179</td>
      <td>0.005629</td>
      <td>0.000234</td>
    </tr>
    <tr>
      <th>CONTIL for time t-0</th>
      <td>0.000672</td>
      <td>-0.001827</td>
      <td>0.000234</td>
      <td>0.004578</td>
    </tr>
  </tbody>
</table>
</div>




```python
Adj_df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL for time t-3</th>
      <th>CONTIL for time t-2</th>
      <th>CONTIL for time t-1</th>
      <th>CONTIL for time t-0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CONTIL for time t-3</th>
      <td>1.000000</td>
      <td>0.076213</td>
      <td>-0.180310</td>
      <td>0.152744</td>
    </tr>
    <tr>
      <th>CONTIL for time t-2</th>
      <td>0.076213</td>
      <td>1.000000</td>
      <td>-0.032594</td>
      <td>-0.368462</td>
    </tr>
    <tr>
      <th>CONTIL for time t-1</th>
      <td>-0.180310</td>
      <td>-0.032594</td>
      <td>1.000000</td>
      <td>0.046181</td>
    </tr>
    <tr>
      <th>CONTIL for time t-0</th>
      <td>0.152744</td>
      <td>-0.368462</td>
      <td>0.046181</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
smt.stattools.acovf(MainDF_first_period['CONTIL'])[:5]
```




    array([ 5.02254820e-03,  1.15328183e-04, -1.30599770e-03,  5.54623490e-04,
            7.23278540e-05])




```python
smt.stattools.acf(MainDF_first_period['CONTIL'])[:5]
```




    array([ 1.        ,  0.02296209, -0.26002691,  0.11042671,  0.01440063])




```python
LujungStatitic, Pvalue=sms.diagnostic.acorr_ljungbox(MainDF_first_period['CONTIL'], lags=15)
```


```python
LujungStatitic
```




    array([ 0.01378059,  1.86511616,  2.21569536,  2.22197128,  2.51331336,
            5.79369477,  6.16667323,  6.67289896,  6.69039293,  7.83214627,
            9.62068974, 11.26534231, 11.30084601, 11.88387249, 12.80880527])




```python
Pvalue
```




    array([0.90655041, 0.3935457 , 0.5288635 , 0.69500887, 0.77448873,
           0.44669298, 0.52042835, 0.57229989, 0.66931804, 0.6452286 ,
           0.56479404, 0.50632614, 0.58563063, 0.61562829, 0.61706293])




```python
plt.rcParams["figure.figsize"] = [20,10]
pyplot.figure()
pyplot.subplot(211)
smgtsplot.plot_acf(MainDF_first_period['CONTIL'], lags=12,  ax=pyplot.gca())
pyplot.subplot(212)

smgtsplot.plot_pacf(MainDF_first_period['CONTIL'], lags=12, ax=pyplot.gca())
pyplot.show()
```


![png](output_174_0.png)



```python
acf,q,pval = smt.acf(MainDF_first_period['CONTIL'],nlags=12,qstat=True)
pacf = smt.pacf(MainDF_first_period['CONTIL'],nlags=12)

correlogram = pd.DataFrame({'acf':acf[1:],
                            'pacf':pacf[1:],
                            'Q':q,
                            'p-val':pval})
correlogram
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acf</th>
      <th>pacf</th>
      <th>Q</th>
      <th>p-val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.022962</td>
      <td>0.024006</td>
      <td>0.013781</td>
      <td>0.906550</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.260027</td>
      <td>-0.285532</td>
      <td>1.865116</td>
      <td>0.393546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.110427</td>
      <td>0.155384</td>
      <td>2.215695</td>
      <td>0.528863</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.014401</td>
      <td>-0.087671</td>
      <td>2.221971</td>
      <td>0.695009</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.095500</td>
      <td>-0.039645</td>
      <td>2.513313</td>
      <td>0.774489</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.311424</td>
      <td>-0.504251</td>
      <td>5.793695</td>
      <td>0.446693</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.101875</td>
      <td>-0.184599</td>
      <td>6.166673</td>
      <td>0.520428</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.114917</td>
      <td>-0.122394</td>
      <td>6.672899</td>
      <td>0.572300</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.020638</td>
      <td>-0.034259</td>
      <td>6.690393</td>
      <td>0.669318</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.160666</td>
      <td>-0.460632</td>
      <td>7.832146</td>
      <td>0.645229</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.193200</td>
      <td>0.373956</td>
      <td>9.620690</td>
      <td>0.564794</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.177378</td>
      <td>-0.195072</td>
      <td>11.265342</td>
      <td>0.506326</td>
    </tr>
  </tbody>
</table>
</div>




```python
x=MainDF_first_period['CONTIL']
a= smt.stattools.arma_order_select_ic(x, max_ar=5, max_ma=3, ic=['aic', 'bic', 'hqic'])
```


```python
a
```




    {'aic':            0          1          2          3
     0 -52.486638 -50.514004 -51.856820 -49.609435
     1 -50.499138 -50.846145 -50.516722 -47.802699
     2 -50.157470 -49.327535 -49.710367 -46.841670
     3 -48.996520 -47.649060 -47.799412 -44.899620
     4 -47.044130 -49.105173        NaN -44.650059
     5 -45.180742        NaN        NaN        NaN,
     'bic':            0          1          2          3
     0 -50.215650 -47.107521 -47.314843 -43.931964
     1 -47.092655 -46.304168 -44.839251 -40.989734
     2 -45.615493 -43.650064 -42.897402 -38.893211
     3 -43.319049 -40.836095 -39.850952 -35.815666
     4 -40.231164 -41.156713        NaN -34.430611
     5 -37.232283        NaN        NaN        NaN,
     'hqic':            0          1          2          3
     0 -51.915491 -49.657283 -50.714525 -48.181567
     1 -49.642417 -49.703851 -49.088854 -46.089257
     2 -49.015176 -47.899667 -47.996926 -44.842655
     3 -47.568652 -45.935619 -45.800396 -42.615031
     4 -45.330688 -47.106157        NaN -42.079896
     5 -43.181727        NaN        NaN        NaN,
     'aic_min_order': (0, 0),
     'bic_min_order': (0, 0),
     'hqic_min_order': (0, 0)}




```python
res=smt.ARIMA(MainDF_first_period['CONTIL'], order=(0,0,1)).fit()
print(res.summary())
```

                                  ARMA Model Results                              
    ==============================================================================
    Dep. Variable:                 CONTIL   No. Observations:                   23
    Model:                     ARMA(0, 1)   Log Likelihood                  28.257
    Method:                       css-mle   S.D. of innovations              0.071
    Date:                Wed, 18 Dec 2019   AIC                            -50.514
    Time:                        16:56:04   BIC                            -47.108
    Sample:                    02-01-1978   HQIC                           -49.657
                             - 12-01-1979                                         
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const            0.0174      0.016      1.116      0.277      -0.013       0.048
    ma.L1.CONTIL     0.0521      0.316      0.165      0.871      -0.568       0.672
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    MA.1          -19.1907           +0.0000j           19.1907            0.5000
    -----------------------------------------------------------------------------



```python
res=smt.ARIMA(MainDF_first_period['CONTIL'], order=(2,0,1)).fit()
print(res.summary())
```

                                  ARMA Model Results                              
    ==============================================================================
    Dep. Variable:                 CONTIL   No. Observations:                   23
    Model:                     ARMA(2, 1)   Log Likelihood                  29.664
    Method:                       css-mle   S.D. of innovations              0.064
    Date:                Wed, 18 Dec 2019   AIC                            -49.328
    Time:                        16:56:24   BIC                            -43.650
    Sample:                    02-01-1978   HQIC                           -47.900
                             - 12-01-1979                                         
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const            0.0176      0.014      1.266      0.221      -0.010       0.045
    ar.L1.CONTIL    -0.7729      0.213     -3.631      0.002      -1.190      -0.356
    ar.L2.CONTIL    -0.1487      0.211     -0.705      0.489      -0.562       0.265
    ma.L1.CONTIL     1.0000        nan        nan        nan         nan         nan
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1           -2.4229           +0.0000j            2.4229            0.5000
    AR.2           -2.7765           +0.0000j            2.7765            0.5000
    MA.1           -1.0000           +0.0000j            1.0000            0.5000
    -----------------------------------------------------------------------------



```python
RF.tsplot(MainDF_first_period['CONTIL'], lags=10)
```


![png](output_180_0.png)



```python
MainDF.index.get_loc('1980-02-01')
```




    24




```python
MainDF.index.get_loc('1983-02-01')
```




    60




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOISE</th>
      <th>CITCRP</th>
      <th>CONED</th>
      <th>CONTIL</th>
      <th>DATGEN</th>
      <th>DEC</th>
      <th>DELTA</th>
      <th>GENMIL</th>
      <th>GERBER</th>
      <th>IBM</th>
      <th>...</th>
      <th>TEXACO</th>
      <th>WEYER</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>CPI</th>
      <th>GPU</th>
      <th>DOW</th>
      <th>DUPONT</th>
      <th>GOLD</th>
      <th>CONOCO</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.90</td>
      <td>125.9</td>
      <td>166.7</td>
      <td>0.05412</td>
      <td>0.18281</td>
      <td>0.24506</td>
      <td>-0.10386</td>
      <td>0.11499</td>
    </tr>
    <tr>
      <th>1976-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>127.6</td>
      <td>167.1</td>
      <td>-0.01429</td>
      <td>0.02307</td>
      <td>-0.02698</td>
      <td>0.05101</td>
      <td>-0.05525</td>
    </tr>
    <tr>
      <th>1976-03-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.79</td>
      <td>128.3</td>
      <td>167.5</td>
      <td>0.01449</td>
      <td>-0.02570</td>
      <td>-0.04105</td>
      <td>0.01071</td>
      <td>0.06876</td>
    </tr>
    <tr>
      <th>1976-04-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.86</td>
      <td>128.7</td>
      <td>168.2</td>
      <td>-0.01886</td>
      <td>-0.00116</td>
      <td>0.03425</td>
      <td>-0.03494</td>
      <td>0.00551</td>
    </tr>
    <tr>
      <th>1976-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.89</td>
      <td>129.7</td>
      <td>169.2</td>
      <td>-0.01493</td>
      <td>-0.08140</td>
      <td>0.00911</td>
      <td>-0.00771</td>
      <td>0.02523</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
train=MainDF[0:25] 
test=MainDF[25:61]
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-02-01</th>
      <td>0.037</td>
      <td>-0.043</td>
      <td>0.010</td>
      <td>0.00494</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>0.006410</td>
      <td>0.002882</td>
      <td>0.046921</td>
      <td>0.011946</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.003</td>
      <td>-0.063</td>
      <td>0.050</td>
      <td>0.00526</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>0.007431</td>
      <td>0.012213</td>
      <td>0.046365</td>
      <td>-0.011868</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.180</td>
      <td>0.130</td>
      <td>0.063</td>
      <td>0.00491</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>0.008957</td>
      <td>0.016324</td>
      <td>0.046057</td>
      <td>-0.006625</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.061</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>0.00513</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>0.009399</td>
      <td>0.004888</td>
      <td>0.045577</td>
      <td>-0.010435</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.059</td>
      <td>-0.004</td>
      <td>0.007</td>
      <td>0.00527</td>
      <td>195.3</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>0.010347</td>
      <td>0.006949</td>
      <td>0.046339</td>
      <td>0.016722</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['CONTIL'].plot(figsize=(7,4), title= 'Monthly Returns of CONTIL Stocks', fontsize=14)
test['CONTIL'].plot(figsize=(7,4), title= 'Monthly Returns of CONTIL| Stocks', fontsize=14)
plt.show()
```


![png](output_186_0.png)



```python

df1 = MainDF.resample('Y').mean()
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-12-31</th>
      <td>0.017636</td>
      <td>0.016818</td>
      <td>0.023545</td>
      <td>0.005841</td>
      <td>196.209091</td>
      <td>8.938182</td>
      <td>145.600000</td>
      <td>0.007530</td>
      <td>0.007330</td>
      <td>0.045563</td>
      <td>0.001733</td>
    </tr>
    <tr>
      <th>1979-12-31</th>
      <td>0.016667</td>
      <td>-0.007417</td>
      <td>0.044417</td>
      <td>0.007783</td>
      <td>217.441667</td>
      <td>12.611667</td>
      <td>152.508333</td>
      <td>0.010300</td>
      <td>0.001180</td>
      <td>0.057661</td>
      <td>0.040099</td>
    </tr>
    <tr>
      <th>1980-12-31</th>
      <td>0.013250</td>
      <td>0.010750</td>
      <td>0.030667</td>
      <td>0.008521</td>
      <td>246.816667</td>
      <td>21.605833</td>
      <td>146.858333</td>
      <td>0.009793</td>
      <td>-0.001261</td>
      <td>0.087353</td>
      <td>0.025363</td>
    </tr>
    <tr>
      <th>1981-12-31</th>
      <td>0.011333</td>
      <td>-0.008917</td>
      <td>-0.003250</td>
      <td>0.010513</td>
      <td>272.350000</td>
      <td>31.850833</td>
      <td>150.891667</td>
      <td>0.007165</td>
      <td>-0.003692</td>
      <td>0.117087</td>
      <td>0.009196</td>
    </tr>
    <tr>
      <th>1982-12-31</th>
      <td>-0.022167</td>
      <td>0.050750</td>
      <td>0.006750</td>
      <td>0.007848</td>
      <td>289.150000</td>
      <td>28.535833</td>
      <td>138.600000</td>
      <td>0.003181</td>
      <td>-0.005170</td>
      <td>0.098732</td>
      <td>-0.010297</td>
    </tr>
    <tr>
      <th>1983-12-31</th>
      <td>0.015250</td>
      <td>0.024250</td>
      <td>0.023500</td>
      <td>0.006703</td>
      <td>298.416667</td>
      <td>26.133333</td>
      <td>147.558333</td>
      <td>0.003112</td>
      <td>0.012435</td>
      <td>0.087595</td>
      <td>-0.009905</td>
    </tr>
    <tr>
      <th>1984-12-31</th>
      <td>-0.025333</td>
      <td>0.004667</td>
      <td>-0.006583</td>
      <td>0.007182</td>
      <td>311.150000</td>
      <td>25.880833</td>
      <td>163.450000</td>
      <td>0.003238</td>
      <td>0.005200</td>
      <td>0.083194</td>
      <td>-0.005912</td>
    </tr>
    <tr>
      <th>1985-12-31</th>
      <td>0.058583</td>
      <td>0.024000</td>
      <td>0.015250</td>
      <td>0.005637</td>
      <td>322.191667</td>
      <td>24.075000</td>
      <td>166.733333</td>
      <td>0.003090</td>
      <td>0.001303</td>
      <td>0.074728</td>
      <td>-0.004734</td>
    </tr>
    <tr>
      <th>1986-12-31</th>
      <td>-0.047250</td>
      <td>-0.018083</td>
      <td>0.005917</td>
      <td>0.004506</td>
      <td>328.383333</td>
      <td>12.450833</td>
      <td>168.108333</td>
      <td>0.000941</td>
      <td>0.000708</td>
      <td>0.037921</td>
      <td>-0.052466</td>
    </tr>
    <tr>
      <th>1987-12-31</th>
      <td>-0.044636</td>
      <td>-0.000455</td>
      <td>-0.000727</td>
      <td>0.004038</td>
      <td>339.900000</td>
      <td>15.522727</td>
      <td>173.963636</td>
      <td>0.003958</td>
      <td>0.004604</td>
      <td>0.045649</td>
      <td>0.022754</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_out_sample= test.copy()
df_in_sample=train.copy()
```


```python

f_Method_NF=RF.Naive_Forecast(df_in_sample, df_out_sample, 'CONTIL')
```


![png](output_189_0.png)



```python
f_Method_NF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
  </tbody>
</table>
</div>




```python
f_Method_AF=RF.Average_Forecast(df_in_sample, df_out_sample, 'CONTIL')
```


![png](output_191_0.png)



```python
f_Method_MAF=RF.Moving_Average_Forecast(df_in_sample, df_out_sample, 'CONTIL', 3)
```


![png](output_192_0.png)



```python
f_Method_MAF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOBIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-03-01</th>
      <td>0.168</td>
      <td>-0.003</td>
      <td>0.107333</td>
      <td>0.00901</td>
      <td>233.166667</td>
      <td>17.9</td>
      <td>152.6</td>
      <td>0.012875</td>
      <td>0.000657</td>
      <td>0.076744</td>
      <td>0.05289</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.168</td>
      <td>-0.003</td>
      <td>0.107333</td>
      <td>0.00901</td>
      <td>233.166667</td>
      <td>17.9</td>
      <td>152.6</td>
      <td>0.012875</td>
      <td>0.000657</td>
      <td>0.076744</td>
      <td>0.05289</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.168</td>
      <td>-0.003</td>
      <td>0.107333</td>
      <td>0.00901</td>
      <td>233.166667</td>
      <td>17.9</td>
      <td>152.6</td>
      <td>0.012875</td>
      <td>0.000657</td>
      <td>0.076744</td>
      <td>0.05289</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.168</td>
      <td>-0.003</td>
      <td>0.107333</td>
      <td>0.00901</td>
      <td>233.166667</td>
      <td>17.9</td>
      <td>152.6</td>
      <td>0.012875</td>
      <td>0.000657</td>
      <td>0.076744</td>
      <td>0.05289</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.168</td>
      <td>-0.003</td>
      <td>0.107333</td>
      <td>0.00901</td>
      <td>233.166667</td>
      <td>17.9</td>
      <td>152.6</td>
      <td>0.012875</td>
      <td>0.000657</td>
      <td>0.076744</td>
      <td>0.05289</td>
    </tr>
  </tbody>
</table>
</div>




```python
f_Method_MAF_1=RF.Moving_Average_Forecast(df_in_sample, df_out_sample, 'MOBIL', 10)
```


![png](output_194_0.png)



```python
f_Method_MAF_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOBIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-03-01</th>
      <td>0.0835</td>
      <td>-0.015</td>
      <td>0.0551</td>
      <td>0.008241</td>
      <td>224.65</td>
      <td>14.87</td>
      <td>152.63</td>
      <td>0.011194</td>
      <td>0.000997</td>
      <td>0.065924</td>
      <td>0.050578</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.0835</td>
      <td>-0.015</td>
      <td>0.0551</td>
      <td>0.008241</td>
      <td>224.65</td>
      <td>14.87</td>
      <td>152.63</td>
      <td>0.011194</td>
      <td>0.000997</td>
      <td>0.065924</td>
      <td>0.050578</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.0835</td>
      <td>-0.015</td>
      <td>0.0551</td>
      <td>0.008241</td>
      <td>224.65</td>
      <td>14.87</td>
      <td>152.63</td>
      <td>0.011194</td>
      <td>0.000997</td>
      <td>0.065924</td>
      <td>0.050578</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.0835</td>
      <td>-0.015</td>
      <td>0.0551</td>
      <td>0.008241</td>
      <td>224.65</td>
      <td>14.87</td>
      <td>152.63</td>
      <td>0.011194</td>
      <td>0.000997</td>
      <td>0.065924</td>
      <td>0.050578</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.0835</td>
      <td>-0.015</td>
      <td>0.0551</td>
      <td>0.008241</td>
      <td>224.65</td>
      <td>14.87</td>
      <td>152.63</td>
      <td>0.011194</td>
      <td>0.000997</td>
      <td>0.065924</td>
      <td>0.050578</td>
    </tr>
  </tbody>
</table>
</div>




```python
f_Method_SES=RF.Simple_Exponential_Smoothing_Forecast(df_in_sample, df_out_sample, 'CONTIL', 0.6)
```


![png](output_196_0.png)



```python
f_Method_SES.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-03-01</th>
      <td>-0.065966</td>
      <td>-0.021739</td>
      <td>0.108678</td>
      <td>0.00981</td>
      <td>234.348543</td>
      <td>18.179768</td>
      <td>152.604285</td>
      <td>0.013293</td>
      <td>-0.000041</td>
      <td>0.077533</td>
      <td>0.041281</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>-0.065966</td>
      <td>-0.021739</td>
      <td>0.108678</td>
      <td>0.00981</td>
      <td>234.348543</td>
      <td>18.179768</td>
      <td>152.604285</td>
      <td>0.013293</td>
      <td>-0.000041</td>
      <td>0.077533</td>
      <td>0.041281</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>-0.065966</td>
      <td>-0.021739</td>
      <td>0.108678</td>
      <td>0.00981</td>
      <td>234.348543</td>
      <td>18.179768</td>
      <td>152.604285</td>
      <td>0.013293</td>
      <td>-0.000041</td>
      <td>0.077533</td>
      <td>0.041281</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>-0.065966</td>
      <td>-0.021739</td>
      <td>0.108678</td>
      <td>0.00981</td>
      <td>234.348543</td>
      <td>18.179768</td>
      <td>152.604285</td>
      <td>0.013293</td>
      <td>-0.000041</td>
      <td>0.077533</td>
      <td>0.041281</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>-0.065966</td>
      <td>-0.021739</td>
      <td>0.108678</td>
      <td>0.00981</td>
      <td>234.348543</td>
      <td>18.179768</td>
      <td>152.604285</td>
      <td>0.013293</td>
      <td>-0.000041</td>
      <td>0.077533</td>
      <td>0.041281</td>
    </tr>
  </tbody>
</table>
</div>




```python
(df_out_sample-f_Method_SES).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-03-01</th>
      <td>0.150966</td>
      <td>-0.100261</td>
      <td>-0.351678</td>
      <td>0.00200</td>
      <td>5.451457</td>
      <td>1.160232</td>
      <td>-0.504285</td>
      <td>0.001089</td>
      <td>-0.003235</td>
      <td>0.003117</td>
      <td>-0.027683</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.139966</td>
      <td>0.005739</td>
      <td>-0.028678</td>
      <td>-0.00228</td>
      <td>8.151457</td>
      <td>2.110232</td>
      <td>-4.304285</td>
      <td>-0.002034</td>
      <td>-0.024942</td>
      <td>0.006137</td>
      <td>-0.003841</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.088966</td>
      <td>0.046739</td>
      <td>-0.046678</td>
      <td>-0.00351</td>
      <td>10.551457</td>
      <td>2.830232</td>
      <td>-8.604285</td>
      <td>-0.003396</td>
      <td>-0.028954</td>
      <td>0.008257</td>
      <td>-0.015944</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.129966</td>
      <td>0.082739</td>
      <td>-0.022678</td>
      <td>-0.00478</td>
      <td>13.251457</td>
      <td>3.350232</td>
      <td>-11.104285</td>
      <td>-0.002268</td>
      <td>-0.017320</td>
      <td>0.009422</td>
      <td>-0.027706</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.031966</td>
      <td>0.132739</td>
      <td>-0.043678</td>
      <td>-0.00379</td>
      <td>13.451457</td>
      <td>4.080232</td>
      <td>-12.204285</td>
      <td>-0.012485</td>
      <td>-0.007733</td>
      <td>0.012297</td>
      <td>-0.008210</td>
    </tr>
  </tbody>
</table>
</div>




```python
((df_out_sample-f_Method_SES)**2).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-03-01</th>
      <td>0.022791</td>
      <td>0.010052</td>
      <td>0.123677</td>
      <td>0.000004</td>
      <td>29.718383</td>
      <td>1.346138</td>
      <td>0.254304</td>
      <td>0.000001</td>
      <td>0.000010</td>
      <td>0.000010</td>
      <td>0.000766</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.019591</td>
      <td>0.000033</td>
      <td>0.000822</td>
      <td>0.000005</td>
      <td>66.446251</td>
      <td>4.453079</td>
      <td>18.526871</td>
      <td>0.000004</td>
      <td>0.000622</td>
      <td>0.000038</td>
      <td>0.000015</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.007915</td>
      <td>0.002185</td>
      <td>0.002179</td>
      <td>0.000012</td>
      <td>111.333245</td>
      <td>8.010212</td>
      <td>74.033724</td>
      <td>0.000012</td>
      <td>0.000838</td>
      <td>0.000068</td>
      <td>0.000254</td>
    </tr>
    <tr>
      <th>1980-06-01</th>
      <td>0.016891</td>
      <td>0.006846</td>
      <td>0.000514</td>
      <td>0.000023</td>
      <td>175.601112</td>
      <td>11.224054</td>
      <td>123.305150</td>
      <td>0.000005</td>
      <td>0.000300</td>
      <td>0.000089</td>
      <td>0.000768</td>
    </tr>
    <tr>
      <th>1980-07-01</th>
      <td>0.001022</td>
      <td>0.017620</td>
      <td>0.001908</td>
      <td>0.000014</td>
      <td>180.941695</td>
      <td>16.648292</td>
      <td>148.944577</td>
      <td>0.000156</td>
      <td>0.000060</td>
      <td>0.000151</td>
      <td>0.000067</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.mean((df_out_sample-f_Method_SES)**2)
```




    CONTIL          0.014358
    IBM             0.004682
    MARKET          0.015962
    RKFREE          0.000006
    CPI          1768.163085
    POIL          108.838698
    FRBIND        100.875058
    RINF            0.000073
    GIND            0.000157
    real_POIL       0.000753
    ROIL            0.003114
    dtype: float64




```python
np.std((df_out_sample-f_Method_SES)**2)
```




    CONTIL          0.020123
    IBM             0.005847
    MARKET          0.023549
    RKFREE          0.000007
    CPI          1235.213421
    POIL           73.032603
    FRBIND         99.424877
    RINF            0.000073
    GIND            0.000189
    real_POIL       0.000748
    ROIL            0.003148
    dtype: float64




```python
np.mean((df_out_sample-f_Method_MAF_1)**2)
```




    CONTIL               NaN
    CPI          2605.754167
    FRBIND        101.286233
    GIND            0.000163
    IBM             0.004186
    MARKET          0.007982
    MOBIL                NaN
    POIL          183.487608
    RINF            0.000046
    RKFREE          0.000005
    ROIL            0.003888
    real_POIL       0.001451
    dtype: float64




```python
co_of_variation_Method_SES=np.std((df_out_sample-f_Method_SES)**2)/np.mean((df_out_sample-f_Method_SES)**2)
co_of_variation_Method_SES
```




    CONTIL       1.401542
    IBM          1.248733
    MARKET       1.475277
    RKFREE       1.116010
    CPI          0.698586
    POIL         0.671017
    FRBIND       0.985624
    RINF         1.011821
    GIND         1.206223
    real_POIL    0.994007
    ROIL         1.011014
    dtype: float64




```python
MainDF_first_period.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-08-01</th>
      <td>0.058</td>
      <td>0.016</td>
      <td>0.095</td>
      <td>0.00789</td>
      <td>221.1</td>
      <td>14.00</td>
      <td>152.1</td>
      <td>0.010050</td>
      <td>-0.005882</td>
      <td>0.063320</td>
      <td>0.035153</td>
    </tr>
    <tr>
      <th>1979-09-01</th>
      <td>-0.033</td>
      <td>-0.032</td>
      <td>0.039</td>
      <td>0.00802</td>
      <td>223.4</td>
      <td>14.57</td>
      <td>152.7</td>
      <td>0.010403</td>
      <td>0.003945</td>
      <td>0.065219</td>
      <td>0.030000</td>
    </tr>
    <tr>
      <th>1979-10-01</th>
      <td>-0.136</td>
      <td>-0.079</td>
      <td>-0.097</td>
      <td>0.00913</td>
      <td>225.4</td>
      <td>15.11</td>
      <td>152.7</td>
      <td>0.008953</td>
      <td>0.000000</td>
      <td>0.067036</td>
      <td>0.027860</td>
    </tr>
    <tr>
      <th>1979-11-01</th>
      <td>0.081</td>
      <td>0.060</td>
      <td>0.116</td>
      <td>0.00819</td>
      <td>227.5</td>
      <td>15.52</td>
      <td>152.3</td>
      <td>0.009317</td>
      <td>-0.002620</td>
      <td>0.068220</td>
      <td>0.017653</td>
    </tr>
    <tr>
      <th>1979-12-01</th>
      <td>0.104</td>
      <td>-0.013</td>
      <td>0.086</td>
      <td>0.00747</td>
      <td>229.9</td>
      <td>17.03</td>
      <td>152.5</td>
      <td>0.010549</td>
      <td>0.001313</td>
      <td>0.074076</td>
      <td>0.085839</td>
    </tr>
  </tbody>
</table>
</div>




```python
MainDF_second_period.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-01</th>
      <td>-0.103</td>
      <td>0.066</td>
      <td>0.124</td>
      <td>0.00883</td>
      <td>233.2</td>
      <td>17.86</td>
      <td>152.7</td>
      <td>0.014354</td>
      <td>0.001311</td>
      <td>0.076587</td>
      <td>0.033897</td>
    </tr>
    <tr>
      <th>1980-02-01</th>
      <td>-0.087</td>
      <td>-0.062</td>
      <td>0.112</td>
      <td>0.01073</td>
      <td>236.4</td>
      <td>18.81</td>
      <td>152.6</td>
      <td>0.013722</td>
      <td>-0.000655</td>
      <td>0.079569</td>
      <td>0.038935</td>
    </tr>
    <tr>
      <th>1980-03-01</th>
      <td>0.085</td>
      <td>-0.122</td>
      <td>-0.243</td>
      <td>0.01181</td>
      <td>239.8</td>
      <td>19.34</td>
      <td>152.1</td>
      <td>0.014382</td>
      <td>-0.003277</td>
      <td>0.080651</td>
      <td>0.013599</td>
    </tr>
    <tr>
      <th>1980-04-01</th>
      <td>0.074</td>
      <td>-0.016</td>
      <td>0.080</td>
      <td>0.00753</td>
      <td>242.5</td>
      <td>20.29</td>
      <td>148.3</td>
      <td>0.011259</td>
      <td>-0.024984</td>
      <td>0.083670</td>
      <td>0.037440</td>
    </tr>
    <tr>
      <th>1980-05-01</th>
      <td>0.023</td>
      <td>0.025</td>
      <td>0.062</td>
      <td>0.00630</td>
      <td>244.9</td>
      <td>21.01</td>
      <td>144.0</td>
      <td>0.009897</td>
      <td>-0.028995</td>
      <td>0.085790</td>
      <td>0.025338</td>
    </tr>
  </tbody>
</table>
</div>




```python
MainDF_1=MainDF_first_period.append(MainDF_second_period)
```


```python
MainDF_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTIL</th>
      <th>IBM</th>
      <th>MARKET</th>
      <th>RKFREE</th>
      <th>CPI</th>
      <th>POIL</th>
      <th>FRBIND</th>
      <th>RINF</th>
      <th>GIND</th>
      <th>real_POIL</th>
      <th>ROIL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1978-02-01</th>
      <td>0.037</td>
      <td>-0.043</td>
      <td>0.010</td>
      <td>0.00494</td>
      <td>188.4</td>
      <td>8.84</td>
      <td>139.2</td>
      <td>0.006410</td>
      <td>0.002882</td>
      <td>0.046921</td>
      <td>0.011946</td>
    </tr>
    <tr>
      <th>1978-03-01</th>
      <td>0.003</td>
      <td>-0.063</td>
      <td>0.050</td>
      <td>0.00526</td>
      <td>189.8</td>
      <td>8.80</td>
      <td>140.9</td>
      <td>0.007431</td>
      <td>0.012213</td>
      <td>0.046365</td>
      <td>-0.011868</td>
    </tr>
    <tr>
      <th>1978-04-01</th>
      <td>0.180</td>
      <td>0.130</td>
      <td>0.063</td>
      <td>0.00491</td>
      <td>191.5</td>
      <td>8.82</td>
      <td>143.2</td>
      <td>0.008957</td>
      <td>0.016324</td>
      <td>0.046057</td>
      <td>-0.006625</td>
    </tr>
    <tr>
      <th>1978-05-01</th>
      <td>0.061</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>0.00513</td>
      <td>193.3</td>
      <td>8.81</td>
      <td>143.9</td>
      <td>0.009399</td>
      <td>0.004888</td>
      <td>0.045577</td>
      <td>-0.010435</td>
    </tr>
    <tr>
      <th>1978-06-01</th>
      <td>-0.059</td>
      <td>-0.004</td>
      <td>0.007</td>
      <td>0.00527</td>
      <td>195.3</td>
      <td>9.05</td>
      <td>144.9</td>
      <td>0.010347</td>
      <td>0.006949</td>
      <td>0.046339</td>
      <td>0.016722</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.mean((df_out_sample-f_Method_MAF_1)**2)
```




    CONTIL               NaN
    CPI          2605.754167
    FRBIND        101.286233
    GIND            0.000163
    IBM             0.004186
    MARKET          0.007982
    MOBIL                NaN
    POIL          183.487608
    RINF            0.000046
    RKFREE          0.000005
    ROIL            0.003888
    real_POIL       0.001451
    dtype: float64




```python

```


```python

```


```python

```


```python



```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
