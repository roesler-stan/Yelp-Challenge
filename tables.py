import pandas as pd
import statsmodels.api as sm
from table_stars import *
import locale

def main():
    infile = '../Data/reviews_clean.csv'
    df = pd.read_csv(infile)

    means_file = '../Output/means.csv'
    models_file = '../Output/models.txt'
    
    means_table(df, means_file)
    ols_table(df, models_file)

def means_table(df, outfile):
    """ Make table of dataset means """
    cols = ['year', 'word_count', 'characters_count', 'stars_business', 'stars_review', 'polarity',
    'good_present', 'bad_present', 'good_count', 'bad_count', 'food_poisoning_present']
    means_table = df[cols].mean().reset_index()
    means_table = means_table.rename(columns = {0: 'Mean'})
    sd_table = df[cols].std().reset_index()
    sd_table = sd_table.rename(columns = {0: 'Std. Dev.'})
    table = means_table.merge(sd_table, on = ['index'])
    table = table.rename(columns = {'index': 'Measure'})
    table['Measure'] = table['Measure'].str.replace('_', ' ').str.title()
    table['Mean'] = table['Mean'].round(2)
    table['Std. Dev.'] = table['Std. Dev.'].round(2)
    table.to_csv(outfile, index = False)

def ols_table(df, outfile):
    """ Run OLS models and put in table """
    feature_set1 = ['polarity']
    feature_set2 = ['word_count', 'good_count', 'bad_count']
    feature_set3 = ['word_count', 'good_count', 'bad_count', 'service_count', 'food_count', 'food_poisoning_present']
    feature_set4 = ['polarity', 'word_count', 'good_count', 'bad_count', 'service_count', 'food_count', 'food_poisoning_present']
    cols = feature_set4 + ['stars_review']
    complete_data = df[cols].dropna()
    # Standardize features (feature set 2 contains all features)
    complete_data[feature_set4] = (complete_data[feature_set4] - complete_data[feature_set4].mean()) / complete_data[feature_set4].std()
    Y = complete_data['stars_review'].astype(float)
    X1 = complete_data[feature_set1]
    X2 = complete_data[feature_set2]
    X3 = complete_data[feature_set3]
    X4 = complete_data[feature_set4]
    X1 = sm.add_constant(X1)
    X2 = sm.add_constant(X2)
    X3 = sm.add_constant(X3)
    X4 = sm.add_constant(X4)
    model1 = sm.OLS(Y, X1).fit()
    model2 = sm.OLS(Y, X2).fit()
    model3 = sm.OLS(Y, X3).fit()
    model4 = sm.OLS(Y, X4).fit()
    # Format some numbers with a comma
    locale.setlocale(locale.LC_ALL, 'en_US')
    title = "Linear Regression Predicting Review's Stars"
    table = s2.summary_col([model1, model2, model3, model4], model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4'],
            stars = True, regressor_order = X4.columns.tolist(), float_format = '%.2f',
            info_dict = {'N': lambda x: locale.format("%d", int(x.nobs), grouping = True),
            'R2':lambda x: "{:.2f}".format(x.rsquared),
            'AIC': lambda x: locale.format("%.1f", int(x.aic), grouping = True)})
    table.add_title(title)
    with open(outfile, 'w') as f:
        f.write(table.as_text())

if __name__ == "__main__":
    main()