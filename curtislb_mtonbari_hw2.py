import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cross_validation as cv
import sklearn.feature_selection as fs
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score

N_FEATURES = 33
# N_FEATURES = 99
CHR1_LINES = 379551
CHR2_LINES = 297172

# Calculates the mean of an array of numbers, ignoring nan values
def mean_vals(arr):
    total = 0.0
    count = 0
    for x in arr:
        if not math.isnan(x):
            total += x
            count += 1
    return total / count

# Replaces all nan values in arr with the mean of the remaining values
def impute_mean(arr):
    mean = mean_vals(arr)
    for i in xrange(len(arr)):
        if math.isnan(arr[i]):
            arr[i] = mean
    return arr

# Converts a line to a start position and list of methylation values
def parse_line(l):
    tokens = l.rstrip().split()
    return int(tokens[1]), impute_mean([float(b) for b in tokens[4:-1]])

# Generates an array of feature sets from a training data file
def beta_from_train(fname, fsize):
    with open(fname) as f:
        X = np.empty((fsize, N_FEATURES), float)
#         curr_pos, curr_beta = parse_line(f.readline())
#         next_pos, next_beta = parse_line(f.readline())
#         line = f.readline()
        i = 0
        for line in f:
            row = parse_line(line)[1]
#         while line != '':
#             prev_pos, prev_beta = curr_pos, curr_beta
#             curr_pos, curr_beta = next_pos, next_beta
#             next_pos, next_beta = parse_line(line)
#             row = curr_beta[:]
# #             row.append(curr_pos - prev_pos)
#             row.extend(prev_beta)
# #             row.append(next_pos - curr_pos)
#             row.extend(next_beta)
#             line = f.readline()
            X[i,:] = np.array(row)
            i += 1
        return X

# Generates an array of methylation values from a sample or test data file
def sample_vals_from_file(fname, fsize):
    with open(fname) as f:
        y = np.empty(fsize, float)
        i = 0
        for line in f:
            y[i] = float(line.rstrip().split()[-2])
            i += 1
        return y

# Generates array masks indicating the train and test sets for a sample file
def train_test_sample_masks(fname, fsize):
    with open(fname) as f:
        train_mask = np.empty(fsize, bool)
        test_mask = np.empty(fsize, bool)
        i = 0
        for line in f:
            tokens = line.rstrip().split()
            if tokens[-2] == 'nan':
                train_mask[i] = False
                test_mask[i] = False
            elif tokens[-1] == '1':
                train_mask[i] = True
                test_mask[i] = False
            else:
                train_mask[i] = False
                test_mask[i] = True
            i += 1
        return train_mask, test_mask
#         return train_mask[1:-1], test_mask[1:-1]

# Generates predictions by computing the mean of each row of features
def naive_mean_impute(X):
    y_pred = np.empty(X.shape[0], float)
    for i in xrange(X.shape[0]):
        y_pred[i] = np.mean(X[i,:])
    return y_pred

# Computes the residuals for a set of predictions compared to correct values
def residuals(y_true, y_pred):
    count = len(y_true)
    res = np.empty(count, float)
    for i in xrange(count):
        res[i] = y_true[i] - y_pred[i]
    return res

# Prints summary statistics and generates residual plot for a set of predictions
def analyze(y_true, y_pred, method=''):
    print method, 'R^2 =', r2_score(y_true, y_pred)
    print method, 'RMSE =', math.sqrt(mean_squared_error(y_true, y_pred))
    res = residuals(y_true, y_pred)
    plt.clf()
    plt.hist(res, bins=50)
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.savefig(method + '_residuals.png')

if __name__ == '__main__':
    X = beta_from_train('data/intersected_final_chr1_cutoff_20_train_revised.bed', CHR1_LINES)
    y_true = sample_vals_from_file('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)#[1:-1]
    train_mask, test_mask = train_test_sample_masks('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)
    
#     X = beta_from_train('data/intersected_final_chr2_cutoff_20_train_revised.bed', CHR2_LINES)
#     y_true = sample_vals_from_file('data/intersected_final_chr2_cutoff_20_test.bed', CHR2_LINES)[1:-1]
#     train_mask, test_mask = train_test_sample_masks('data/intersected_final_chr2_cutoff_20_test.bed', CHR2_LINES)
    
    # Naive mean imputation
    y_pred = naive_mean_impute(X)
    analyze(y_true[test_mask], y_pred[test_mask], 'NaiveMean')

    # Linear model imputation
    model = lm.LinearRegression()
    model.fit(X[train_mask], y_true[train_mask])
    y_pred = model.predict(X[test_mask])
    analyze(y_true[test_mask], y_pred, 'LinearRegression')
    
#     model = lm.PassiveAggressiveRegressor()
#     model.fit(X[train_mask], y_true[train_mask])
#     y_pred = model.predict(X[test_mask])
#     analyze(y_true[test_mask], y_pred, 'PassiveAggressiveRegressor')

    # Determining best features
    skb = fs.SelectKBest(k=7)
    skb.fit(X[train_mask], y_true[train_mask])
    feature_mask = skb.get_support()
    for i in xrange(len(feature_mask)):
        if feature_mask[i]:
            print i
    for score in skb.scores_[feature_mask]:
        print score

    val_mask = train_mask | test_mask
    X = X[val_mask]
    y_true = y_true[val_mask]
     
    # K-fold cross validation
    kf = cv.KFold(len(y_true), n_folds=5)
    model = lm.SGDRegressor()
    fold = 1
    for train, test in kf:
        print '--- Fold', fold, '-------------------'
        y_pred = naive_mean_impute(X[test])
        analyze(y_true[test], y_pred, 'NaiveMean_neighbor_cv' + str(fold))
        model.fit(X[train], y_true[train])
        y_pred = model.predict(X[test])
        analyze(y_true[test], y_pred, 'SGDRegressor_neighbor_cv' + str(fold))
        fold += 1
