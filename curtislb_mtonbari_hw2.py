import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer

# N_FEATURES = 33
N_FEATURES = 99
CHR1_LINES = 379551

def parse_line(l):
    tokens = l.rstrip().split()
    return int(tokens[1]), [float(b) for b in tokens[4:-1]]

def beta_from_train(fname, fsize):
    with open(fname) as f:
        X = np.empty((fsize, N_FEATURES), float)
        curr_pos, curr_beta = parse_line(f.readline())
        next_pos, next_beta = parse_line(f.readline())
        line = f.readline()
        i = 0
#         for line in f:
#             row = parse_line(line)[1]
#             X[i,:] = np.array(row)
        while line != '':
            prev_pos, prev_beta = curr_pos, curr_beta
            curr_pos, curr_beta = next_pos, next_beta
            next_pos, next_beta = parse_line(line)
            row = curr_beta[:]
#             row.append(curr_pos - prev_pos)
            row.extend(prev_beta)
#             row.append(next_pos - curr_pos)
            row.extend(next_beta)
            line = f.readline()
            i += 1
        return X

def sample_vals_from_file(fname, fsize):
    with open(fname) as f:
        y = np.empty(fsize, float)
        i = 0
        for line in f:
            y[i] = float(line.rstrip().split()[-2])
            i += 1
        return y

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
#         return train_mask, test_mask
        return train_mask[1:-1], test_mask[1:-1]

def naive_mean_impute(X):
    y_pred = np.empty(X.shape[0], float)
    for i in xrange(X.shape[0]):
        y_pred[i] = np.mean(X[i,:])
    return y_pred

def residuals(y_true, y_pred):
    count = len(y_true)
    res = np.empty(count, float)
    for i in xrange(count):
        res[i] = y_true[i] - y_pred[i]
    return res

def analyze(y_true, y_pred, method=''):
    print method, 'R^2 =', r2_score(y_true, y_pred)
    print method, 'RMSE =', math.sqrt(mean_squared_error(y_true, y_pred))
    res = residuals(y_true, y_pred)
    plt.clf()
    plt.hist(res, 50)
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.savefig(method + '_residuals.png')

if __name__ == '__main__':
    X = beta_from_train('data/intersected_final_chr1_cutoff_20_train_revised.bed', CHR1_LINES)
    X = Imputer(strategy='mean', axis=1).transform(X)
    
    y_true = sample_vals_from_file('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)[1:-1]
    train_mask, test_mask = train_test_sample_masks('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)
    
#     # Naive mean imputation
#     y_pred = naive_mean_impute(X)
#     analyze(y_true[test_mask], y_pred[test_mask], 'NaiveMean')

    # Linear model imputation
    model = lm.PassiveAggressiveRegressor()
    model.fit(X[train_mask], y_true[train_mask])
    y_pred = model.predict(X[test_mask])
    analyze(y_true[test_mask], y_pred, 'PassiveAggressiveRegressor_neighbor')
