import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

N_FEATURES = 33
CHR1_LINES = 379551

def rmse(y_pred, y_true):
    sse = 0.0
    n = 0
    for i in xrange(len(y_pred)):
        if not math.isnan(y_true[i]):
            err = y_pred[i] - y_true[i]
            sse += err * err
            n += 1
    return math.sqrt(sse / n)

def naive_mean_impute(fname, fsize):
    with open(fname) as f:
        y = np.empty(fsize, float)
        i = 0
        for line in f:
            beta = [float(b) for b in line.rstrip().split()[4:-1]]
            b_sum = 0.0
            b_count = 0
            for b in beta:
                if not math.isnan(b):
                    b_sum += b
                    b_count += 1
            y[i] = b_sum / float(b_count)
            i += 1
        return y
    
def sample_vals_from_file(fname, fsize):
    with open(fname) as f:
        y = np.empty(fsize, float)
        i = 0
        for line in f:
            y[i] = float(line.rstrip().split()[-2])
            i += 1
        return y
    
def beta_from_train(fname, fsize):
    with open(fname) as f:
        X = np.empty((fsize, N_FEATURES), float)
        i = 0
        for line in f:
            beta = [float(b) for b in line.rstrip().split()[4:-1]]
            X[i,:] = np.array(beta)
            i += 1
        return X

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

if __name__ == '__main__':
    y_pred = naive_mean_impute('data/intersected_final_chr1_cutoff_20_train_revised.bed', CHR1_LINES)
    y_true = sample_vals_from_file('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)
    print 'Naive Mean RMSE:', rmse(y_pred, y_true)

    X = beta_from_train('data/intersected_final_chr1_cutoff_20_train_revised.bed', CHR1_LINES)
    X = Imputer(strategy='mean', axis=1).transform(X)

    train_mask, test_mask = train_test_sample_masks('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)
    
    lr = LinearRegression()
    lr.fit(X[train_mask], y_true[train_mask])
    
    y_pred = lr.predict(X[test_mask])
    print 'Model RMSE:', rmse(y_pred, y_true[test_mask])
