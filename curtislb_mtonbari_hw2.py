import math
import numpy as np

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
    
def test_vals_from_file(fname, fsize):
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
            if i % 1000 == 0:
                print i
            beta = [float(b) for b in line.rstrip().split()[4:-1]]
            X[i,:] = np.array(beta)
            i += 1
        return X

def train_model(model):
    pass

def test_model(model):
    pass

if __name__ == '__main__':
#     y_pred = naive_mean_impute('data/intersected_final_chr1_cutoff_20_train_revised.bed', CHR1_LINES)
#     y_true = test_vals_from_file('data/intersected_final_chr1_cutoff_20_test.bed', CHR1_LINES)
#     print rmse(y_pred, y_true)
    X = beta_from_train('data/intersected_final_chr1_cutoff_20_train_revised.bed', CHR1_LINES)
