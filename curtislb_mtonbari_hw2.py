import numpy as np

def beta_from_train(fname):
    with open(fname) as f:
        X = np.array([])
        for line in f:
            print 'hi'
            beta = line.rstrip().split()[4:-1]
            X = np.vstack(X, beta)
        return X

def train_model(model):
    pass

def test_model(model):
    pass

if __name__ == '__main__':
    X = beta_from_train('data/intersected_final_chr1_cutoff_20_train_revised.bed')
    print X
