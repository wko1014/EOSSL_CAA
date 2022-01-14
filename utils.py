import numpy as np
from mne.filter import filter_data

"""
EEG sample has to have [Ns, Nc, Nt, 1] form 
where Ns, Nc, and Nt are the number of trials, electrodes, and timepoints.
Label has to have [Ns, No] form where No is the number of possible classes.
"""

def load_D_MI(subject, fold, num_training_samples=None):
    # Load motor imagery EEGs
    path = 'define/your/own/path/'
    X = np.load(path + f'data_sbj{subject:02d}.npy')
    Y = np.load(path + f'label_sbj{subject:02d}.npy')

    # Randomize samples
    rand_idx = np.random.RandomState(seed=42).permutation(X.shape[0])
    num_tests = int(X.shape[0]/5) # 5-fold cross-validation, num_trains = 280

    if num_training_samples is None:
        num_training_samples = X.shape[0] - num_tests

    test_idx = rand_idx[num_tests * (fold - 1):num_tests * fold]
    train_idx = np.setdiff1d(rand_idx, test_idx)

    D_train = (X[train_idx, ...][:num_training_samples, ...], Y[train_idx, ...][:num_training_samples, ...])
    D_test = (X[test_idx, ...], Y[test_idx, ...])
    return D_train, D_test

def make_D_prime_MI(D_train):
    """
    This function makes D' which is used for the first pretext task,
    the stopped band prediction task using training samples.
    :return: D_train_prime, D_test_prime
    """
    X = np.squeeze(D_train[0])
    X = np.reshape(X, newshape=(-1, X.shape[-1]))
    rand_idx = np.random.RandomState(seed=42).permutation(X.shape[0])
    # Randomize single-channel training samples and randomly select 1000 samples
    num_samples = 1000
    X = X[rand_idx[: num_samples], ...]

    # Remove baseline
    X = X[..., 128:]

    # Band-stop filtering
    # delta range
    X[:int(num_samples/5), ...] = filter_data(
        data=X[:int(num_samples/5), ...], sfreq=100, l_freq=4, h_freq=.5, verbose=False
    )

    # theta range
    X[int(num_samples/5):2 * int(num_samples/5), ...] = filter_data(
        data=X[int(num_samples/5):2 * int(num_samples/5), ...], sfreq=100, l_freq=8, h_freq=4, verbose=False
    )

    # alpha range
    X[2 * int(num_samples/5):3 * int(num_samples/5), ...] = filter_data(
        data=X[2 * int(num_samples/5):3 * int(num_samples/5), ...], sfreq=100, l_freq=15, h_freq=8, verbose=False
    )

    # beta range
    X[3 * int(num_samples/5):4 * int(num_samples/5), ...] = filter_data(
        data=X[3 * int(num_samples/5):4 * int(num_samples/5), ...], sfreq=100, l_freq=30, h_freq=8, verbose=False
    )

    # gamma range
    X[4 * int(num_samples/5):, ...] = filter_data(
        data=X[4 * int(num_samples/5):, ...], sfreq=100, l_freq=59, h_freq=30, verbose=False
    )

    # Make label
    Y = np.reshape(np.tile(np.array([0, 1, 2, 3, 4]), (int(num_samples/5), 1)).T, (-1))
    # One-hot encoding
    Y = np.eye(np.unique(Y).shape[0])[Y]

    # Randomize
    rand_idx = np.random.RandomState(seed=5930).permutation(X.shape[0])
    test_idx = rand_idx[:int(X.shape[0]/5)]
    train_idx = np.setdiff1d(rand_idx, test_idx)
    D_train_prime = (X[train_idx, ...], Y[train_idx, ...])
    D_test_prime = (X[test_idx, ...], Y[test_idx, ...])
    return D_train_prime, D_test_prime

def make_D_double_prime_MI(D_train):
    """
    This function makes D" which is used for the second pretext task,
    the stationary condition detection task using training samples.
    :return: D_train_double_prime, D_test_double_prime
    """
    X = np.squeeze(D_train[0])
    X = np.reshape(X, newshape=(-1, X.shape[-1]))
    rand_idx = np.random.RandomState(seed=24).permutation(X.shape[0])
    # Randomize single-channel training samples and randomly select 1000 samples
    num_samples = 1000
    X = X[rand_idx[: num_samples], ...]

    # Remove baseline
    X = X[..., 128:]

    # stationary using moving average filter
    def maf(x, window):
        x = np.squeeze(x) # to suppress the final dimension
        for ns in range(x.shape[0]):
            for nc in range(x.shape[1]):
                x[ns, nc] -= np.convolve(x[ns, nc], np.ones(window), 'same') / window
        return np.expand_dims(x, -1) # to recover the final dimension
    X[int(num_samples / 4):2 * int(num_samples / 4), ...] = maf(X[:int(num_samples / 5), ...], 20)

    # trendstationay by adding/subtracting linearly spaced values
    def trend(x, adding=True):
        if adding:
            tmp = 1
        else:
            tmp = -1
        x = np.squeeze(x)  # to suppress the final dimension
        for ns in range(x.shape[0]):
            for nc in range(x.shape[1]):
                x[ns, nc] += tmp * (np.random.random(1) * np.linspace(0, 10, x.shape[-1]) - 5)
        return np.expand_dims(x, -1) # to recover the final dimension
    # adding
    X[4 * int(num_samples / 8):5 * int(num_samples / 8), ...] = trend(
        X[4 * int(num_samples / 8):5 * int(num_samples / 8), ...])
    # subtracting
    X[5 * int(num_samples / 8):6 * int(num_samples / 8), ...] = trend(
        X[5 * int(num_samples / 8):6 * int(num_samples / 8), ...], False)

    # cyclostationay by adding periodic values
    def cyclo(x):
        x = np.squeeze(x)  # to suppress the final dimension
        for ns in range(x.shape[0]):
            for nc in range(x.shape[1]):
                tmp = np.random.random(1)
                x[ns, nc] += np.random.random(1) * np.sin(np.linspace(-np.pi + tmp, np.pi + tmp, x.shape[-1]))
        return np.expand_dims(x, -1) # to recover the final dimension
    X[3 * int(num_samples / 4):, ...] = cyclo(X[3 * int(num_samples / 4):, ...])

    # Make label
    Y = np.reshape(np.tile(np.array([0, 1, 2, 3]), (int(num_samples/4), 1)).T, (-1))
    # One-hot encoding
    Y = np.eye(np.unique(Y).shape[0])[Y]

    # Randomize
    rand_idx = np.random.RandomState(seed=5930).permutation(X.shape[0])
    test_idx = rand_idx[:int(X.shape[0]/5)]
    train_idx = np.setdiff1d(rand_idx, test_idx)
    D_train_prime = (X[train_idx, ...], Y[train_idx, ...])
    D_test_prime = (X[test_idx, ...], Y[test_idx, ...])
    return D_train_prime, D_test_prime