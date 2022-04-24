# Import APIs
import network
import utils
import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans # for ALN

# to check the GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class experiment():
    def __init__(self, test_sbj, Nk=4):
        self.test_sbj = test_sbj
        self.Fs = 100

        # K-Means++ Buffer
        self.kmeans = KMeans(n_clusters=Nk, init='k-means++')

        # Learning schedules
        self.N_batches = 1 # for easily understandable implementation
        self.N_iters_pretext = 100
        self.N_iters = 200
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-3,
                                                            decay_steps=100, decay_rate=.96)
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

    def training(self):
        print(f'Start Pretext Task Training: Test Subject {self.test_sbj}')

        # Load dataset
        D_train, D_test = utils.load_D_MI(self.test_sbj)

        # Call Networks
        e1 = network.E1(Fs=self.Fs, Nc=D_train[0][0, ...].shape[0], Nt=D_train[0][0, ...].shape[1])
        e2 = network.E2(Nc=D_train[0][0, ...].shape[0], Nt=D_train[0][0, ...].shape[1])
        aln = network.ALN()
        c = network.C(No=D_train[1][0, ...].shape[-1])

        # Call loss functions
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Generate stopped band prediction pretext dataset
        D_train_prime, _ = utils.make_D_prime_MI(D_train) # not check the pretext performance here

        tmp = int(D_train_prime[1].shape[0]/self.N_batches)

        for iteration in range(self.N_iters_pretext):
            # randomization
            rand_idx = D_train_prime[0].shape[0]
            D_train_prime[0] = D_train_prime[0][rand_idx, ...]
            D_train_prime[1] = D_train_prime[1][rand_idx, ...]
            for batch in range(tmp):
                # Draw a minibatch
                xb = D_train_prime[0][batch * self.N_batches:(batch + 1) * self.N_batches, ...]
                yb = D_train_prime[1][batch * self.N_batches:(batch + 1) * self.N_batches, ...]

                # Estimate gradient and loss
                with tf.GradientTape() as tape:
                    f1 = e1.spectral_embedding(x=xb)
                    y1 = e1.band_prediction(f1)

                    # CCE loss
                    L = cce(yb, y1)
                theta = e1.trainable_variables
                grad = tape.gradient(L, theta)
                self.opt.apply_gradients(zip(grad, theta))

        del D_train_prime

        # Generate stationary condition detection dataset
        D_train_double_prime, _ = utils.make_D_double_prime_MI(D_train) # not check the pretext performance here

        for iteration in range(self.N_iters_pretext):
            # randomization
            rand_idx = D_train_double_prime[0].shape[0]
            D_train_double_prime[0] = D_train_double_prime[0][rand_idx, ...]
            D_train_double_prime[1] = D_train_double_prime[1][rand_idx, ...]
            for batch in range(tmp):
                # Draw a minibatch
                xb = D_train_double_prime[0][batch * self.N_batches:(batch + 1) * self.N_batches, ...]
                yb = D_train_double_prime[1][batch * self.N_batches:(batch + 1) * self.N_batches, ...]

                # Estimate gradient and loss
                with tf.GradientTape() as tape:
                    f2 = e2.temporal_embedding(x=xb)
                    y2 = e2.stationary_detection(f2)

                    # CCE loss
                    L = cce(yb, y2)
                theta = e2.trainable_variables
                grad = tape.gradient(L, theta)
                self.opt.apply_gradients(zip(grad, theta))

        del D_train_double_prime

        print(f'Start Downstream Task Training: Test Subject {self.test_sbj}')

        tmp = int(D_train[1].shape[0]/self.N_batches)


        for iteration in range(self.N_iters):
            # randomization
            rand_idx = D_train[0].shape[0]
            D_train[0] = D_train[0][rand_idx, ...]
            D_train[1] = D_train[1][rand_idx, ...]
            for batch in range(tmp):
                # Draw a minibatch
                xb = D_train[0][batch * self.N_batches:(batch + 1) * self.N_batches, ...]
                yb = D_train[1][batch * self.N_batches:(batch + 1) * self.N_batches, ...]

                # Estimate gradient and loss
                with tf.GradientTape() as tape:
                    f1 = e1.spectral_embedding(x=xb)
                    f2 = e2.temporal_embedding(x=xb)

                    mu, sigma, f_concat = aln.statistics(f1=f1, f2=f2)
                    self.kmeans.fit(np.concatenate((mu, sigma), -1))
                    stats = self.kmeans.cluster_centers_[self.kmeans.labels_]
                    mu_star, sigma_star = stats[..., :120], stats[..., 120:]
                    f_aln = aln.normalization(f_concat=f_concat, mu_star=mu_star, sigma_star=sigma_star)
                    y_hat = c.classification(cluster=self.kmeans, f_ALN=f_aln)

                    # WCE loss
                    L = tf.nn.weighted_cross_entropy_with_logits(labels=yb, logits=y_hat, pos_weight=.5)
                var_e1 = e1.trainable_variables
                var_e2 = e2.trainable_variables
                var_c = c.trainable_variables
                theta = [var_e1[0], var_e1[1], var_e1[2], var_e1[3], var_e1[4],
                         var_e1[5], var_e1[6], var_e1[7], var_e1[8], var_e1[9],
                         var_e2[0], var_e2[1], var_e2[2], var_e2[3], var_e2[4],
                         var_e2[5], var_e2[6], var_e2[7], var_e2[8], var_e2[9],
                         var_e2[10], var_e2[11], var_c]
                grad = tape.gradient(L, theta)
                self.opt.apply_gradients(zip(grad, theta))




