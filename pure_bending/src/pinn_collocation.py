import numpy as np
import tensorflow as tf


class first_derivatives(tf.keras.layers.Layer):

    def __init__(self, neural_net, **kwargs):
        self.neural_net = neural_net
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        with tf.GradientTape(persistent=True) as grad1:
            grad1.watch(x)
            grad1.watch(y)
            u = self.neural_net(tf.concat([x, y], axis=-1))
        du_dx = grad1.gradient(u, x)
        du_dy = grad1.gradient(u, y)
        del grad1
        return du_dx, du_dy


class second_derivatives(tf.keras.layers.Layer):

    def __init__(self, neural_net, **kwargs):
        self.neural_net = neural_net
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        with tf.GradientTape(persistent=True) as grad2:
            grad2.watch(x)
            grad2.watch(y)
            with tf.GradientTape(persistent=True) as grad1:
                grad1.watch(x)
                grad1.watch(y)
                u = self.neural_net(tf.concat([x, y], axis=-1))
            du_dx = grad1.gradient(u, x)
            du_dy = grad1.gradient(u, y)
            del grad1
        du_dxx = grad2.gradient(du_dx, x)
        du_dxy = grad2.gradient(du_dx, y)
        du_dyy = grad2.gradient(du_dy, y)
        del grad2
        return du_dxx, du_dxy, du_dyy


class displacement_x_PINN(tf.keras.layers.Layer):

    def __init__(self, neural_net_x, **kwargs):
        self.neural_net_x = neural_net_x
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        u = self.neural_net_x(tf.concat([x, y], axis=-1))
        return u


class displacement_x_RBMR(tf.keras.layers.Layer):

    def __init__(self, neural_net_x, neural_net_y, **kwargs):
        self.neural_net_x = neural_net_x
        self.neural_net_y = neural_net_y
        self.origin1 = tf.convert_to_tensor(np.array([0, 0]), dtype='float32')
        self.origin2 = tf.convert_to_tensor(np.array([0, 0.05]), dtype='float32')
        self.origin1 = tf.expand_dims(self.origin1, axis=0)
        self.origin2 = tf.expand_dims(self.origin2, axis=0)
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        u = self.neural_net_x(tf.concat([x, y], axis=-1))
        origins = tf.concat([self.origin1, self.origin2], axis=0)
        temp_u = self.neural_net_x(origins)
        temp1_u, temp2_u = temp_u[0], temp_u[1]
        u = u - temp1_u - y * (temp2_u - temp1_u) / (0.05 - 0.)
        return u


class displacement_y_PINN(tf.keras.layers.Layer):

    def __init__(self, neural_net_y, **kwargs):
        self.neural_net_y = neural_net_y
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        u = self.neural_net_y(tf.concat([x, y], axis=-1))
        return u


class displacement_y_RBMR(tf.keras.layers.Layer):

    def __init__(self, neural_net_x, neural_net_y, **kwargs):
        self.neural_net_x = neural_net_x
        self.neural_net_y = neural_net_y
        self.origin1 = tf.convert_to_tensor(np.array([0, 0]), dtype='float32')
        self.origin2 = tf.convert_to_tensor(np.array([0, 0.05]), dtype='float32')
        self.origin1 = tf.expand_dims(self.origin1, axis=0)
        self.origin2 = tf.expand_dims(self.origin2, axis=0)
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        u = self.neural_net_y(tf.concat([x, y], axis=-1))
        origins = tf.concat([self.origin1, self.origin2], axis=0)
        temp_u = self.neural_net_x(origins)
        temp1_u, temp2_u = temp_u[0], temp_u[1]
        temp1_v = self.neural_net_y(self.origin1)
        u = u - temp1_v + x * (temp2_u - temp1_u) / (0.05 - 0.)
        return u


class pinn_collocation:
    """
    """

    def __init__(self, method, post, Xp, Xp_errNorm, npx, npy, young, nu, hidden_layers, nEpochs, refSol):
        self.method = method
        self.nEpochs = nEpochs
        self.post = post
        self.Xp = Xp
        self.Xp_errNorm = Xp_errNorm
        self.npx = npx
        self.npy = npy
        self.E = young
        self.nu = nu
        self.C_matrix = self.E / (1. + self.nu) / (1. - 2. * self.nu) * np.array(
            [[1. - self.nu, self.nu, 0.], [self.nu, 1. - self.nu, 0.], [0., 0., 0.5 - self.nu]])
        self.refSol = refSol
        tf.keras.utils.set_random_seed(42)
        x1 = tf.keras.layers.Input(shape=[2])
        temp1 = x1
        for l in hidden_layers:
            temp1 = tf.keras.layers.Dense(l, activation='silu', kernel_initializer='he_normal')(temp1)
        y1 = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='he_normal')(temp1)
        nnx = tf.keras.models.Model(inputs=x1, outputs=y1)
        tf.keras.utils.set_random_seed(42)
        x2 = tf.keras.layers.Input(shape=[2])
        temp2 = x2
        for l in hidden_layers:
            temp2 = tf.keras.layers.Dense(l, activation='silu', kernel_initializer='he_normal')(temp2)
        y2 = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='he_normal')(temp2)
        nny = tf.keras.models.Model(inputs=x2, outputs=y2)
        self.neural_net = [nnx, nny]
        self.iter = int(0)

    def construct(self):
        self.model = self.generate_collocation_model()

    def generate_collocation_model(self):
        """
          Model construction
        """
        X = tf.keras.layers.Input(shape=(2,))
        X_up = tf.keras.layers.Input(shape=(2,))
        X_bottom = tf.keras.layers.Input(shape=(2,))
        X_right = tf.keras.layers.Input(shape=(2,))
        X_dbcx = tf.keras.layers.Input(shape=(2,))
        X_dbcy = tf.keras.layers.Input(shape=(2,))
        if self.method == 'standard PINN':
            displacement_x = displacement_x_PINN(self.neural_net[0])
        else:
            displacement_x = displacement_x_RBMR(self.neural_net[0], self.neural_net[1])
        u_dbcx = displacement_x(X_dbcx)
        if self.method == 'standard PINN':
            displacement_y = displacement_y_PINN(self.neural_net[1])
        else:
            displacement_y = displacement_y_RBMR(self.neural_net[0], self.neural_net[1])
        u_dbcy = displacement_y(X_dbcy)
        mu = self.E / 2. / (1. + self.nu)
        lambda_ = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        first_derivatives_ux = first_derivatives(self.neural_net[0])
        dux_dx, dux_dy = first_derivatives_ux(X_up)
        first_derivatives_uy = first_derivatives(self.neural_net[1])
        duy_dx, duy_dy = first_derivatives_uy(X_up)
        sx_up = mu * dux_dy + mu * duy_dx
        sy_up = (2 * mu + lambda_) * duy_dy + lambda_ * dux_dx
        dux_dx, dux_dy = first_derivatives_ux(X_bottom)
        duy_dx, duy_dy = first_derivatives_uy(X_bottom)
        sx_bottom = mu * dux_dy + mu * duy_dx
        sy_bottom = (2 * mu + lambda_) * duy_dy + lambda_ * dux_dx
        dux_dx, dux_dy = first_derivatives_ux(X_right)
        duy_dx, duy_dy = first_derivatives_uy(X_right)
        sy_right = mu * dux_dy + mu * duy_dx
        sx_right = (2 * mu + lambda_) * dux_dx + lambda_ * duy_dy
        second_derivatives_ux = second_derivatives(self.neural_net[0])
        dux_dxx, dux_dxy, dux_dyy = second_derivatives_ux(X)
        second_derivatives_uy = second_derivatives(self.neural_net[1])
        duy_dxx, duy_dxy, duy_dyy = second_derivatives_uy(X)
        PDEx = (2 * mu + lambda_) * dux_dxx + mu * dux_dyy + (mu + lambda_) * duy_dxy
        PDEy = (mu + lambda_) * dux_dxy + (2 * mu + lambda_) * duy_dyy + mu * duy_dxx
        return tf.keras.models.Model(inputs=[X, X_up, X_bottom, X_right, X_dbcx, X_dbcy],
                                     outputs=[PDEx, PDEy, sx_up, sy_up, sx_bottom, sy_bottom, sx_right, sy_right,
                                              u_dbcx, u_dbcy])

    def train_data_loader(self):
        """
          X_train, y_train
        """
        numpar = (self.npx + 1) * (self.npy + 1)
        n_up = np.arange((self.npx + 1) * self.npy, numpar)
        n_bot = np.arange(0, self.npx + 1)
        n_right = np.arange(self.npx, numpar, self.npx + 1)
        ndbc_x = np.arange(0, numpar, self.npx + 1)
        ndbc_y = np.arange(0, numpar, self.npx + 1)
        self.X_train = [self.Xp, self.Xp[n_up, :], self.Xp[n_bot, :], self.Xp[n_right, :], self.Xp[ndbc_x, :],
                        self.Xp[ndbc_y, :]]
        self.y_train = [np.expand_dims(np.zeros(self.npx + 1), axis=1), np.expand_dims(np.zeros(self.npx + 1), axis=1),
                        np.expand_dims(np.zeros(self.npx + 1), axis=1), np.expand_dims(np.zeros(self.npx + 1), axis=1),
                        np.expand_dims(-1000. * self.Xp[n_right, 1], axis=1),
                        np.expand_dims(np.zeros(self.npy + 1), axis=1),
                        np.expand_dims(np.zeros(self.npy + 1), axis=1),
                        np.expand_dims(0.195 * self.Xp[ndbc_y, 1] ** 2, axis=1)]
        return

    def Loss_tf(self, epoch):
        """
          Loss_tensorflow function
        """
        X_train = [tf.constant(x, dtype='float32') for x in self.X_train]
        y_train = [tf.constant(y, dtype='float32') for y in self.y_train]
        if self.iter % 20 == 0:
            loss, self.loss_coef1, self.loss_coef2, self.loss_coef3 = self.Loss_tf_grad_standard_ntk(X_train, y_train)
        else:
            loss = self.Loss_tf_grad_standard(X_train, y_train, self.loss_coef1, self.loss_coef2, self.loss_coef3)
        if (self.iter % 100 == 0) or (self.iter == self.nEpochs - 1):
            outputs = self.predict_outputs(self.Xp_errNorm)
            disp_err, energy_err = self.post.postprocess_err(outputs)
            print(f'Epoch={epoch}, Loss={loss.numpy():.3e}, Error Norms: {disp_err:.2e}, {energy_err:.3e}')

        self.iter += 1
        return loss

    @tf.function
    def Loss_tf_grad_standard(self, x, y, loss_coef1, loss_coef2, loss_coef3):
        l1, l2, l3, n1, n2, n3 = self.calc_loss(x, y)
        loss = loss_coef1 * l1 + loss_coef2 * l2 + loss_coef3 * l3
        return loss

    @tf.function
    def calc_loss(self, x, y):
        n1 = tf.constant(2 * x[0].shape[0], dtype='float32')
        n2 = tf.constant(y[0].shape[0] + y[1].shape[0] + y[2].shape[0] + y[3].shape[0] + y[4].shape[0]
                         + y[5].shape[0], dtype='float32')
        n3 = tf.constant(y[6].shape[0] + y[7].shape[0], dtype='float32')
        l1 = (tf.reduce_sum(tf.square(self.model(x)[0])) + tf.reduce_sum(tf.square(self.model(x)[1])))
        l2 = (tf.reduce_sum(tf.square(self.model(x)[2] - y[0]))
              + tf.reduce_sum(tf.square(self.model(x)[3] - y[1]))
              + tf.reduce_sum(tf.square(self.model(x)[4] - y[2]))
              + tf.reduce_sum(tf.square(self.model(x)[5] - y[3]))
              + tf.reduce_sum(tf.square(self.model(x)[6] - y[4]))
              + tf.reduce_sum(tf.square(self.model(x)[7] - y[5])))
        l3 = (tf.reduce_sum(tf.square(self.model(x)[8] - y[6]))
              + tf.reduce_sum(tf.square(self.model(x)[9] - y[7])))
        return l1, l2, l3, n1, n2, n3

    def Loss_tf_grad_standard_ntk(self, x, y):
        l1, l2, l3, n1, n2, n3 = self.calc_loss(x, y)
        ntk_coefficient1, ntk_coefficient2, ntk_coefficient3 = self.calc_ntk(x, n1, n2, n3)
        loss = ntk_coefficient1 * l1 + ntk_coefficient2 * l2 + ntk_coefficient3 * l3
        return loss, ntk_coefficient1, ntk_coefficient2, ntk_coefficient3

    @tf.function
    def calc_ntk(self, x, n1, n2, n3):
        def compute_ntk_for_idx(i, idx):
            ntk_coefficient1 = tf.constant(0.)
            ntk_coefficient2 = tf.constant(0.)
            ntk_coefficient3 = tf.constant(0.)

            xxx = [tf.zeros((1, 2)) for _ in range(len(x))]
            xxx[idx] = x[idx][tf.newaxis, i, ...]

            with tf.GradientTape(persistent=True) as grad:
                y = self.model(xxx)

            def l2_loss(tensor):
                return tf.reduce_sum(tf.square(tensor)) / 2

            if idx == 0:
                Grad = grad.gradient(y[0], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient1 += 2. * l2_loss(j)
                Grad = grad.gradient(y[1], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient1 += 2. * l2_loss(j)
            elif idx == 1:
                Grad = grad.gradient(y[2], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
                Grad = grad.gradient(y[3], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
            elif idx == 2:
                Grad = grad.gradient(y[4], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
                Grad = grad.gradient(y[5], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
            elif idx == 3:
                Grad = grad.gradient(y[6], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
                Grad = grad.gradient(y[7], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
            elif idx == 4:
                Grad = grad.gradient(y[8], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient3 += 2. * l2_loss(j)
            elif idx == 5:
                Grad = grad.gradient(y[9], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient3 += 2. * l2_loss(j)

            return ntk_coefficient1, ntk_coefficient2, ntk_coefficient3

        ntk_coefficient1, ntk_coefficient2, ntk_coefficient3 = tf.constant(0.), tf.constant(0.), tf.constant(0.)

        for idx in range(6):
            results = tf.vectorized_map(lambda i: compute_ntk_for_idx(i, idx), tf.range(x[idx].shape[0]))
            ntk_coefficient1 += tf.reduce_sum(results[0])
            ntk_coefficient2 += tf.reduce_sum(results[1])
            ntk_coefficient3 += tf.reduce_sum(results[2])

        ntk_coefficient1 /= n1
        ntk_coefficient2 /= n2
        ntk_coefficient3 /= n3
        sum_ntk = ntk_coefficient1 + ntk_coefficient2 + ntk_coefficient3
        ntk_coefficient1 = sum_ntk / ntk_coefficient1 if ntk_coefficient1 != 0 else tf.constant(0.)
        ntk_coefficient2 = sum_ntk / ntk_coefficient2 if ntk_coefficient2 != 0 else tf.constant(0.)
        ntk_coefficient3 = sum_ntk / ntk_coefficient3 if ntk_coefficient3 != 0 else tf.constant(0.)

        return ntk_coefficient1, ntk_coefficient2, ntk_coefficient3

    def predict_outputs(self, X):
        Up = self.predict_displacement(X)
        Ep = self.predict_strain(X)
        material_tangent = self.C_matrix
        Sp = np.zeros_like(Ep)
        for i in range(len(Ep)):
            Sp[i, :] = np.dot(material_tangent, Ep[i, :])
        return [Up, Sp, Ep]

    def predict_displacement(self, X):
        if self.method == 'standard PINN':
            displacement_x = displacement_x_PINN(self.neural_net[0])
        else:
            displacement_x = displacement_x_RBMR(self.neural_net[0], self.neural_net[1])
        ux = displacement_x(X)
        if self.method == 'standard PINN':
            displacement_y = displacement_y_PINN(self.neural_net[1])
        else:
            displacement_y = displacement_y_RBMR(self.neural_net[0], self.neural_net[1])
        uy = displacement_y(X)
        u = np.concatenate((ux, uy), axis=1)
        return u

    def predict_strain(self, X):
        first_derivatives_ux = first_derivatives(self.neural_net[0])
        dux_dx, dux_dy = first_derivatives_ux(X)
        first_derivatives_uy = first_derivatives(self.neural_net[1])
        duy_dx, duy_dy = first_derivatives_uy(X)
        return np.concatenate((dux_dx, duy_dy, dux_dy + duy_dx), axis=1)
