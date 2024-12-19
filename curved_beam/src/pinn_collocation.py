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
            temp = self.neural_net(tf.concat([x, y], axis=-1))
            u = temp * x * 0. + temp
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
                temp = self.neural_net(tf.concat([x, y], axis=-1))
                u = temp * x * 0. + temp
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
        self.point1 = np.array([4, 0])
        self.point2 = np.array([4, 0])
        self.point3 = np.array([5, 0])
        self.origin1 = tf.convert_to_tensor(self.point1, dtype='float32')
        self.origin2 = tf.convert_to_tensor(self.point2, dtype='float32')
        self.origin3 = tf.convert_to_tensor(self.point3, dtype='float32')
        self.origin1 = tf.expand_dims(self.origin1, axis=0)
        self.origin2 = tf.expand_dims(self.origin2, axis=0)
        self.origin3 = tf.expand_dims(self.origin3, axis=0)
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        u = self.neural_net_x(tf.concat([x, y], axis=-1))
        origins = tf.concat([self.origin1, self.origin3], axis=0)
        temp_u = self.neural_net_x(origins)
        temp1_u, temp3_u = temp_u[0], temp_u[1]
        origins = tf.concat([self.origin2, self.origin3], axis=0)
        temp_v = self.neural_net_y(origins)
        temp2_v, temp3_v = temp_v[0], temp_v[1]
        theta = (temp3_v - temp2_v) / (self.point3[0] - self.point2[0])
        u = u - temp1_u + (y - self.point1[1]) * theta
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
        self.point1 = np.array([4, 0])
        self.point2 = np.array([4, 0])
        self.point3 = np.array([5, 0])
        self.origin1 = tf.convert_to_tensor(self.point1, dtype='float32')
        self.origin2 = tf.convert_to_tensor(self.point2, dtype='float32')
        self.origin3 = tf.convert_to_tensor(self.point3, dtype='float32')
        self.origin1 = tf.expand_dims(self.origin1, axis=0)
        self.origin2 = tf.expand_dims(self.origin2, axis=0)
        self.origin3 = tf.expand_dims(self.origin3, axis=0)
        super().__init__(**kwargs)

    def call(self, X):
        x, y = (X[..., i, tf.newaxis] for i in range(X.shape[-1]))
        u = self.neural_net_y(tf.concat([x, y], axis=-1))
        origins = tf.concat([self.origin2, self.origin3], axis=0)
        temp_v = self.neural_net_y(origins)
        temp2_v, temp3_v = temp_v[0], temp_v[1]
        theta = (temp3_v - temp2_v) / (self.point3[0] - self.point2[0])
        u = u - temp2_v - (x - self.point2[0]) * theta
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
        X_bottom = tf.keras.layers.Input(shape=(2,))
        X_left = tf.keras.layers.Input(shape=(2,))
        X_out = tf.keras.layers.Input(shape=(2,))
        n_out_vec1 = tf.keras.layers.Input(shape=(1,))
        n_out_vec2 = tf.keras.layers.Input(shape=(1,))
        X_in = tf.keras.layers.Input(shape=(2,))
        n_in_vec1 = tf.keras.layers.Input(shape=(1,))
        n_in_vec2 = tf.keras.layers.Input(shape=(1,))
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
        dux_dx, dux_dy = first_derivatives_ux(X_left)
        first_derivatives_uy = first_derivatives(self.neural_net[1])
        duy_dx, duy_dy = first_derivatives_uy(X_left)
        sx_left = -(2 * mu + lambda_) * dux_dx - lambda_ * duy_dy
        sy_left = -mu * dux_dy - mu * duy_dx
        dux_dx, dux_dy = first_derivatives_ux(X_out)
        duy_dx, duy_dy = first_derivatives_uy(X_out)
        s11 = (2 * mu + lambda_) * dux_dx + lambda_ * duy_dy
        s22 = (2 * mu + lambda_) * duy_dy + lambda_ * dux_dx
        s12 = mu * dux_dy + mu * duy_dx
        s_out_1 = s11 * n_out_vec1 + s12 * n_out_vec2
        s_out_2 = s12 * n_out_vec1 + s22 * n_out_vec2
        dux_dx, dux_dy = first_derivatives_ux(X_in)
        duy_dx, duy_dy = first_derivatives_uy(X_in)
        s11 = (2 * mu + lambda_) * dux_dx + lambda_ * duy_dy
        s22 = (2 * mu + lambda_) * duy_dy + lambda_ * dux_dx
        s12 = mu * dux_dy + mu * duy_dx
        s_in_1 = s11 * n_in_vec1 + s12 * n_in_vec2
        s_in_2 = s12 * n_in_vec1 + s22 * n_in_vec2
        second_derivatives_ux = second_derivatives(self.neural_net[0])
        dux_dxx, dux_dxy, dux_dyy = second_derivatives_ux(X)
        second_derivatives_uy = second_derivatives(self.neural_net[1])
        duy_dxx, duy_dxy, duy_dyy = second_derivatives_uy(X)
        PDEx = (2 * mu + lambda_) * dux_dxx + mu * dux_dyy + (mu + lambda_) * duy_dxy
        PDEy = (mu + lambda_) * dux_dxy + (2 * mu + lambda_) * duy_dyy + mu * duy_dxx
        return tf.keras.models.Model(
            inputs=[X, X_bottom, X_left, X_out, n_out_vec1, n_out_vec2, X_in, n_in_vec1, n_in_vec2, X_dbcx, X_dbcy],
            outputs=[PDEx, PDEy, sx_left, sy_left, s_out_1, s_out_2, s_in_1, s_in_2, u_dbcx, u_dbcy])

    def train_data_loader(self):
        """
          X_train, y_train
        """
        numpar = (self.npx + 1) * (self.npy + 1)
        ndbc_x = np.arange(0, numpar, self.npy + 1)
        ndbc_y = np.arange(0, numpar, self.npy + 1)
        n_bot = np.arange(0, numpar, self.npy + 1)
        n_left = np.arange(self.npy, numpar, self.npy + 1)
        n_out = np.arange((self.npy + 1) * self.npx, numpar)
        Xp_lenth = np.sqrt(self.Xp[n_out, 0] * self.Xp[n_out, 0] + self.Xp[n_out, 1] * self.Xp[n_out, 1])
        n_out_vec1 = self.Xp[n_out, 0] / Xp_lenth
        n_out_vec2 = self.Xp[n_out, 1] / Xp_lenth
        n_in = np.arange(0, self.npy + 1)
        Xp_lenth = np.sqrt(self.Xp[n_in, 0] * self.Xp[n_in, 0] + self.Xp[n_in, 1] * self.Xp[n_in, 1])
        n_in_vec1 = -self.Xp[n_in, 0] / Xp_lenth
        n_in_vec2 = -self.Xp[n_in, 1] / Xp_lenth
        self.X_train = [self.Xp, self.Xp[n_bot, :], self.Xp[n_left, :], self.Xp[n_out, :],
                        np.expand_dims(n_out_vec1, axis=1), np.expand_dims(n_out_vec2, axis=1),
                        self.Xp[n_in, :], np.expand_dims(n_in_vec1, axis=1), np.expand_dims(n_in_vec2, axis=1),
                        self.Xp[ndbc_x, :], self.Xp[ndbc_y, :]]
        nbc_left = np.zeros((len(n_left), 2))
        nbc_out = np.zeros((len(n_out), 2))
        nbc_in = np.zeros((len(n_in), 2))
        dbc_x = np.zeros(0)
        dbc_y = np.zeros(self.npx + 1)
        for count, i in enumerate(n_left):
            dummy = self.refSol.exact_stress(self.Xp[i, :])
            nbc_left[count, 0] = -dummy[0]
            nbc_left[count, 1] = -dummy[2]
        for count, i in enumerate(n_out):
            dummy = self.refSol.exact_stress(self.Xp[i, :])
            nbc_out[count, 0] = dummy[0] * n_out_vec1[count] + dummy[2] * n_out_vec2[count]
            nbc_out[count, 1] = dummy[2] * n_out_vec1[count] + dummy[1] * n_out_vec2[count]
        for count, i in enumerate(n_in):
            dummy = self.refSol.exact_stress(self.Xp[i, :])
            nbc_in[count, 0] = dummy[0] * n_in_vec1[count] + dummy[2] * n_in_vec2[count]
            nbc_in[count, 1] = dummy[2] * n_in_vec1[count] + dummy[1] * n_in_vec2[count]
        for count, i in enumerate(n_bot):  # hack for DBC
            dummy = self.refSol.exact_disp(self.Xp[i, :])
            dbc_x = np.append(dbc_x, dummy[0])
        self.y_train = [np.expand_dims(nbc_left[:, 0], axis=1), np.expand_dims(nbc_left[:, 1], axis=1),
                        np.expand_dims(nbc_out[:, 0], axis=1), np.expand_dims(nbc_out[:, 1], axis=1),
                        np.expand_dims(nbc_in[:, 0], axis=1), np.expand_dims(nbc_in[:, 1], axis=1),
                        np.expand_dims(dbc_x, axis=1), np.expand_dims(dbc_y, axis=1)]
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
            if idx == 3:
                xxx[4] = x[4][tf.newaxis, i, ...]
                xxx[5] = x[5][tf.newaxis, i, ...]
            if idx == 6:
                xxx[7] = x[7][tf.newaxis, i, ...]
                xxx[8] = x[8][tf.newaxis, i, ...]

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
            elif idx == 2:
                Grad = grad.gradient(y[2], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
                Grad = grad.gradient(y[3], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
            elif idx == 3:
                Grad = grad.gradient(y[4], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
                Grad = grad.gradient(y[5], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
            elif idx == 6:
                Grad = grad.gradient(y[6], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
                Grad = grad.gradient(y[7], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient2 += 2. * l2_loss(j)
            elif idx == 9:
                Grad = grad.gradient(y[8], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient3 += 2. * l2_loss(j)
            elif idx == 10:
                Grad = grad.gradient(y[9], self.model.trainable_variables)
                for j in Grad:
                    if j is not None:
                        ntk_coefficient3 += 2. * l2_loss(j)

            return ntk_coefficient1, ntk_coefficient2, ntk_coefficient3

        ntk_coefficient1, ntk_coefficient2, ntk_coefficient3 = tf.constant(0.), tf.constant(0.), tf.constant(0.)

        for idx in range(11):
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
