import tensorflow as tf
from src.pinn_collocation import pinn_collocation
from src.ref_solution import ref_solution
from src.postprocess import postprocess
from src.visualizer import viz_2D_mesh

def run_simulation_instance(method, Xp, Xp_errNorm, Vp_errNorm, npx, npy, young, nu, hidden_layers, nEpochs,
                            initial_learning_rate, decay_steps, decay_rate):

    print('-----------------------------------------------------')
    print('|            Welcome to PySolidSimPINN              |')
    print('|  Python-based Solid Mechanics Simulator via PINN  |')
    print('-----------------------------------------------------')

    refSol = ref_solution(young, nu)

    post = postprocess(Xp, Xp_errNorm, Vp_errNorm, refSol)
    pinn = pinn_collocation(method, post, Xp, Xp_errNorm, npx, npy, young, nu, hidden_layers, nEpochs, refSol)
    viz_2D_mesh(Xp)

    pinn.construct()
    pinn.train_data_loader()

    """
      Train the PINN
    """
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False)
    opt = tf.keras.optimizers.Adam(learning_rate=decayed_lr)
    Res = []
    for epoch in range(1, nEpochs + 1):
        with tf.GradientTape() as tape:
            res = pinn.Loss_tf(epoch)
            Res.append(res.numpy())
            grad_w = tape.gradient(res, pinn.model.trainable_variables)
            opt.apply_gradients(zip(grad_w, pinn.model.trainable_variables))

    outputs = pinn.predict_outputs(Xp)

    post.postprocess(Res, outputs)
