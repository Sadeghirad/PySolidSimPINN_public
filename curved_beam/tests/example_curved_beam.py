import json
import pickle
from src.pySolidSimPINN import run_simulation_instance


input_file = 'input_curved_beam_FFNN2_SP1_RBMR.json'

if __name__ == "__main__":

    with open('../in/'+input_file, 'r') as file:
        data = json.load(file)

    method = data['method']
    sampling_points = data['sampling_points']
    points_for_error_calculation = data['points_for_error_calculation']
    young = data['young']
    nu = data['nu']
    hidden_layers = data['hidden_layers']
    initial_learning_rate = data['initial_learning_rate']
    decay_steps = data['decay_steps']
    decay_rate = data['decay_rate']
    nEpochs = data['nEpochs']
    npx = data['npx']
    npy = data['npy']

    with open(sampling_points, 'rb') as file:
        Xp = pickle.load(file)
    with open(points_for_error_calculation, 'rb') as file:
        Xp_errNorm, Vp_errNorm = pickle.load(file)
    run_simulation_instance(method, Xp, Xp_errNorm, Vp_errNorm, npx, npy, young, nu, hidden_layers, nEpochs,
                            initial_learning_rate, decay_steps, decay_rate)
