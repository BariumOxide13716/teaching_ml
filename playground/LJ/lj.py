from data_generation.potentials import lennardjones as lj
from data_generation.noise import noise_gaussian, noise_decay, noise_sine
from plotting.plot2D import plot2d
from model.mlp import MLP, quick_mlp
from data_generation.abscissa_gen import abscissa_gen, use_partial
import numpy as np

"""
This is a integrated test for MLP to learn the 
Lennard-Jones potential with noise.
The MLP is trained on the noisy data and then used to predict the potential on a test set.
"""

def main():

    ratio_to_discard = 0.1
    figure_title = "MLP Prediction of Lennard-Jones Potential with Noise"
    x_label = "Distance (r)"
    y_label = "Potential Energy (V)"
    markers = ['o', 'x']
    colors = ['blue', 'red']
    save_path = "lj_mlp_prediction.png"


#   generating data
    x_measured = abscissa_gen(0.5, 3.0, 0.1)
    y_measured = lj(x_measured)

#   adding noise
    noises_1 = noise_gaussian(x_measured, seed=0, amplitude=0.5, mean=0.0, standard_deviation=0.5)
    noises_2 = noise_decay(x_measured, initial_state=0.5, decay_rate=0.5)
    noises_3 = noise_sine(x_measured, amplitude=0.5, frequency=1.0, initial_state=0.0, bias=0.0)
    noises_total = [ n1 + n2*n3 for n1, n2, n3 in zip(noises_1, noises_2, noises_3) ]
    y_noisy = np.array([ e + n for e, n in zip(y_measured, noises_total) ])

#   getting partial data for training
    x_train = use_partial(x_measured, ratio_to_discard)
    y_noisy_train = use_partial(y_noisy, ratio_to_discard)

#   training MLP
    my_model = quick_mlp(x_train,
                        y_noisy_train,
                        hidden_layers=[ 10, 10 ],
                        activation=[ 'relu', 'relu', 'linear' ],
                        learning_rate=0.01,
                        test_size=0.2,
                        random_seed=42,
                        epochs=100,)

#   predicting with MLP    
    y_predicted = my_model.predict([[d] for d in x_measured])

#   making a figure to compare the original data, the noisy data and the MLP prediction
    xs = [x_measured, x_measured]
    ys = [y_noisy, y_predicted.flatten().tolist()]
    labels = ["Original Data", "MLP Prediction"]

    plot2d(xs, 
           ys, 
           labels, 
           figure_title, 
           x_label, 
           y_label, 
           markers=markers, 
           colors=colors, 
           save_path=save_path)

if __name__ == "__main__":
    main()