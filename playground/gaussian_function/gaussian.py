from data_generation.potentials import gaussian as mygauss
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
    figure_title = "MLP Prediction of Gaussian Potential with Noise"
    x_label = "Distance (r)"
    y_label = "Potential Energy (V)"
    markers = ['o', 'x']
    colors = ['blue', 'red']
    save_path = "gaussian_mlp_prediction.png"
    noise_amplitude = 0.01


#   generating data
    x_measured = abscissa_gen(0.7, 3.0, 0.1)
    y_measured = mygauss(x_measured)

#   adding noise
#    noises_1 = noise_gaussian(x_measured, seed=0, amplitude=noise_amplitude, mean=0.0, standard_deviation=0.5)
#    noises_2 = noise_decay(x_measured, initial_state=0.5, decay_rate=0.5)
#    noises_3 = noise_sine(x_measured, amplitude=noise_amplitude, frequency=1.0, initial_state=0.0, bias=0.0)
#    noises_total = [ n1 + n2*n3 for n1, n2, n3 in zip(noises_1, noises_2, noises_3) ]
#    y_noisy = np.array([ e + n for e, n in zip(y_measured, noises_total) ])

#   getting partial data for training
    x_to_learn = use_partial(x_measured, ratio_to_discard)
#    y_to_learn = use_partial(y_noisy, ratio_to_discard)
    y_to_learn = use_partial(y_measured, ratio_to_discard)

#   training MLP
    n_loop = 1
    for i in range(n_loop):
        print(f"Loop {i+1}/{n_loop}")
        my_model = quick_mlp(x_to_learn,
                            y_to_learn,
                            model_path="model.keras",
                            learning_rate=0.01,
                            hidden_layers=[ 128, 128 ],
                            activation=[ 'relu', 'relu', 'linear' ],
                            epochs=100)

    #   predicting with MLP    
        y_predicted = my_model.predict(x_to_learn)

    #   making a figure to compare the original data, the noisy data and the MLP prediction
    #    xs = [x_measured, x_measured]
    #    ys = [y_noisy, y_predicted]
        xs = [x_to_learn, x_to_learn]
        ys = [y_to_learn, y_predicted]
        labels = ["Original Data", "MLP Prediction"]

        plot2d(xs, 
            ys, 
            labels, 
            figure_title, 
            x_label, 
            y_label, 
            markers=markers, 
            colors=colors, 
            save_path=f"learn_Gaussian_{i+1}.png")


if __name__ == "__main__":
    main()
