# Clustering neuron from allen db

for 2nd paper using allen db, clustering neurons with MLs, collaborating incheol.

## Data pipeline

1. preparing raw data from allen brain institute using AllenSDK [notebook](./allen_data_analysis.ipynb)
1. Visualizing raw data using density plot [notebook](./allen_data_analysis.ipynb)
1. data processing for tensorflow learning from R data(incheol) [notebook](./data_processing_180227.ipynb)
    1. one-hot coding
    1. minmax scaling

## ANN learning pipeline

1. coarse searching hyperparameter(learning rate and L2 beta) = NO_1_output_input_coarse_searching.py
1. fine searching = NO_2_output_input_fine_searching.py
1. top 10 model tensorboard logging and model saving = NO_3_output_input_logging.py
1. selection top model by inspecting tensorboard log (./logs/output_input/)
1. top model restore and choosing best epoch step, saving results = NO_4_output_input_restore.py
1. all of the final results in ./results/