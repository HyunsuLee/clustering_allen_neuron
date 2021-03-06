# Clustering neuron from allen db

* for 2nd paper using allen db, clustering neurons with MLs, collaborating incheol.
* [our paper](https://mail.google.com/mail/u/0/#inbox/FMfcgxwDqfMPDTDDBtlxZlQmdWVtcbSJ) was published at Brain Research Bulletin 
* Caution!! These codes are no longer maintained and very dirty. Be careful.

## Data pipeline

1. preparing raw data from allen brain institute using AllenSDK [notebook](./Data_prep/allen_data_download_2018FEB.ipynb)
    1. preparing new raw data for revising manuscript. [notebook](./Data_prep/revising_data.ipynb)
1. Visualizing raw data using density plot [notebook](./Data_prep/allen_data_visualize.ipynb), just verification
1. Dividing data into train, test set in [rmd](./Data_prep/dividing_data.Rmd)
    1. for revising manuscript [rmd](./Data_pre/dividing_revised_data.Rmd)

## LASSO, RF done by incheol

1. For binary classification in [rmd](./lasso_rf/binary_model.Rmd), excitatory line classification in [rmd](./lasso_rf/eline_model.Rmd), and inhibitory line classification in [rmd](./lasso_rf/iline_model.Rmd).
1. models saved in ./lasso_rf/R_models/
1. reload models test in [rmd](./lasso_rf/reload_model_test.Rmd)

## ANN learning pipeline

1. data processing for tensorflow learning from R data(incheol) [notebook](./Data_prep/data_processing_for_ANN_180227.ipynb)
    1. one-hot coding
    1. minmax scaling
1. For ANN, view in the folder named ANN
1. coarse searching hyperparameter(learning rate and L2 beta) = ./ANN/NO_1_output_input_coarse_searching.py
1. fine searching = ./ANN/NO_2_output_input_fine_searching.py
1. top 10 model tensorboard logging and model saving = ./ANN/NO_3_output_input_logging.py
1. selection top model by inspecting tensorboard log (./ANN/logs/output_input/)
1. top model restore and choosing best epoch step, saving results = ./ANN/NO_4_output_input_restore.py
1. all of the final results in ./ANN/results/

