# ML-project
Repo for INT3305E Fall 2023 course's project in Predicting Movie genres from poster, title and user rating data.

# Project structure
```
├── dataset
    code to visualize and preprocessing data
├── model
    main models implementation
    ├── saved_models
        contains trained models
├── utils
    utility functions
main.py : main file to run the project
model_config.yaml : config file for models
```

# Installation
``` bash
pip install -r requirements.txt
```

# Usage
1. Arguments parsing

    To select the models to be used and other training hyperparameters, parsing arguments to main.py. Use
    ``` bash
    python main.py --help
    ```
    to print the list of arguments and their default values and descriptions.

2. Training

    To train the models, use
    ``` bash
    python main.py train --title_model <titleModelName> --poster_model <posterModelName> --urating <userRatingsModelName> --checkpoint <checkpointFileName>
    ```

    For example, for training the top score model, use
    ``` bash
    python main.py train --title_model LSTM --poster_model DenseNet169  --urating_model FNN --use_dropped_data False --batch_size 32  --image_size 256 --max_epochs 20 --checkpoint lstm_den169_fnn_nodrop
    ```

3. Testing

    To test the models, use
    ``` bash
    python main.py test --title_model <titleModelName> --poster_model <posterModelName> --urating <userRatingsModelName> --checkpoint <checkpointFileName>
    ```

    For example, for testing the top score model, use
    ``` bash
    python main.py test --title_model LSTM --poster_model DenseNet169  --urating_model FNN --use_dropped_data False --batch_size 32  --image_size 256 --max_epochs 20 --checkpoint lstm_den169_fnn_nodrop
    ```

#### `test` and `train` can be replaced with `train_test` for automatic testing after training

Notebook file for training and testing models: [ml-project.ipynb]