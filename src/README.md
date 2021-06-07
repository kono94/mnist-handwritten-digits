## Structure
### Folders

- `/data` loads in the datasets and makes tranformation to eventually export train and test sets that are ready for learning

- `/model` definitions of different model architecture (fully connected MLPs and CNNs)

- `/util` includes functions to display images of the dataset or print parameters of the model

### Files
 - `main.py` invokes training based upon parsed input arguments 
 - `apply.py` loads in a trained model to make a prediction

- `predefined.py` instead of setting input arguments, models and training parameters are all defined in this file

- `train.py` wraps training and test loop inside a custom `Trainer` to encapsulate learning routine

