# Multi-task Deep Network for Ophthalmology Screening on Fundus Images

*Authors: Lijing Song, Bozhao Liu, Stanford CS230 teaching staff*

This is our Stanford CS230 final project developed on top of the [sample code in PyTorch](https://github.com/cs230-stanford/cs230-code-examples) given by the teaching staff.

## Requirements

We use python3, PyTorch (source activate pytorch_p36 on AWS EC2), and a virtual env.

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given an image of a eye fundus representing healthy, diabetic ophthalmical disease, and/or glaucoma, predict the correct label.

## Our dataset

The dataset containing photos of fundus images is collected by ourselves from Rjukan Synssenter Optometri and hosted on box, download it [here][GlaucomaVSDiabetes].

Here is the structure of the data:
```
GlaucomaVSDiabetes/
    diabetes/
        GlaucomaVSDiabetes_1_0 (1).jpg
        GlaucomaVSDiabetes_1_0 (2).jpg
        ...
    glaucoma/
        GlaucomaVSDiabetes_0_1 (1).jpg
        GlaucomaVSDiabetes_0_1 (2).jpg
        ...
    healthy/
        GlaucomaVSDiabetes_0_0 (1).jpg
        GlaucomaVSDiabetes_0_0 (2).jpg
        ...
```

The images are named following `GlaucomaVSDiabetes_{label} ({id}).jpg` where the label is in `[0,0], [0,1], [1,0], [1,1]`.
The diabetes set contains 598 images and the glaucoma set contains 339 images.

Once the download is complete, move the dataset into `data/GlaucomaVSDiabetes`.
Run the script `build_dataset.py` which will resize the images to size `(177, 128)`. The new resized dataset will be located by default in `data/ResizedData`:

```bash
python build_dataset.py --data_dir data/GlaucomaVSDiabetes --output_dir data/ResizedData
```

## Quickstart (~10 min)

1. __Build the dataset of size 177x128__: make sure you complete this step before training
```bash
python build_dataset.py --data_dir data/GlaucomaVSDiabetes --output_dir data/ResizedData
```

2. __First experiment__ We started with a baseline model using a `base_model` directory under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
For every new experiment, we create a new directory under `experiments` with a similar `params.json` file, for instance, greyscale.

3. __Train__ experiment. Simply run
```
python train.py --data_dir data/ResizedData --model_dir experiments/base_model
```
or
```
python Train_Net.py --data_dir data/ResizedData --network (networkname)
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set.

4. __Hyperparameters search__ There is a new directory `learning_rate` in `experiments`. Now, run
```
python search_hyperparams.py --data_dir data/ResizedData --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

5. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

6. __Evaluation on the test set__ Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py --data_dir data/ResizedData --model_dir experiments/base_model
```


## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the training loop steps:
- loading the hyperparameters for the experiment (the `params.json`)
- loading the training and validation data
- creating the model, loss_fn and metrics
- training the model for a given number of epochs by calling `train_and_evaluate(...)`

You can then have a look at `data_loader.py` to understand:
- how jpg images are loaded and transformed to torch Tensors
- how the `data_iterator` creates a batch of data and labels and pads sentences

Once you get the high-level idea, depending on your dataset, you might want to modify
- `model/net.py` to change the neural network, loss function and metrics
- `model/data_loader.py` to suit the data loader to your specific needs
- `train.py` for changing the optimizer
- `train.py` and `evaluate.py` for some changes in the model or input require changes here

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)

[GlaucomaVSDiabetes]: https://stanford.box.com/s/36h81ro7i87zxcafqbbphxksc1pdduj3
