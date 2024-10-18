# Pytorch-Lightning Skeleton
This repository contains a skeleton project to train Pytorch models using [Lightning](https://lightning.ai/docs/pytorch/stable/).

This skeleton defaults to using GPU if available and CPU if not.
It also has default monitors using Tensorboard.

## How to use
1. Implement your model in `models.MyModel`
   - Modify `models.__init__().get_model()` so it initializes your model as your definition.

2. Add configuration to `LightningModel.LightningWrapper` to match your training.

3. Implement you dataset in `data.MyDataset`
   - Modify `data.__init__().get_dataloader()` so it initializes your model as your definition.


4. Modify `main.py` to include:
	1. New CLI parameters.
	2. Any parameters in `pipeline()`.


## Run

First create a virtual environment for your project starting with the provided `requirements.txt` which contains all dependencies from the skeleton project.

```
python -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You only need to run this once.

**If you added new dependencies like `torchvision` you will need to install them**.

Run the project with:

```
python main.py <your_args>
```

## Viewing logs
You can view the default logs using `tensorboard`.

Open a new shell and run (if you used default arguments):
```
source venv/bin/activate
tensorboard --logdir ./logs
```
