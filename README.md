## Setting up Conda for U-Net:
All of the following commands are run in Anaconda Prompt

#### Step 1 (Optional)
First install a faster Conda solver to speed up the environment setup. In Anaconda Prompt make sure you are on (base) environment and run
```commandline
conda install -n base conda-libmamba-solver
```
Then run
```commandline
conda config --set solver libmamba
```
This step basically make Conda faster and is recommended by the Anaconda developers.

#### Step 2
Create a new Conda environment. I choose to call it "seg" but you can call it whatever:
```commandline
conda create -n seg python=3.9
```
Next activate the new environment
```commandline
conda activate seg
```
#### Step 3
Install the following packages one-by-one. To do that, copy one of the following code, run it in Anaconda Prompt, confirm the installation, then continue to the next line of code when the installation finishes.
```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-lightning -c conda-forge
conda install -c conda-forge monai
conda install matplotlib
conda install scipy=1.11.1
conda install pandas
conda install scikit-image=0.19.3
pip install "jsonargparse[signatures]"
pip install nibabel
pip install wandb
```
**Toubleshooting** If you have issues, you can try installing the specific package versions that I used to develop this code. The package versions are listed in `requirements.txt`. To install a specific package version you should replace a command like `conda install matplotlib` with `conda install matplotlib=3.7.1`.

## Preparing Files for U-Net Prediction
The U-Net can run on a batch of scans. The scans are provided to the U-Net in the form of a CSV file with the following format:
```
img
<path to scan 1>
<path to scan 2>
<path to scan 3>
...
<path to scan K>
```
In short, the first line of the CSV must be "img" and then each scan path goes on a new line. This CSV will be given to the U-Net, and it will segment every scan one-by-one in order. The current U-Net will also localize landmarks, but later U-Nets might not do this.

## Running the U-Net
Open Anaconda Prompt and navigate to this segmentation folder.
While in the new Segmentation folder and with the Conda environment activated, run this line:
```commandline
python main.py predict -c cfg\c5_prediction.yaml --data.data_csvs.pred "path/to/csv/file/that/you/made"
```
This version should take about 5 minutes per scan, but can take longer (e.g., 15 minutes), if the scan is large.

You can also try running a faster, but more GPU-memory-demanding version by running:
```commandline
python main.py predict -c cfg\c5_prediction.yaml --data.data_csvs.pred "/path/to/csv/file/that/you/made" --data.low_memory_predict false --model.sw_batch_size <INTEGER>
```

## U-Net Output
An explanation for the U-Net output will go here.
<!--- TODO -->

## More Information
There are many ways to customize the U-Net's behavior that are not touched on in this introductory README. I encourage you to checkout the `cfg/pred_config_options.yaml` file which lists the parameter that are relevant for predicting with the U-Net. You can change or add parameters in the `cfg/c5_prediction.yaml` file for changes to take effect. Checkout PyTorch-Lightning's command line interface (CLI) interface for more information on how the command line and configuration parameters are passed to the code. Feel free to contact bakertim@umich.edu with questions.

[Lightning CLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli.html#lightning-cli)