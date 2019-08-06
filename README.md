# agar-pad-processing -- APP
Scripts written to analyze images obtained from agar pad imaging.

## Getting started
A docker image containing the required Python libraries is available in `dockerfile/`. The directions below explain how to get a working docker image that can run the APP code.

### Prerequisites

You will need to install [Docker](https://www.docker.com). See [the documentation](https://docs.docker.com).

### Installing

The first step is to create a docker image that contains all required compilers and applications.

```
cd dockerfile/
sudo docker build -t root/app:18.04 root/.
```

Optionally you can create an additional image with user name and ID matching your local machine. Your `/home/user/` directory will be mounted at the same location within your container so that you can easily access your file system. First edit the file `dockerfile/user/Dockerfile` and replace the user name, user ID, group name and group ID to match those on your local machine. Then:

```
sudo docker build -t user/app:18.04 user/.
```

In order to compile and run code of this project, you can then simply create a docker container and run the commands from inside. First create an alias:
```
alias docker-app='docker run --rm -it -w=$PWD -v $HOME:$HOME -h app user/app:18.04'
```

You might also consider adding directly this line to your `~/.bash_aliases` or `~/.bashrc` files. Feel also free to mount any other useful location through the option `-v` (see [doc](https://docs.docker.com/storage/bind-mounts/)). Then execute the alias to run a new container from your image in a terminal:
```
docker-app
```

## Image processing

The starting point of the workflow is a a Nikon ND2 file. Let us call it `agarpad_images.nd2`. This repository contains several scripts that should be run sequencially. The typical command you will run is:
```
python script.py -f param_file.yml other arguments
```

The file `param_file.yml` contains configuration options for the script. A template parameter file is available in `roles/`. We start from a directory containing the following:
```
.
|-- agarpad_images.nd2
|-- code -> ../code
`-- roles
    `-- params.yml
```

The directory `code/` should point to the root of the APP repository. Note that help on the arguments of a particular script can always be obtained using the `-h` option:
```
python script.py -h
```

### Converting ND2 file to TIFFs
We start by converting the ND2 file to TIFFs file:
```
python code/image_processing/process_nd2.py -f roles/params.yml -d . agarpad_images.nd2
```

With the template parameter file provided, we obtained:
```
.
|-- TIFFS
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- metadata.txt
|   `-- process_nd2.yml
|-- agarpad_images.nd2
|-- code -> ../code
`-- roles
    `-- params.yml
```

### Pre-processing the TIFFs
This step is optional. As is standard, raw images generally need some pre-processing before proceeding to the actual image analysis (eg segmentation). Concretely, this script will apply a Gaussian blur filter, as well as morphological operations (closing/opening). Optionally, it can also subtract the background, and invert the signal (namely transform light background images into dark background images, typically to make phase contrast similar to fluorescence channels). The background is computed by applying a sliding window of fixed size across the image and computing the median in that window.

A typical command will be:
```
python code/image_processing/preprocess_images.py -f roles/params.yml -d TIFFS_preprocessed/ --debug -i TIFFS/agarpad_images_f0*.tif
```

We obtained:
```
.
|-- TIFFS
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- metadata.txt
|   `-- process_nd2.yml
|-- TIFFS_preprocessed
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- debug
|   |   |-- agarpad_images_f00.png
|   |   |-- agarpad_images_f02.png
|   |   `-- agarpad_images_f09.png
|   |-- metadata.txt
|   `-- preprocess_images.yml
|-- agarpad_images.nd2
|-- code -> ../code
|-- metadata.txt
`-- roles
    `-- params.yml
```

Note that the `--debug` argument is optional: it creates a `debug/` folder in the output directory for visual inspection of the steps involved in the preprocessing.

### Segmentation
Now comes the actual segmentation:
```
python code/image_processing/segmentation_cells.py -f roles/params.yml -d . --debug -i TIFFS_preprocessed/agarpad_images_f0*.tif
```

We obtained:
```
.
|-- TIFFS
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- metadata.txt
|   `-- process_nd2.yml
|-- TIFFS_preprocessed
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- debug
|   |   |-- agarpad_images_f00.png
|   |   |-- agarpad_images_f02.png
|   |   `-- agarpad_images_f09.png
|   |-- metadata.txt
|   `-- preprocess_images.yml
|-- agarpad_images.nd2
|-- cells
|   `-- segmentation
|       |-- estimators
|       |   |-- agarpad_images_f00.npz
|       |   |-- agarpad_images_f02.npz
|       |   |-- agarpad_images_f09.npz
|       |   `-- debug
|       |       |-- agarpad_images_f00_estimator_debug.png
|       |       |-- agarpad_images_f02_estimator_debug.png
|       |       `-- agarpad_images_f09_estimator_debug.png
|       |-- index_tiffs.txt
|       |-- labels
|       |   |-- agarpad_images_f00.npz
|       |   |-- agarpad_images_f02.npz
|       |   |-- agarpad_images_f09.npz
|       |   `-- debug
|       |       |-- agarpad_images_f00_labels_debug.png
|       |       |-- agarpad_images_f02_labels_debug.png
|       |       `-- agarpad_images_f09_labels_debug.png
|       |-- masks
|       |   |-- agarpad_images_f00.npz
|       |   |-- agarpad_images_f02.npz
|       |   |-- agarpad_images_f09.npz
|       |   `-- debug
|       |       |-- agarpad_images_f00_mask_debug.png
|       |       |-- agarpad_images_f02_mask_debug.png
|       |       `-- agarpad_images_f09_mask_debug.png
|       `-- params.yml
|-- code -> ../code
|-- metadata.txt
`-- roles
    `-- params.yml
```

Several remarks:
* The result of the segmentation is available in the `cells/segmentation` directory.
* The `--debug` argument is optional: it allows the user to visualize and troubleshoot the segmentation for different FOVs. Plots are written in directories named `debug/` corresponding to the masks, labels and estimators creations.
* The segmentation is simply based on a threshold value. The channel from which the segmentation is performed must be passed in the parameter file.
* There is a `threshold` parameter in the parameter file. If no value is given, or if `null`, then a threshold is determined with the OTSU method for each FOV. If a value is passed, it must be a value between 0 and 1. For example, for a typical 16-bits images, the total range of value [0, 65535] is scaled down linearly to [0,1]. For that purpose, one can also use the script `utils.py` with the optional argument `--otsu` in order to compute the value of the OTSU threshold from several images at once. This might prove useful especially when some FOV are empty and the user wants to use the same value across all the FOVs of the experiment. Alternatively, the script might be run on a restricted number of FOVs (eg 10). The threshold value spitted out by the OTSU thresholding might be then used when re-running the script on the total number of FOVs.
* The masks are written in sparse matrix format (`.npz`). Those are binary matrices with same xy dimensions as the corresponding FOVs. Entries are set to 1 when a cell is detected and 0 otherwise.
* The estimators are computed from the masks. For each connected component of the mask (i.e. candidate cell), several criteria are computed according to the arguments passed in the `params.yml` file. To each cell is given a score that reflects how good it satisfies those criteria.
* The labels are computed from those cells that passed a minimum score, defined as the `mask_params->threshold` parameter in the parameter file.

### Construction of cell dictionary
The final step of the image analysis pipeline is to collect all those cells and build a dictionary comprising all their properties:
```
python code/image_processing/collection_cells.py -f roles/params.yml -d .
```

We obtained:
```
.
|-- TIFFS
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- metadata.txt
|   `-- process_nd2.yml
|-- TIFFS_preprocessed
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- debug
|   |   |-- agarpad_images_f00.png
|   |   |-- agarpad_images_f02.png
|   |   `-- agarpad_images_f09.png
|   |-- metadata.txt
|   `-- preprocess_images.yml
|-- agarpad_images.nd2
|-- cells
|   |-- collection
|   |   |-- collection.js
|   |   |-- masks
|   |   |   |-- f00y0022x1467.tif
|   |   |   |-- f00y0028x1602.tif
|   |   |   |-- f00y0062x1682.tif
|   |   |   `-- etc...
|   |   |-- params.yml
|   |   `-- tiffs
|   |   |   |-- f00y0022x1467.tif
|   |   |   |-- f00y0028x1602.tif
|   |   |   |-- f00y0062x1682.tif
|   |   |   `-- etc...
|   `-- segmentation
|       |-- estimators
|       |-- index_tiffs.txt
|       |-- labels
|       |-- masks
|       `-- params.yml
|-- code -> ../code
|-- metadata.txt
|-- preprocess_images.yml
`-- roles
    `-- params.yml
```

Several remarks:
* The dictionary is available in JSON format in `cells/collection/collection.js`.
* Cropped images corresponding to non-rotated bounding boxes of each cells are available in `masks/` and `tiffs/`. The masks are just binary images that define the cell object, whereas the tiffs are simply cropped images of the original FOVs. Note that the latter tiffs have the same number of channels as the original images.
* Saving one mask and one cropped tiff image for every cell might take a lot of disk space. It is possible to not write any of those by passing the `--lean` optional argument. In that case, the the cell dictionary in JSON format is the only output.

##  Data analysis
The script `analysis.py` allows one to visualize some information from the segmented cells. For example, dimensions distributions are useful. One may also first run the above analysis with a restrained number of FOVs, control attributes such as cell width, aspect ratio, box filling fraction, and adjust accordingly the segmentation parameters before running the segmentation on all FOVs and all channels.

Using the parameter file provided as template:
```
python code/analysis/analysis.py -f roles/params.yml -d . cells/collection/collection.js
```

We obtained:
```
.
|-- TIFFS
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- metadata.txt
|   `-- process_nd2.yml
|-- TIFFS_preprocessed
|   |-- agarpad_images_f00.tif
|   |-- agarpad_images_f02.tif
|   |-- agarpad_images_f09.tif
|   |-- debug
|   |   |-- agarpad_images_f00.png
|   |   |-- agarpad_images_f02.png
|   |   `-- agarpad_images_f09.png
|   |-- metadata.txt
|   `-- preprocess_images.yml
|-- agarpad_images.nd2
|-- analysis
|   |-- dimensions
|   |   |-- analysis_dimensions_other.pdf
|   |   |-- analysis_dimensions_other.png
|   |   |-- analysis_dimensions_other.svg
|   |   |-- analysis_dimensions_um.pdf
|   |   |-- analysis_dimensions_um.png
|   |   `-- analysis_dimensions_um.svg
|   `-- params.yml
|-- cells
|   |-- collection
|   |   |-- collection.js
|   |   |-- masks
|   |   |-- params.yml
|   |   `-- tiffs
|   `-- segmentation
|       |-- estimators
|       |-- index_tiffs.txt
|       |-- labels
|       |-- masks
|       `-- params.yml
|-- code -> ../code
|-- metadata.txt
|-- preprocess_images.yml
`-- roles
    `-- params.yml
```

The results of the analysis are written as figures in various formats in the `analysis/` folder.
