# Convolutional Neural Network (CNN) for Estimating Tropical Cyclone Wind Intensity Through Infrared Satellite Imagery
This repo contains the code to download and prepare data to train and validate a convolutional neural network. The neural network takes a satellite image of a tropical cyclone as an input and output an estimate for the tropical cyclone’s maximum sustained wind speed.
## Project Background and Motivation
Since hurricanes are typically located over large bodies of water where weather stations are sparse, meteorologists often have to estimate the wind speed of hurricanes. They usually use buoy observations, microwave satellite imagery, and infrared satellite imagery to make these estimates.

There is growing interest in applying AI and machine learning techniques to improve the accuracy of operational meteorological tasks, including estimating hurricane wind speed. I began looking into applying deep learning to hurricane wind speed estimation during the COVID-19 pandemic when the American Meteorological Society (AMS) made their journal articles publicly available at no cost. <a href="https://journals.ametsoc.org/mwr/article/147/6/2261/344590/Using-Deep-Learning-to-Estimate-Tropical-Cyclone">Wimmers et al. 2019</a> and <a href="https://journals.ametsoc.org/waf/article/34/2/447/291/Estimating-Tropical-Cyclone-Intensity-by-Satellite">Chen et al. 2019</a> both applied deep learning to hurricane wind speed estimation, achieving considerable accuracy. This piqued my interest, so I decided to take a stab at it.
## Data Sources
I used images of hurricanes from the <a href="https://www.ncdc.noaa.gov/hursat/">HURSAT data project</a> run by the National Centers for Environmental Information. This database contains satellite images of hurricanes in NetCDF file format. The best part about this database: the center of each hurricane was in the middle of each image.

I also used best track data from the <a href="https://www.nhc.noaa.gov/data/#hurdat">HURDAT2 database</a> provided by the National Hurricane Center. It contains records of all known hurricanes in the Atlantic and Pacific basins, as well as their wind speeds at 6-hour intervals.
## Overview of Files
`environment.yml`: Lists out all libraries (and their dependencies) neccessary to run the Python files. Anaconda reads this file to create an environment in which the Python files can be interpreted and executed.

`besttrack.csv`: Contains the data cleaned from the National Hurricane Center's HURDAT2 database found at this <a href="https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2019-052520.txt">link</a>. `download.py` and `assemble.py` rely on this data to function properly.

`download.py`: Downloads the satellite images of hurricanes from the HURSAT database as NetCDF files.

`assemble.py`: Labels satellite images of hurricanes with their wind speed from the best track dataset. Saves these images and labels as NumPy array files.

`model.py`: Separates data into folds for k-fold validation. Augments training data. Builds, trains, and validates the neural network on each fold. Prints out information and saves two graphs about the model’s accuracy on validation data.

`view_images.py` (Optional): Does not play a pivotal role in running the model, but may be of interest to the curious developer. Shows 10 random images from the set of satellite images in `images.npy`.
## Highlights of Methodology
<b>Optimized Data Downloading</b>: `download.py` does not download files that cannot be used in the neural network. The HURSAT database has images of hurricanes from around the world, but the best track data from HURDAT2 only contains wind speeds for hurricanes from the Atlantic and Pacific Oceans. So, before downloading a hurricane’s satellite imagery from HURSAT, the code checks to see whether the best track data has records for that hurricane. If best track has no data for that hurricane, the satellite image is not downloaded. This conserves local storage space and cuts down on execution time.

<b>“Cropping” Satellite Images</b>: The most valuable information about a hurricane’s intensity is near the center. So, `assemble.py` crops satellite images to remove the outer part of the hurricane from the image. After reading the satellite image from its NetCDF file, turning it into a NumPy array, the code crops the image so that only a 50-by-50-pixel square at the center is remaining. Removing this data reduces the amount of time spent on data augmentation and model training in `model.py`.

<b>Merging HURSAT and HURDAT2 to Match Satellite Images with their Wind Speed</b>: Each satellite image file provides us with the name of the hurricane, as well as the time and date of the satellite image. However, it does not provide us with the wind speed of the hurricane at that time. `assemble.py` finds the wind speed for each satellite image by searching for the hurricane’s name, date, and time in the best track dataset. Once the wind speed is retrieved, both the satellite image and wind speed are appended to NumPy arrays in unison. This effectively labels the satellite image with its wind speed, since can the satellite image and wind speed will be retrieved in unison in `model.py`.

<b>Augmenting Images using Keras</b>: An analysis of the data before training the model shows that weak tropical cyclones (tropical depressions and tropical storms) significantly outnumber strong tropical cyclones (hurricanes) in the dataset. This is not surprising since tropical depressions and tropical storms are much more common than hurricanes. However, this discrepancy means that the dataset is unbalanced and causes the neural network to perform poorly on hurricanes. To balance the dataset, `model.py` uses <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">Keras’s ImageDataGenerator</a> to augment data from existing hurricanes in the dataset. Including this augmented data in model training significantly improves model performance, especially in estimating the wind speeds for hurricanes.
## Results
This table shows one example of the neural network’s root-mean-square error (RMSE) for each fold in k-fold validation, using hurricanes from 2007 to 2016.

<table>
  <tr>
    <td>Validation Fold</td>
    <td>Fold 1</td>
    <td>Fold 2</td>
    <td>Fold 3</td>
    <td>Fold 4</td>
    <td>Fold 5</td>
    <td>All Folds</td>
  <tr>
  <tr>
    <td>RMSE</td>
    <td>16.9 knots</td>
    <td>12.5 knots</td>
    <td>11.5 knots</td>
    <td>16.6 knots</td>
    <td>10.9 knots</td>
    <td>13.9 knots</td>
  <tr>
</table>
<a href="https://journals.ametsoc.org/mwr/article/147/6/2261/344590/Using-Deep-Learning-to-Estimate-Tropical-Cyclone">Wimmers et al. 2019</a> achieved an RMSE of 14.3 knots when using the same HURDAT2 best track dataset to test their model on hurricanes that occurred in 2007 and 2012. Please note, my table shows <i>validation</i> results, and Wimmers provides <i>test</i> results.

## Install and Run the Project
Following these steps will allow you to run `model.py`, which performs k-fold validation on the model using the downloaded data. These instructions use Anaconda for package management and deployment.
1.	Create a directory to store the contents of this project
2.	Download `environment.yml` to this directory
3.	Navigate to this directory in Anaconda Prompt and run `conda env create --file environment.yml`
4.	Download `download.py`, `assemble.py`, `model.py`, and `besttrack.csv` to this directory. When you run these files in steps 5, 6, and 7, make sure you are running them in the conda environment you created in step 3.
5.	Run `download.py`, which will create a directory called `Satellite Imagery` where the satellite image files will be downloaded. <i>Warning</i>: one year of hurricane satellite images is about 500 MB. Multiple hurricane seasons can take up a GB or more of local storage.
6.	Run `assemble.py`, which will create `images.npy` and `labels.npy` containing data prepared for training and validating the neural network.
7.	Run `model.py`, which will print information to the console and save two seaborn graphs to the directory, providing information about the model’s accuracy.

Note: Running this project from start to finish may take several hours.
## Contact the Developer
This project was created during summer 2020 by <a href="https://www.linkedin.com/in/connor-cozad/">Connor Cozad</a>, an undergraduate student in data science and meteorology at the College of Charleston. Feel free to reach out to me by email at 23ccozad@gmail.com.
