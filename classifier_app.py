# %%
import numpy as np
import pandas as pd
import rasterio
import joblib
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from rasterio.io import MemoryFile
import tempfile
import os

model_data = joblib.load(
    "landsat_random_forest.pkl"
)

rf = model_data["model"]
accuracy = model_data["accuracy"]

st.metric(
    label="Model Test Accuracy",
    value=f"{accuracy:.1%}"
)

#prompt user for the Landsat bands 1-7 as GeoTIFF files. The user can upload multiple files at once, and the accepted file types are .tif and .tiff. The uploaded files are stored in a list for further processing.
st.title("Landsat Vegetation Classifier")
st.success("STREAMLIT IS WORKING")
st.write("Upload Landsat Bands 1–7")
uploaded_files = st.file_uploader(
    "Upload Landsat Bands 1–7",
    type=["tif", "tiff"],
    accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    print(uploaded_file.name)
# %%


#prompt user for the Landsat bands 1-7 as GeoTIFF files. The user can upload multiple files at once, and the accepted file types are .tif and .tiff. The uploaded files are stored in a list for further processing.



#find the file paths of the Landsat bands in the directory. The bands are identified by their file names, which contain the band number (1-7) in the format "_SR_B[band_number].TIF". The file paths are stored in a list, and a dictionary is created to map each band number to its corresponding file path.
#file_paths = []
band_files = {}

for uploaded_file in uploaded_files:
    match = re.search(r"_SR_B(\d+)\.TIF$", uploaded_file.name, re.IGNORECASE)
    if match:
        band_number = int(match.group(1))
        band_files[band_number] = uploaded_file

#verify the bands were found and raise an error if any are missing. The required bands are 1-7, which correspond to the Landsat bands used for classification.

if uploaded_files:
    required_bands = set(range(1, 8))
    if set(band_files.keys()) != required_bands:
        missing = required_bands - set(band_files.keys())

        st.error(
            f"Missing Landsat bands: {sorted(missing)}"
        )
        st.stop()

#put bands in correct order and read them into a list. The bands are read using the rasterio library, which allows for reading and writing geospatial raster data. Each band is read as a 2D array and appended to the bands list.
    bands = []
    profile = None

    for band_number in range(1, 8):

        uploaded_file = band_files[band_number]

        with MemoryFile(uploaded_file.getvalue()) as memfile:

            with memfile.open() as src:

                bands.append(src.read(1))

                if band_number == 1:
                    profile = src.profile.copy()

# Stack into (rows, columns, 7 bands)
# Stack uploaded Landsat bands
    stack = np.stack(bands, axis=-1)

    st.write("Stack shape:", stack.shape)


# Get raster dimensions
    rows, cols, num_bands = stack.shape


# Reshape into:
# rows = individual pixels
# columns = Landsat bands B1-B7
    pixels = stack.reshape(-1, 7)

    st.write("Pixel matrix shape:", pixels.shape)


#Classify in chunks as it is a large image and may not fit in memory all at once. 

    chunk_size = 500_000

    prediction_chunks = []

    for start in range(0, len(pixels), chunk_size):

        end = start + chunk_size

        chunk = pixels[start:end]

        chunk_pred = rf.predict(chunk)

        prediction_chunks.append(chunk_pred)


    predictions = np.concatenate(
        prediction_chunks
    )



    classified = predictions.reshape(
        rows,
        cols
    )

    print("\nClassified raster shape:")
    print(classified.shape)

    print("\nFirst 5 x 5 classified pixels:")
    print(classified[:5, :5])

    print("\nClass distribution:")
    print(
        np.unique(
            classified,
            return_counts=True
        )
    )


#Write the classified raster to a new GeoTIFF file

    profile.update(
        dtype=rasterio.uint8,
        count=1
    )

#write the classified raster to a new GeoTIFF file in a temporary directory. The output file is named "vegetation_classification.tif" and is saved in the system's temporary directory. The metadata from the original raster is preserved in the output file.
    temp_dir = tempfile.gettempdir()

    output_path = os.path.join(
        temp_dir,
        "vegetation_classification.tif"
    )

    with rasterio.open(
        output_path,
        "w",
        **profile # this is where the metadata comes from the original raster
    ) as dst:

        dst.write(
            classified.astype(
                rasterio.uint8
            ),
            1 # we only need one band for the classified output
        )

    print("\nSaved:")
    print(output_path)
    print("\nDone!")

   


    with open(output_path, "rb") as output_file:

        st.download_button(
        label="Download Classified GeoTIFF",
        data=output_file,
        file_name="vegetation_classification.tif",
        mime="image/tiff"
    )