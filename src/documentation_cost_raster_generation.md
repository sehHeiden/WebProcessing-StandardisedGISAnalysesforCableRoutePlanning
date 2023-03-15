# Cost Raster Generation
Cost raster can be generated from several vector data files (that can be processed by fiona).

For this purpose the `rules2weights.py`can be applied on a `.ods` file, that define the processing rules.

# Rules definition with a `.ods`-file
The rules are described in the Sheet **ProcessingRules**.

| Column      | DataType                   | Description                                                                                                              |
|-------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Directory   | string                     | Path to the file folder.                                                                                                 |
| FileName    | string                     | name of the file. To construct full path from Path an file name.                                                         |
| Description | string                     | Name given to the processing rule. One rule has to be called *base*.                                                     |
| Layer       | optional string            | Layer name for (if file has Layer).                                                                                      |
| Column      | optional string            | Name of the column to be used to Filter the entries in the vector files.                                                 |
| ColumnValue | optional string or  number | Values to filter within `Column` Either a value as number or string or a SCHEME like expression. Described below.        |
| Level       | string                     | Weights given in Text form.                                                                                              |
| Buffer      | optional string of number  | A number to buffer a geo-object or a SCHEME like expression to compute the number (or Array of number). Described below. |
| Source      | optional string            | Naming the source of the data. Not used for processing the data, but to attribute the sources.                           |
| Use         | Yes/No                     | Use or ignore the line.                                                                                                  |
| Commentary  | optional string            | User comments, why rules is processed in a special manner.                                                               |


The **Weights** Sheets describes which Weight (coded as Text) correspondents to which weight as number.

| Column  | DataType | Description                                |
|---------|----------|--------------------------------------------|
| Level   | string   | Weight in Text form, for better usability. |
| Weight  | number   | Weight as number, used In the Cost Raster. |

Commands for Values:

| Command                | Description                                                                                                         |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| (NOT DataFrame)        | Select columns that are included in the original main DataFrame, but not in the in the DataFrame of the expression. |
| (OR column_entry...)   | For allowed entries in the `column` Concat the resulting DataFrame.                                                 |
| (STARTSWITH substring) | Select Entries which `Column`'s values start with the given substring.                                              |

Commands for buffer:

| Command                           | Description                                                                                                                                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| (COL column_name)                 | Extract numbers from column `column_name` to use these as buffer.                                                                                                                       |
| (MAX number... or array[numbers]) | Takes the maximum number of maximum along the array. When mixed number and array of numbers, the number wil be casts to a array filled with the value of the length of the other array. | 

# Script Parameter
The rules can be processed with a common set parameters per cost raster. The common Parameters are:
Positional Arguments:
* 'config_file_path' - path to the config file.
* 'vectors_main_folder' - Path to the main folder of the vector files.
* 'save_dir'  - Save folder for the cost raster and used to add a sub folder for the layer raster.

Optional (named) Parameters:
* '--resolution' - Resolution for the rasterisation. Optional defaults to 1000 m.
* '--all_touched' - Will be set to True, whe all_touched is set, and False when not set.
* '--crs' - Optional integer of the EPSG-CODE of the Reference system. Will be set to EPSG:3157  (Google Pseudo Mercator) when not set.

all_touched describes the which Part of the (Multi-)Polygon overlays the square of a pixel of the new raster.
If all_touched is set to True, then a pixel is viewed as overlaying, if any part of the pixel's square is covered by any polygon.
If all_touched is set to False, then a pixel is viewed as overlaying, if the center of the pixels square is covered by any part of any polygon.

For any pixel that is viewed as overlay, then a valid value is set for that pixel. If the pixel is viewed as not covered,
then a NODATA value is set.

The heuristic is: if only the center point is used for selecting a valid pixel, 
the probability for selecting the pixel is low, if the pixel size is huge compared to vector size, than the probability
that a part of the polygon is covering the centre is low. In this case the probability of using base layer rises.
The base layer does describe the standard value, which is relatively low, but not the lowest value. Hence, all-touched
True thus results is HIGHER weights than all_touched False.

Consequences are:
* all_touched TRUE raster are larger by circa the half resolution in any direction.
* When using the Maximum of different raster, then all_touched TRUE describes the WORST CASE and all_touched False an average case for reasonable raster resolution and best case for low resolution.


# Computation
Each Rule will be converted into a rule wise layer raster. After computing the Raster for each Rule. The Maximum value for all Raster (except for the BASE-Raster) is computed.
This Maximum-Values are written on top of the Result-Raster of the Basis-Rule. This result is taken as cost raster.

