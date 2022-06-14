# 3dphoto landmarking
This is a repository for the [Graph convolutional network with probabilistic spatial regression: application to craniofacial landmark detection from 3D photogrammetry](https://github.com/cuMIP/3dphoto/).
This repository contains the final model as described in the manuscript. The model is stored as a [TorchScript](https://pytorch.org/docs/stable/jit.html) jit file ``MiccalFinalModel.dat``.


![Network diagram as found in published manuscript](/diagrams/NetworkArchitectureDiagram.jpg)

## Dependencies:
- [Python](python.org) `(version 3.6 or higher)`
- [Pytorch](https://pytorch.org/get-started/locally)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [NumPy](https://numpy.org/install/)
- [VTK](https://pypi.org/project/vtk/)

    *Once Python is installed, each of these packages can be downloaded using [Python pip](https://pip.pypa.io/en/stable/installation/)*


## Using the code
Due to data privacy agreements, we are not able to share any example data. To run this code, you will need an image in VTK [PolyData format (.vtp)](https://vtk.org/doc/nightly/html/classvtkPolyData.html) containing the entire head and face that is cropped around the mid-neck. This model was trained on data collected using a [3DMD-Head system](https://3dmd.com/products/).

### Quick summary
**Input**: VTK PolyData file.

**Output**: VTK PolyData containing 13 craniofacial landmarks.

### Code example
The entire code to place a series of landmarks on an image in .vtp format can be seen here: 
```python
from MiccaiModel import RunInference
import vtk

#define the path to the data
data_filename = '../path/to/example/data/Data.vtp'
#VTK xml reader
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(data_filename)
reader.Update()
#read in the image data
image = reader.GetOutput()
#run the inference
landmarks = RunInference(image)
```
*When using this code, be sure to change the ```data_filename``` variable to a path to a valid VTK PolyData file*.

### The RunInference Function
The **RunInference** function executes three steps: 

1. **ConvertSurfaceToGraph**: Converts the .vtp into a graph with the proper attributes.
2. **PlacePatientLandmarksGraph**: Passes this graph to the model and gets the output.
3. **ConvertToVTP**: Converts the output back into .vtp.

Each of these functions are contained separately within the ```MiccaiModel.py``` file. To see more information about the graph datatype, please see the custom dataset made for this model in ```DataSetGraph.py``` or the Pytorch geometric website on [datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data).

## Troubleshooting
- Example code will not run:
    - Be sure you have not altered the "MiccaiFinalModel.dat" file in any way.
    - Be sure that you are trying to load a valid VTK PolyData file.
    - Be sure that you have installed the required dependencies as noted in the **Dependencies** section.
- Landmarks are not placed correctly:
    - Your data may be something we have not seen before! If possible, please email the corresponding author (connor.2.elkhill@cuanschutz.edu) so that we may evaluate.
- Model inference takes a long time:
    - This is surprising! Our model is very small.
    - If you have a GPU and a Cuda-compatible version of Pytorch, feel free to modify the ```RunInference``` source code within "MiccaiModel.py" to use the GPU available.
    - You may also want to adjust the amount of points in the graph. You can lower the ```target_points``` argument within the ```ConvertSurfaceToGraph``` function ("MiccaiModel.py") to reduce the number of points.

Any other questions? Please email the corresponding author Connor Elkhill at connor.2.elkhill@cuanschutz.edu