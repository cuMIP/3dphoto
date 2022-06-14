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