import torch
import torch_geometric
import DataSetGraph as DataSet
import vtk


def ConvertToVTP(data, landmarks):
    #unnormalize the result
    landmarks = landmarks.squeeze() * data.norm_values
    out_landmarks = vtk.vtkPolyData()
    out_landmarks.SetPoints(vtk.vtkPoints())
    for p in range(len(landmarks)):
        out_landmarks.GetPoints().InsertNextPoint(landmarks[p, 0 ], landmarks[p, 1], landmarks[p, 2])
    return out_landmarks
    
def DownsampleMesh(mesh, target_reduction = 0.1):

    # Making sure there are only triangles
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()
    
    filter = vtk.vtkQuadricDecimation()
    filter.SetInputData(mesh)
    filter.SetTargetReduction(1-target_reduction)
    filter.Update()
    decimated_mesh = filter.GetOutput()

    # Calculating normals
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(decimated_mesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    decimated_mesh = filter.GetOutput()
    return decimated_mesh

def ConvertSurfaceToGraph(surface, target_points = 7000):
    initial_points = surface.GetNumberOfPoints()
    downsampled_surface = DownsampleMesh(surface, target_reduction = target_points/initial_points)
    graphdata = DataSet.convert_to_graph(downsampled_surface, None)
    return graphdata

def AddArraysToLandmarks(landmarks, landmark_names = ['Glabella','EuryonL','EuryonR','Opisthocranion','Vertex','Nasion','Sellion','TragionL','TragionR','EndocanthionL','EndocanthionR','ExocanthionL','ExocanthionR']):
    defaultColors = [
    [255, 0, 0], # r
    [0, 255, 0], # g
    [0, 0, 255], # b
    [255, 0, 255], # m
    [0, 255, 255], # c
    [255, 255, 0]  # y
    ]
    
    num_landmarks = landmarks.GetNumberOfPoints()
    colorArray = vtk.vtkFloatArray()
    colorArray.SetName('Color')
    colorArray.SetNumberOfComponents(3)
    for x in range(num_landmarks):
        color = defaultColors[x % len(defaultColors)]
        colorArray.InsertNextTuple3(color[0],color[1],color[2])
    landmarks.GetPointData().AddArray(colorArray)
    nameArray = vtk.vtkStringArray()
    nameArray.SetName("LandmarkName")
    if len(landmark_names) != num_landmarks:
        print('More landmarks in data than names specified. Leaving names blank for now...')
        landmark_names = [f'Landmark{x}' for x in range(num_landmarks)]
    for name in landmark_names:
        nameArray.InsertNextValue(name)
    landmarks.GetPointData().AddArray(nameArray)
    return landmarks

def PlacePatientLandmarksGraph(graph):
    model = torch.jit.load('./MiccaiFinalModel.dat')
    model.eval()
    return model(graph.pos,graph.x,graph.edge_index,graph.node_weight,graph.batch)

def RunInference(surface):
    graph = ConvertSurfaceToGraph(surface)
    landmarks = PlacePatientLandmarksGraph(graph)
    landmarks_vtp = ConvertToVTP(graph, landmarks)
    landmarks_vtp = AddArraysToLandmarks(landmarks_vtp)
    return landmarks_vtp