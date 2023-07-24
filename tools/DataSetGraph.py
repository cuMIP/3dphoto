import vtk
import pdb
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch

def ReadPolyData(filename):
    if filename.endswith('.vtk'):
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def WritePolyData(data, filename):
    if filename.endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    # Saving landmarks
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Update()
    return


class PhotoLandmarkDatasetGraph(Dataset):

    def __init__(self, photo_filenames, landmark_filenames, subsample_points = None,transform = None, pre_transform = None, normalize = True):
        super().__init__(None, transform, pre_transform)
        self.landmarks = landmark_filenames
        self.photos = photo_filenames
        self.subsample_points = subsample_points
        self.num_classes = 27 * 3
        self.normalize = normalize

    #checks to skip the downloading
    @property
    def raw_file_names(self):
        return list(zip(self.photo_filenames,self.landmark_filenames))

    #checks to skip the processing
    @property
    def processed_file_names(self):
        return list(zip(self.photo_filenames,self.landmark_filenames))

    def __len__(self):
        return len(self.photos)
    
    def __getitem__(self, idx):
    #torch geometric uses the "get" method instead of "getitem"
    # def get(self, idx):

        image_vtp = ReadPolyData(self.photos[idx])

        image = vtk_to_numpy(image_vtp.GetPoints().GetData())

        landmarks_vtp = ReadPolyData(self.landmarks[idx])
        landmarks = vtk_to_numpy(landmarks_vtp.GetPoints().GetData())

        # image_vtp = self.smooth_mesh(image_vtp)
        # if self.transform:
        #     image = self.transform(image)
        #     landmarks = self.transform(landmarks)
        data = convert_to_graph(image_vtp, landmarks)
        if self.normalize:
            data = normalize_data(data)
        return data

def Scaler(t):
    ##center at 0,0,0 at divide by the vector magnitude
    return (t - torch.mean(t, dim = 0)) / torch.mean(torch.linalg.norm(t, dim = 1))

def normalize_data(data):
    #first let's unnormalize everything
    # data.pos = data.pos * data.norm_values
    # data.y = data.y * data.norm_values

    data.norm_values = (torch.mean(data.pos, dim = 0),torch.mean(torch.linalg.norm(data.pos, dim = 1)))
    #first the landmarks
    if data.y is not None:
        data.y = (data.y - data.norm_values[0]) / data.norm_values[1]

    #position features!
    data.pos = Scaler(data.pos)

    #normals are already scaled how we want them
    # data.x[:,:3] = data.x[:,:3]

    #now for RGB features
    data.x[:,3:] = Scaler(data.x[:,3:])
    return data

# def normalize_data(data):
#     data.norm_values = torch.mean(torch.abs(data.pos), axis = 0)
#     data.pos = data.pos / data.norm_values
#     if data.y is not None:
#         data.y = data.y / data.norm_values
#     return data

def unnormalize_data(data):
    if not hasattr(data, 'norm_values'):
        raise ValueError('Norm value was not found! The data may not be currently normalized.')
    else:
        data.pos = (data.pos * data.norm_values[1]) + data.norm_values[0]
        if data.y is not None:
            data.y = (data.y * data.norm_values[1]) + data.norm_values[0]
    return data

# def unnormalize_data(data):
#     if not hasattr(data.norm_values):
#         raise ValueError('Norm value was not found! The data may not be currently normalized.')
#     else:
#         data.pos = data.pos * data.norm_values
#         if data.y is not None:
#             data.y = data.y * data.norm_values
#     return data

def convert_to_graph(image_vtp, landmarks, use_texture = True):
    '''
        Function to convert a 3D photograph (VTP mesh) and its landmarks to a graph
        According to https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        x = node features
        y = labels
        pos = node positions
        edge_indices = COO format of graph 
    '''
    y = landmarks
    pos = vtk_to_numpy(image_vtp.GetPoints().GetData())
    x = vtk_to_numpy(image_vtp.GetPointData().GetNormals())
    edge_table = get_edges_of_mesh(image_vtp)
    edge_indices = convert_to_coo(edge_table)
    node_weights = calc_node_weights(torch.tensor(pos))
    if use_texture:
        #normalize the texture
        texture = vtk_to_numpy(image_vtp.GetPointData().GetArray('Texture'))/255
        x = torch.cat((torch.tensor(x), torch.tensor(texture)), dim = 1)
    else:
        x = torch.tensor(x)
    if landmarks is not None:
        data = Data(x = x, y = torch.tensor(y), pos = torch.tensor(pos), edge_index = torch.tensor(edge_indices, dtype = torch.long), node_weight = node_weights, num_nodes = len(pos))
    else:
        data = Data(x = x, pos = torch.tensor(pos), edge_index = torch.tensor(edge_indices, dtype = torch.long), node_weight = node_weights, num_nodes = len(pos))


    data = normalize_data(data)
    #now adjust the batch
    data.batch = torch.zeros(data.pos.shape[0], dtype = torch.int64)
    return data

def calc_node_weights(pos):
    return torch.empty(pos.shape[0])
    # pdb.set_trace()
    # dist = torch.cdist(pos, pos)
    # average_dist = torch.sum(dist, dim = 0)/(dist.shape[0]-1)
    # normalized_dist = (average_dist - average_dist.min())/ (average_dist.max() - average_dist.min())
    # return torch.stack([normalized_dist,1-normalized_dist], dim = 1)

def get_edges_of_mesh( mesh):
    
    '''
    Construct an edge list using COO format
    '''
    edge_table = {}
    for point in range(mesh.GetNumberOfPoints()):
        # print(f'Extracting edges for point {point} out of {mesh.GetNumberOfPoints()}', end = '\r')
        #for each cell
        cellidlist = vtk.vtkIdList()
        mesh.GetPointCells(point, cellidlist)
        points = []
        for cellid in range(cellidlist.GetNumberOfIds()):
            #find the points for each cell
            pointidlist = vtk.vtkIdList()
            mesh.GetCellPoints(cellidlist.GetId(cellid), pointidlist) # get cell ids
            #get the actual points belonging to the cells
            points += [pointidlist.GetId(x) for x in range(pointidlist.GetNumberOfIds())]
        #only take each point once
        edge_table[point] = list(np.unique(points))
    return edge_table

def convert_to_coo(edge_table):
    # print('Converting format to coo...')
    in_edges = []
    out_edges = []
    for key, val in edge_table.items():
        in_edges += [key] * len(val)
        out_edges += val
    return np.array([in_edges, out_edges])

def smooth_mesh(mesh):
    #smooth the mesh!
    # Making sure there are only triangles
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    sampledMesh = filter.GetOutput()

    filter = vtk.vtkSmoothPolyDataFilter()
    filter.SetInputData(sampledMesh)
    filter.SetNumberOfIterations(200)
    filter.SetRelaxationFactor(1)
    filter.FeatureEdgeSmoothingOff()
    filter.Update()
    sampledMesh = filter.GetOutput()

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(sampledMesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    return filter.GetOutput()

def interpolate_texture_to_points(mesh):
    textures = np.zeros([mesh.GetNumberOfPoints(), 3])
    celltextures = mesh.GetCellData().GetArray('Texture')
    for point in range(mesh.GetNumberOfPoints()):
        #for each cell
        cellidlist = vtk.vtkIdList()
        mesh.GetPointCells(point, cellidlist)
        textures[point, :] = np.mean(np.array([celltextures.GetTuple(cellidlist.GetId(cellid)) for cellid in range(cellidlist.GetNumberOfIds())]), axis = 0)/255
    return textures