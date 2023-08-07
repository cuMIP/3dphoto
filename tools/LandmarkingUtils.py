import torch
import SimpleITK as sitk
from vtk.util import numpy_support
import numpy as np
import pdb
import tools.DataSetGraph as DataSet
import vtk
from __init__ import CRANIOFACIAL_LANDMARKING_MODEL_PATH, CRANIOFACIAL_LANDMARKING_NOTEXTURE_MODEL_PATH

def ConvertToVTP(data, landmarks):
    #unnormalize the result
    data.y = landmarks.squeeze()
    # landmarks = landmarks.squeeze() * data.norm_values
    data = DataSet.unnormalize_data(data)
    landmarks = data.y 
    out_landmarks = vtk.vtkPolyData()
    out_landmarks.SetPoints(vtk.vtkPoints())
    for p in range(len(landmarks)):
        out_landmarks.GetPoints().InsertNextPoint(landmarks[p, 0 ], landmarks[p, 1], landmarks[p, 2])
    return out_landmarks 

def InterpolateTextureToPoints(mesh, new_mesh):
    #building the cell locator and extracting the cell textures
    celltopd = vtk.vtkCellDataToPointData()
    celltopd.SetInputData(mesh)
    celltopd.Update()
    mesh = celltopd.GetOutput()

    #looping through all of the points, finding the closest cell, and taking the texture
    textureArray = vtk.vtkFloatArray()
    textureArray.SetName('Texture')
    textureArray.SetNumberOfComponents(3)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.Update()
    for point in range(new_mesh.GetNumberOfPoints()):
        try:
            textureArray.InsertNextTuple3(*mesh.GetPointData().GetArray('Texture').GetTuple(locator.FindClosestPoint(*new_mesh.GetPoint(point))))
        except:
            raise ValueError('No texture array found in photo!')
    
    new_mesh.GetPointData().AddArray(textureArray)
    return new_mesh

def DownsampleMesh(mesh, target_reduction = 0.1, use_texture = True):

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
    if use_texture:
        decimated_mesh = InterpolateTextureToPoints(mesh, decimated_mesh)

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

def HasTexture(surface):
    return bool(surface.GetCellData().HasArray('Texture'))

def ConvertSurfaceToGraph(surface, target_points = 20000, use_texture = True):
    initial_points = surface.GetNumberOfPoints()
    downsampled_surface = DownsampleMesh(surface, target_reduction = target_points/initial_points, use_texture=use_texture)
    graphdata = DataSet.convert_to_graph(downsampled_surface, None, use_texture=use_texture)
    return graphdata

def AddArraysToLandmarks(landmarks, landmark_names = None):
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
    if landmark_names is None:
        landmark_names = DefaultLandmarkNames()
    if len(landmark_names) != num_landmarks:
        print('More landmarks in data than names specified. Leaving names blank for now...')
        landmark_names = [f'Landmark{x}' for x in range(num_landmarks)]
    for name in landmark_names:
        nameArray.InsertNextValue(name)
    landmarks.GetPointData().AddArray(nameArray)
    return landmarks

def DefaultLandmarkNames():
    return ["TRAGION_RIGHT","SELLION","TRAGION_LEFT","EURYON_RIGHT","EURYON_LEFT","FRONTOTEMPORALE_RIGHT","FRONTOTEMPORALE_LEFT","VERTEX","NASION","GLABELLA","OPISTHOCRANION","GNATHION","STOMION","ZYGION_RIGHT","ZYGION_LEFT","GONION_RIGHT","GONION_LEFT","SUBNASALE","ENDOCANTHION_RIGHT","ENDOCANTHION_LEFT","EXOCANTHION_RIGHT","EXOCANTHION_LEFT","ALAR_RIGHT","ALAR_LEFT","NASALE_TIP","SUBLABIALE","UPPER_LIP"]

def PlacePatientLandmarksGraph(data, use_texture = True):
    if use_texture:
        model = torch.jit.load(CRANIOFACIAL_LANDMARKING_MODEL_PATH)
    else:
        model= torch.jit.load(CRANIOFACIAL_LANDMARKING_NOTEXTURE_MODEL_PATH)
    model.eval()
    return model(data.pos, data.x, data.batch, data.edge_index)

def FitLandmarksOnMesh(landmarks, heat_map, surface, graph):
    normal_vectors = (heat_map[:,:,None] * graph.x[:,None,:3]).mean(dim = 0)
    normal_vectors = normal_vectors/(normal_vectors**2).sum(dim = 1).sqrt()[:,None]
    
    # This speeds up finding the intersecitons when doing ray-casting
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(surface) 
    cellLocator.BuildLocator()

    radius = 1000
    # Ray-casting
    rayEnd = np.zeros([3], dtype=np.float64)
    final_landmarks = vtk.vtkPolyData()
    final_landmarks.SetPoints(vtk.vtkPoints())
    for idx in range(landmarks.GetNumberOfPoints()):
        landmark = landmarks.GetPoint(idx)
        normal_vector = normal_vectors[idx,:]
        rayEnd[0] = landmark[0] + radius* normal_vector[0]
        rayEnd[1] = landmark[1] + radius* normal_vector[1]
        rayEnd[2] = landmark[2] + radius* normal_vector[2]

        intersectedPoint = np.zeros([3], dtype=np.float64)
        intersectionPoints = vtk.vtkPoints()
        intersectionCellIds = vtk.vtkIdList()
        obbTree = vtk.vtkOBBTree()
        obbTree.SetDataSet(surface)
        obbTree.BuildLocator()
        if obbTree.IntersectWithLine(landmark, rayEnd, intersectionPoints, intersectionCellIds):
            closestDist = np.inf
            closestId = 0
            for p in range(intersectionPoints.GetNumberOfPoints()):

                intersectionPoints.GetPoint(p, intersectedPoint)
                dist = np.sqrt((landmark[0] - intersectedPoint[0])**2 + (landmark[1] - intersectedPoint[1])**2 + (landmark[2] - intersectedPoint[2])**2 )
            
                if dist < closestDist:
                    closestDist = dist
                    closestId = p
            intersectionPoints.GetPoint(closestId, intersectedPoint)
            final_landmarks.GetPoints().InsertNextPoint(*intersectedPoint)
        else:
            final_landmarks.GetPoints().InsertNextPoint(*landmark)
    return final_landmarks

def CutMeshWithCranialBaseLandmarks(mesh, landmarkCoords, extraSpace=0, useTwoLandmarks=False, invertCropDirection = False):
    """
    Crops the input surface model using the planes defined by the input landmarks

    Parameters
    ----------
    mesh: vtkPolyData
        Cranial surface model
    landmarks: vtkPolyData
        Cranial base landmarks (4 points)
    extraSpace: int
        Indicates the amount of extract space to keep under the planes defined by the cranial base landmarks
    useTwoLandmarks: bool
        Indicates if the cut is done only using the first and fourth landmarks, or using the two planes defined by all the landmarks

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """


    # landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    # for p in range(landmarks.GetNumberOfPoints()):
    #     landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))

    if not useTwoLandmarks:
        
        # normal of first plane
        v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
        v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
        n0 = np.cross(v0, v1)
        n0 = n0 / np.sqrt(np.sum(n0**2))
    
        ###########
        ## Moving landmark coordinates 1 cm away from cranial base so we don't miss the squamousal suture

        distanceToMove = (extraSpace/100.0) * np.abs(np.dot(np.mean(landmarkCoords[1:3,:], axis=0, keepdims=False) - landmarkCoords[3,:], n0))

        landmarkCoords[1:3,:] +=  (n0*distanceToMove).reshape((1,3))

        # Recalculating normal of first plane
        v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
        v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
        n0 = np.cross(v0, v1)
        n0 = n0 / np.sqrt(np.sum(n0**2))
        ###########
    
        # normal of second plane
        v0 = landmarkCoords[2, :] - landmarkCoords[3, :] # For plane 2 (posterior)
        v1 = landmarkCoords[1, :] - landmarkCoords[3, :]
        n1 = np.cross(v0, v1)
        n1 = n1 / np.sqrt(np.sum(n1**2))

        if invertCropDirection:
            plane1 = vtk.vtkPlane()
            plane1.SetNormal(n0)
            plane2 = vtk.vtkPlane()
            plane2.SetNormal(n1)
        else:
            plane1 = vtk.vtkPlane()
            plane1.SetNormal(-n0)
            plane2 = vtk.vtkPlane()
            plane2.SetNormal(-n1)

        plane1.SetOrigin(landmarkCoords[0,:])
        plane2.SetOrigin(landmarkCoords[3,:])

        intersectionFunction = vtk.vtkImplicitBoolean()
        intersectionFunction.AddFunction(plane1)
        intersectionFunction.AddFunction(plane2)
        intersectionFunction.SetOperationTypeToIntersection()
    else:
        if extraSpace > 0:
            # normal of first plane
            v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
            v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
            n0 = np.cross(v0, v1)
            n0 = n0 / np.sqrt(np.sum(n0**2))
            landmarkCoords[0,:] +=  (n0*extraSpace)

            # normal of second plane
            v0 = landmarkCoords[3, :] - landmarkCoords[2, :] # For plane 1 (frontal)
            v1 = landmarkCoords[3, :] - landmarkCoords[1, :]
            n0 = np.cross(v0, v1)
            n0 = n0 / np.sqrt(np.sum(n0**2))
            landmarkCoords[3,:] +=  (n0*extraSpace)
        
        # Normal to plane
        dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :]
        dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

        p0 = landmarkCoords[0, :] + 10 * dorsumVect
        p1 = landmarkCoords[0, :] - 10 * dorsumVect
        p2 = landmarkCoords[3, :]


        v0 = p2 - p1
        v1 = p2 - p0
        n = np.cross(v0, v1)
        n = n / np.sqrt(np.sum(n**2))


        plane = vtk.vtkPlane()
        if invertCropDirection:
            plane.SetNormal(n)
        else:
            plane.SetNormal(-n)
        plane.SetOrigin(p2)


        intersectionFunction = plane

    #cutter = vtk.vtkClipPolyData()
    cutter = vtk.vtkExtractPolyDataGeometry()
    cutter.ExtractInsideOff()
    cutter.SetInputData(mesh)
    #cutter.SetClipFunction(intersectionFunction)
    cutter.SetImplicitFunction(intersectionFunction)
    cutter.Update()

    return cutter.GetOutput()

def vtkPolyDataToNumpy(polydata,arrayName =  None):
    if not arrayName:
        numpyArray = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    else:
        numpyArray = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray(arrayName))
    return numpyArray

def CropSurface(surface, percentage = .4):
    '''
    Inputs
        surface: the VTK PolyData object
        percentage: the amount of data along the Y-axis to cut from the bottom

    Outputs
        The cropped surface
    '''
    # generate the landmarks!
    points = vtkPolyDataToNumpy(surface)
    bottom = np.min(points[:,1])
    top = np.max(points[:,1])
    # make some landmarks at the percentage!
    landmark_y = np.abs(top-bottom) * percentage + bottom
    landmarks = np.zeros([4, 3], dtype=np.float32)
    landmarks[:,1] = landmark_y

    #front!
    landmarks[0,0] = np.mean(points[:,0])
    landmarks[0,2] = np.max(points[:,2])

    #back!
    landmarks[3,0] = np.mean(points[:,0])
    landmarks[3,2] = np.min(points[:,2])

    #right size
    landmarks[1,0] = np.min(points[:,0])
    landmarks[1,2] = np.mean(points[:,2])

    #left side
    landmarks[2,0] = np.max(points[:,0])
    landmarks[2,2] = np.mean(points[:,2])

    cropped_surface = CutMeshWithCranialBaseLandmarks(surface, landmarks, extraSpace=0, useTwoLandmarks=True)
    return cropped_surface

def RunInference(surface, crop = True, return_cropped_image = False, crop_percentage = 0.4):
    # crop if needed 
    if crop:
        if crop_percentage>=1:
            raise ValueError('Cropping percentage is >= 1. Please lower cropping percentage to < 1.')
        cropped_surface = CropSurface(surface, percentage = crop_percentage)
    else:
        cropped_surface = surface

    use_texture = HasTexture(cropped_surface)
    graph = ConvertSurfaceToGraph(cropped_surface, use_texture = use_texture)
    landmarks, heat_map = PlacePatientLandmarksGraph(graph, use_texture = use_texture)
    landmarks_vtp = ConvertToVTP(graph, landmarks)
    landmarks_vtp = FitLandmarksOnMesh(landmarks_vtp, heat_map, surface, graph)
    landmarks_vtp = AddArraysToLandmarks(landmarks_vtp)
    if return_cropped_image:
        return landmarks_vtp, cropped_surface
    else:
        return landmarks_vtp
