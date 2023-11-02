from __init__ import *
import pandas as pd
from tools.LandmarkingUtils import CutMeshWithCranialBaseLandmarks, vtkPolyDataToNumpy, SelectLandmarks, ComputeVolume
import tools.DataSetGraph as DataSet
import vtk
from vtk.util import numpy_support
import SimpleITK as sitk
import os
import numpy as np
import shelve
import warnings

#filter the scikit learn warning about pickled models
warnings.filterwarnings("ignore", category = UserWarning)

# This prevents VTK from opening windows with warnings. It may be useful to remove for debugging purposes
vtk.vtkObject.GlobalWarningDisplayOff()

def CreateContinuousModelOfExternalCranialSurface(inputMesh, landmarks, cutMesh = False, numberOfThetas=150, maskResult=False, verbose=False, reversePlane = False):
    """
    Creates a continuous model of the external cranial surface from the bone model segmented from the CT image.
    Landmarking is done via raycasting in spherical coordinates 

    Parameters
    ----------
    inputMesh: vtkPolyData
        Bone surface model
    landmarks: vtkPolyData
        Cranial base landmarks
    numberOfThetas: int
        Sampling resolution in the elevation angle
    maskResult: bool
        Indicates if the resulting mesh will be cropped at the cranial base using the input landmarks
    useGlabella: bool
        if True, the landmarks are located at the glabella, temporal processes of the dorsum sellae and opisthion.
        If False, the landmarks are located at the nasion, temporal processes of the dorsum sellae and opisthion.
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    vtkPolyData
        The resulting external cranial surface model
    """

    # Creating a copy of the input meshes
    a = vtk.vtkPolyData()
    a.DeepCopy(inputMesh)
    inputMesh = a

    a = vtk.vtkPolyData()
    a.DeepCopy(landmarks)
    landmarks = a

    a = None

    """
    # Wrapping a sphere around the mesh
    """

    if cutMesh:
        inputMesh = CutMeshWithCranialBaseLandmarks(inputMesh, landmarks, extraSpace=20, useTwoLandmarks=False)

    # Warping a sphere to the outer surface
    center = np.zeros([3], dtype=np.float64)
    for p in range(inputMesh.GetNumberOfPoints()):
        center += np.array(inputMesh.GetPoint(p))
    center /= inputMesh.GetNumberOfPoints()

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(500)
    sphere.SetPhiResolution(120)
    sphere.SetThetaResolution(120)
    sphere.SetCenter(center)
    sphere.Update()
    sphere = sphere.GetOutput()

    wrappedSphere = vtk.vtkSmoothPolyDataFilter()
    wrappedSphere.SetInputData(0, sphere)
    wrappedSphere.SetInputData(1, inputMesh)
    wrappedSphere.Update()
    wrappedSphere = wrappedSphere.GetOutput()

    # Subdividing
    filter = vtk.vtkButterflySubdivisionFilter()
    filter.SetInputData(wrappedSphere)
    filter.SetNumberOfSubdivisions(2)
    filter.Update()
    wrappedSphere = filter.GetOutput()

    # Moving points to the bone so adjustment to the shape is perfect
    for p in range(wrappedSphere.GetNumberOfPoints()):
        coords = wrappedSphere.GetPoint(p)
        closestId = inputMesh.FindPoint(coords[0], coords[1], coords[2])
        coords = np.array(inputMesh.GetPoint(closestId))
        wrappedSphere.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])
    inputMesh = wrappedSphere

    """
    # Aligning with the template (only for rotation), and correcting for rotation inaccuracies of the template
    """
    templateLandmarks = DataSet.ReadPolyData(GLABELLA_CRANIALBASE_LANDMARKS_PATH)

    npLandmarks = np.zeros((landmarks.GetNumberOfPoints(), 3), dtype=np.float64)
    npTemplateLandmarks = np.zeros((templateLandmarks.GetNumberOfPoints(), 3), dtype=np.float64)
    for p in range(landmarks.GetNumberOfPoints()):
        npLandmarks[p,:] = landmarks.GetPoint(p)
        npTemplateLandmarks[p,:] = templateLandmarks.GetPoint(p)
    
    center = np.mean(npLandmarks, axis=0)

    R, t = RegisterPointClouds(npLandmarks, npTemplateLandmarks, scaling=False)

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R.ravel())
    transform.SetCenter(center)
    transform.SetTranslation(t)

    rotationTransform = sitk.Euler3DTransform()
    rotationTransform.SetIdentity()
    rotationTransform.SetRotation(0, 0, -174 * np.pi / 180.0)
    rotationTransform.SetCenter(center)
    

    for p in range(landmarks.GetNumberOfPoints()):
        coords = rotationTransform.TransformPoint(transform.TransformPoint(landmarks.GetPoint(p)))
        landmarks.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    for p in range(inputMesh.GetNumberOfPoints()):
        coords = rotationTransform.TransformPoint(transform.TransformPoint(inputMesh.GetPoint(p)))
        inputMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    """
    # Cutting planes and ray casting initialization
    """

    # Center for ray casting is the dorsum sellae
    center = (np.array(landmarks.GetPoint(1)) + np.array(landmarks.GetPoint(2))) / 2.0

    # Mesh bounds
    meshBounds = np.zeros([6], dtype=np.float64)
    inputMesh.GetBounds(meshBounds)

    ## New mesh
    sampledMesh = vtk.vtkPolyData()
    sampledMesh.SetPoints(vtk.vtkPoints())
    
    ## Calculating the cutting plane
    landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):
        landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))

    # Normal to plane
    dorsumCoords = (landmarkCoords[1, :] + landmarkCoords[2, :]) / 2.0 # Center of dorsum sellae
    midCoords = (landmarkCoords[0, :] + landmarkCoords[3, :]) / 2.0 # Center of nasion and opisthion

    dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :] # Vector of dorsum sellae
    dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

    p0 = landmarkCoords[0, :] + 10 * dorsumVect # Points parallel to dorsum sellae at the sides of the nasion
    p1 = landmarkCoords[0, :] - 10 * dorsumVect
    p2 = landmarkCoords[3, :] # opisthion

    # Normal to the plane nasion - opisthion
    v0 = p2 - p1
    v1 = p2 - p0
    n = np.cross(v0, v1)
    n = n / np.sqrt(np.sum(n**2))

    plane = vtk.vtkPlane()
    if not reversePlane:
        plane.SetNormal(n)
    else:
        plane.SetNormal(-n)
    plane.SetOrigin(p2)
    
    """
    # Sampling
    """
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(inputMesh)
    obbTree.BuildLocator()

    intersectionPoints = vtk.vtkPoints()
    intersectionCellIds = vtk.vtkIdList()

    thetaDistance = np.pi / (numberOfThetas - 1.0)

    rayEnd = np.zeros([3], dtype=np.float64)
    radius = np.sqrt( (meshBounds[1] - meshBounds[0])**2 + (meshBounds[3] - meshBounds[2])**2 + (meshBounds[5] - meshBounds[4])**2 )
    intersectedPoint = np.zeros([3], dtype=np.float64)

    ## Adding all arrays with one component
    arrayList = []
    for id in range(inputMesh.GetCellData().GetNumberOfArrays()):

        if inputMesh.GetCellData().GetArray(id).GetNumberOfComponents() == 1:
            newArray = vtk.vtkFloatArray()
            newArray.SetName(inputMesh.GetCellData().GetArray(id).GetName())
            newArray.SetNumberOfComponents(1)
            arrayList += [newArray]

    # Adding the array with reconstruction information
    newArray = vtk.vtkFloatArray()
    newArray.SetName('coords')
    newArray.SetNumberOfComponents(3)
    arrayList += [newArray]

    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    triangle = vtk.vtkGenericCell()

    nasion = landmarkCoords[0,:]
    dist = np.linalg.norm(nasion - center) # Distance nasion to center

    nasionSinTheta =  (nasion[2] - center[2])/dist
    nasionSinPhi = (nasion[1] - center[1]) / (dist * np.sqrt( (1 - nasionSinTheta**2) ) )

    opisthion = landmarkCoords[3,:]
    dist = np.linalg.norm(opisthion - center) # Distance nasion to center

    opisthionSinTheta =  (opisthion[2] - center[2])/dist
    opisthionSinPhi = (opisthion[1] - center[1]) / (dist * np.sqrt( (1 - opisthionSinTheta**2) ) )

    for latitude in range(numberOfThetas):

        if verbose:
            print('Sampling spherical space: {:05d}/{:05d}'.format(latitude, numberOfThetas), end='\r')

        theta = np.pi/2 - latitude*np.pi/(numberOfThetas-1)

        thetaLongitude = 2.0*np.pi*np.cos(theta)
        nPointsAtTheta = int(np.floor(thetaLongitude / thetaDistance))

        for longitude in range(nPointsAtTheta):
            if nPointsAtTheta != 1:
                phi = -np.pi + longitude*2*np.pi / (nPointsAtTheta - 1)
            else:
                phi = 0


            distToNasion = (nasionSinPhi-np.sin(phi))**2
            distToOpisthion = (opisthionSinPhi-np.sin(phi))**2
            
            limitTheta = nasionSinTheta * distToOpisthion/(distToNasion + distToOpisthion) + opisthionSinTheta * distToNasion/(distToNasion + distToOpisthion)

            if not maskResult or np.sin(theta) >= limitTheta:
            
                rayEnd[0] = center[0] + radius*np.cos(phi)*np.cos(theta)
                rayEnd[1] = center[1] + radius*np.sin(phi)*np.cos(theta)
                rayEnd[2] = center[2] + radius*np.sin(theta)

                if obbTree.IntersectWithLine(center, rayEnd, intersectionPoints, intersectionCellIds):
                    closestDist = 0
                    closestId = 0
                    for p in range(intersectionPoints.GetNumberOfPoints()):

                        intersectionPoints.GetPoint(p, intersectedPoint)
                        dist = np.sqrt((center[0] - intersectedPoint[0])**2 + (center[1] - intersectedPoint[1])**2 + (center[2] - intersectedPoint[2])**2 )
                    
                        if dist > closestDist:
                            closestDist = dist
                            closestId = p

                    intersectionPoints.GetPoint(closestId, intersectedPoint)
                    # Intersected point has the Euclidean coordinates of the points at the specific longitude and latitude

                    # Finding the color
                    pCoords = np.zeros([3], dtype=np.float64)
                    weights = np.zeros([3], dtype=np.float64)
                    subId = vtk.reference(0)
                    cellId = cellLocator.FindCell(intersectedPoint, 0, triangle, pCoords, weights)

                    #Calculating the cartesian coordinates for the spherical map
                    angle = phi
                    rho = float(latitude)/(numberOfThetas-1)

                    x = rho * np.cos(phi)
                    y = rho * np.sin(phi)


                    if maskResult:
                        toDraw = plane.EvaluateFunction(intersectedPoint) > 0.1
                    else:
                        toDraw = True

                    if toDraw:
                        sampledMesh.GetPoints().InsertNextPoint(x, y, 0)

                        for arrayId in range(len(arrayList)-1):
                    
                            if cellId >= 0:
                                arrayList[arrayId].InsertNextTuple1(inputMesh.GetCellData().GetArray(arrayList[arrayId].GetName()).GetTuple1(cellId))
                            else:
                                arrayList[arrayId].InsertNextTuple1(0)

                        arrayList[-1].InsertNextTuple3(intersectedPoint[0], intersectedPoint[1], intersectedPoint[2])

    for thisArray in arrayList:
        sampledMesh.GetPointData().AddArray(thisArray)

    """
    # Triagulating and recosntructing from spherical to cranial mesh
    """
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(sampledMesh)
    filter.Update()
    sampledMesh = filter.GetOutput()
    if verbose:
        print()

    for p in range(sampledMesh.GetNumberOfPoints()):
        coords = sampledMesh.GetPointData().GetArray('coords').GetTuple3(p)
        sampledMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    sampledMesh.GetPointData().RemoveArray('coords')

    # Cleaning: eliminating duplicated points and cells
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(sampledMesh)
    filter.PointMergingOn()
    filter.Update()
    sampledMesh = filter.GetOutput()

    # Making sure there are only triangles
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(sampledMesh)
    filter.Update()
    sampledMesh = filter.GetOutput()

    """
    # Transforming back to the original coordinates space
    """
    transform = transform.GetInverse()
    rotationTransform = rotationTransform.GetInverse()
    for p in range(sampledMesh.GetNumberOfPoints()):
        coords = transform.TransformPoint(rotationTransform.TransformPoint(sampledMesh.GetPoint(p)))
        sampledMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    """
    # Smoothing to avoid problems with normals when the bone model comes from low quality images
    """
    filter = vtk.vtkSmoothPolyDataFilter()
    filter.SetInputData(sampledMesh)
    filter.SetNumberOfIterations(30)
    filter.Update()
    sampledMesh = filter.GetOutput()
    

    """
    # Calculating normals
    """
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(sampledMesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    sampledMesh = filter.GetOutput()

    return sampledMesh

### Closes PolyData mesh with holes
def closeMesh (mesh):

    # We eliminate duplicates, because they may be marked as boundaries incorrectly
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()
    
    # We make sure that there are only triangle cells in the mesh
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.PassLinesOff()
    filter.PassVertsOff()
    filter.Update()
    mesh = filter.GetOutput()
    
    # Get edges
    filter = vtk.vtkFeatureEdges()
    filter.SetInputData(mesh)
    filter.ExtractAllEdgeTypesOff()
    filter.BoundaryEdgesOn()
    filter.Update()
    exteriorEdges = filter.GetOutput()
    
    # Triagulate edges
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(exteriorEdges)
    filter.SetProjectionPlaneMode(2) # VTK_BEST_FITTING_PLANE
    filter.Update()
    triangulatedEdges = filter.GetOutput()
    
    # Append meshes
    filter = vtk.vtkAppendPolyData()
    filter.AddInputData(mesh)
    filter.AddInputData(triangulatedEdges)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Fill any small holes that may remain
    filter = vtk.vtkFillHolesFilter()
    filter.SetInputData(closedMesh)
    filter.SetHoleSize(1e6)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Clean mesh
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(closedMesh)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # We make sure that there are only triangle cells in the mesh
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(closedMesh)
    filter.PassLinesOff()
    filter.PassVertsOff()
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Update normals
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(closedMesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    closedMesh = filter.GetOutput()
    
    return closedMesh

def meshToVolume (mesh):
    
    # Fills all holes in mesh
    mesh = closeMesh(mesh)  
    
    # Gets bounds, spacing, origin, and dimensions of mesh
    bounds = np.array(mesh.GetBounds())
    #give the bounds a bit of extra space
    bounds = bounds * 1.1
    spacing = (1, 1, 1)
    origin = bounds[0::2]
    dims = (bounds[1::2] - origin) / spacing
    dims = dims.astype(np.int32)
    
    # Creates image with above defined spacing, origin, dimensions, and extent; allocates scalars
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDimensions(dims)
    image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    image.AllocateScalars(3, 1) # vtk.VTK_UNSIGNED_CHAR
    
    # Fills the image with white voxels (outValue is used for background)
    inValue = 1
    outValue = 0
    
    for i in range(image.GetNumberOfPoints()):
        image.GetPointData().GetScalars().SetTuple1(i, inValue)
    
    # Converts PolyData (mesh) to ImageStencil, then converts ImageStencil to ImageData
    polyStencil = vtk.vtkPolyDataToImageStencil()
    imageStencil = vtk.vtkImageStencil()
    
    polyStencil.SetInputData(mesh)
    polyStencil.SetOutputOrigin(image.GetOrigin())
    polyStencil.SetOutputSpacing(image.GetSpacing())
    polyStencil.SetOutputWholeExtent(image.GetExtent())
    polyStencil.Update()
    
    imageStencil.SetInputData(image)
    imageStencil.SetStencilData(polyStencil.GetOutput())
    imageStencil.ReverseStencilOff()
    imageStencil.SetBackgroundValue(outValue)
    imageStencil.Update()
       
    # Returns ImageData output
    image = imageStencil.GetOutput()
    return image

def RegisterPointClouds(A, B, scaling=False):
    """
    Calculates analytically the least-squares best-fit transform between corresponding 3D points A->B.

    Parameters
    ----------
    A: np.array
        Moving point cloud with shape Nx3, where N is the number of points
    B: np.array
        Fixed point cloud with shape Nx3, where N is the number of points
    scaling: bool
        Indicates if the calculated transformation is purely rigid (False) or contains isotropic scaling (True)

    Returns
    -------
    np.array
        Rotation (+scaling) matrix with shape 3x3
    np.array
        Translation vector with shape 3
    """

    assert len(A) == len(B) # Both point clouds must have the same number of points
    
    zz = np.zeros(shape=[A.shape[0],1])
    A = np.append(A, zz, axis=1)
    B = np.append(B, zz, axis=1)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Scaling
    if scaling:
        AA = np.dot(R.T, AA.T).T
        s = np.mean(np.linalg.norm(BB[:,:-1], axis=1)) / np.mean(np.linalg.norm(AA[:,:-1], axis=1))
        R *= s

    R = R[:3,:3]#.T
    t = (centroid_B - centroid_A)[:3]

    return R, t

def RegisterPatientToTemplate(photo_landmarks, template_landmarks):
    moving_landmarks_arr = np.zeros([photo_landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    template_landmarks_arr = np.zeros([photo_landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(photo_landmarks.GetNumberOfPoints()):
        moving_landmarks_arr[p,:3] = np.array(photo_landmarks.GetPoint(p))
        template_landmarks_arr[p,:] = np.array(template_landmarks.GetPoint(p))
    R, t = RegisterPointClouds(moving_landmarks_arr, template_landmarks_arr, scaling = False)
    center = np.mean(moving_landmarks_arr, axis=0).astype(np.float64)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R.ravel())
    transform.SetCenter(center)
    transform.SetTranslation(t)
    return transform

def SmoothMesh(sampledMesh):
    #smooth the mesh!
    # Making sure there are only triangles
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(sampledMesh)
    filter.Update()
    sampledMesh = filter.GetOutput()

    filter = vtk.vtkSmoothPolyDataFilter()
    filter.SetInputData(sampledMesh)
    filter.SetNumberOfIterations(50)
    filter.SetRelaxationFactor(0.001)
    filter.Update()
    sampledMesh = filter.GetOutput()
    
    filter = vtk.vtkWindowedSincPolyDataFilter()
    filter.SetInputData(sampledMesh)
    filter.SetNumberOfIterations(50)
    filter.SetPassBand(0.001)
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

def ApplyTransform(data, transform):
    for p in range(data.GetNumberOfPoints()):
        coords = np.array(data.GetPoint(p))
        newCoords = transform.TransformPoint(coords)
        data.GetPoints().SetPoint(p, newCoords[0], newCoords[1], newCoords[2])
    # Recalculating the normals and saving
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(data)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    data = filter.GetOutput()
    return data

def vtkToSitkImage(vtkImage):

    numpyImage = numpy_support.vtk_to_numpy(vtkImage.GetPointData().GetScalars()).reshape(vtkImage.GetDimensions()[::-1])
    
    sitkImage = sitk.GetImageFromArray(numpyImage)
    sitkImage.SetOrigin(vtkImage.GetOrigin())
    sitkImage.SetSpacing(vtkImage.GetSpacing())
    
    return sitkImage

def ConvertSitkImage(image, sitk_type=sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk_type)
    return castImageFilter.Execute(image)

def CreateMeshFromBinaryImage(binaryImage, insidePixelValue=1):
    """
    Uses the marching cubes algorithm to create a surface model from a binary image

    Parameters
    ----------
    binaryImage: sitkImage
        The binary image
    insidePixelValue: {int, float}
        The pixel value to use for mesh creation

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """

    numpyImage = sitk.GetArrayViewFromImage(binaryImage).astype(np.ubyte)
    
    dataArray = numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(),  deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)

    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(binaryImage.GetSpacing()[0], binaryImage.GetSpacing()[1], binaryImage.GetSpacing()[2])
    vtkImage.SetOrigin(binaryImage.GetOrigin()[0], binaryImage.GetOrigin()[1], binaryImage.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)

    # filter = vtk.vtkMarchingCubes()
    filter = vtk.vtkFlyingEdges3D()
    filter.SetInputData(vtkImage)
    filter.SetValue(0, insidePixelValue)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    return mesh

def AlignPatientToTemplate(surface, landmarks):

    l = vtk.vtkPolyData()
    l.DeepCopy(landmarks)
    landmarks = l

    l = vtk.vtkPolyData()
    l.DeepCopy(surface)
    surface = l

    #### POSE REGISTRATION ####
    # generate target points from template
    templateLandmarks_forRegisitration = DataSet.ReadPolyData(EURYON_CRANIALBASE_LANDMARKS_PATH)
    template_space_landmarks = DataSet.ReadPolyData(GLABELLA_CRANIALBASE_LANDMARKS_PATH)
    

    # select some landmarks? 
    landmarks_we_want = ["SELLION","SUBNASALE","TRAGION_RIGHT","TRAGION_LEFT",
                     "ENDOCANTHION_RIGHT","ENDOCANTHION_LEFT",
                     "VERTEX", 'OPISTHOCRANION']
    selected_templateLandmarks_forRegisitration= SelectLandmarks(templateLandmarks_forRegisitration, landmarks_we_want)
    landmarks_forregistration = SelectLandmarks(landmarks, landmarks_we_want)

    alignment_transform = RegisterPatientToTemplate(landmarks_forregistration, selected_templateLandmarks_forRegisitration)
    transform = sitk.CompositeTransform([alignment_transform])

    #we may need a quick ICP to match the centers
    template_space_photo = ApplyTransform(surface, transform)
    template_space_patient_landmarks = ApplyTransform(landmarks, transform)

    #### CENTERING REGISTRATION ####
    # compute the transform to center the nasion - tragia plane to the template plane
    landmarks_we_want = ["SELLION","TRAGION_RIGHT","TRAGION_LEFT"]
    selected_template_landmarks_forregistration= SelectLandmarks(templateLandmarks_forRegisitration, landmarks_we_want)
    landmarks_forregistration = SelectLandmarks(template_space_patient_landmarks, landmarks_we_want)
    # figure out the scaling
    centering_transform = RegisterPatientToTemplate(landmarks_forregistration, selected_template_landmarks_forregistration)
    rotation = np.eye(3)
    centering_transform.SetMatrix(rotation.ravel())
    transform.AddTransform(centering_transform)

    #move the template stuff
    template_space_photo = ApplyTransform(template_space_photo, centering_transform)
    template_space_patient_landmarks = ApplyTransform(template_space_patient_landmarks, centering_transform)


    #### NOW MAKE A GOOD EXTERNAL HEAD SURFACE ####
    #make it a binary image
    template_space_mesh = CreateContinuousModelOfExternalCranialSurface(template_space_photo, template_space_landmarks, cutMesh=False, maskResult=False)

    #smooth it!
    template_space_mesh = SmoothMesh(template_space_mesh)
    #cut it!
    template_space_mesh = CutMeshWithCranialBaseLandmarks(template_space_mesh, vtkPolyDataToNumpy(template_space_landmarks), extraSpace=120)

    #### SCALING ####
    #figure out the scale, based on the average distance from the cranial base
    cutting_landmarks = DataSet.ReadPolyData(CUTTING_LANDMARKS_PATH)
    cut_patient = CutMeshWithCranialBaseLandmarks(template_space_mesh, vtkPolyDataToNumpy(cutting_landmarks))

    #compute the scale
    scale = (TEMPLATE_VOLUME / ComputeVolume(cut_patient)) ** (1/3)
    scaling_transform = sitk.AffineTransform(3)
    scaling_transform.SetIdentity()
    rotation = np.eye(3) * scale
    scaling_transform.SetMatrix(rotation.ravel())
    center = vtkPolyDataToNumpy(selected_template_landmarks_forregistration).mean(axis = 0).astype(np.float64)
    scaling_transform.SetCenter(center)
    transform.AddTransform(scaling_transform)
    template_space_mesh = ApplyTransform(template_space_mesh, scaling_transform)
    template_space_patient_landmarks = ApplyTransform(template_space_patient_landmarks, scaling_transform)

    transform = sitk.CompositeTransform([scaling_transform, centering_transform, alignment_transform])
    transform.FlattenTransform() # this allows us to save it out

    #### TRANSFORM IS DONE, MOVE IT BACK INTO PATIENT SPACE ####
    #back into template space and smooth again
    patient_space_photo = ApplyTransform(template_space_mesh, transform.GetInverse())

    # Calculating normals to improve visualization
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(patient_space_photo)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    output_mesh = filter.GetOutput()
    
    #smoothed mesh is in patient space!
    return output_mesh, transform

def CreateSphericalMapFromSurfaceModel(inputMesh, numberOfThetas=150, maskResult=True):
    """
    Creates a spherical map representation of a cranial bone surface model.

    Parameters
    ----------
    inputMesh: vtkPolyData
        Cranial bone surface model
    numberOfThetas: int
        Sampling resolution in the elevation angle
    maskResult: bool
        Indicates whether the result will be cropped using the cranial base landmarks or not. Only the first and fourth landmark are used for masking.

    Returns
    -------
    vtkPolyData
        The spherical map as a flat surface model
    """

    # Creating a copy of the input models
    a = vtk.vtkPolyData()
    a.DeepCopy(inputMesh)
    inputMesh = a

    # Reading the reference/template landmarks 
    landmarks = DataSet.ReadPolyData(GLABELLA_CRANIALBASE_LANDMARKS_PATH)

    meshBounds = np.zeros([6], dtype=np.float64)
    inputMesh.GetBounds(meshBounds)
    center = np.zeros((3), dtype=np.float32)
    
    # Calculating the plane to mask out the information below the cranial base landmarks
    # Important: We only cut using the first and fourth landmark!
    landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):
        landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))

    dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :] # Left to right vector in the dursum sellae
    dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

    p0 = landmarkCoords[0, :] + 10 * dorsumVect # Two points in the forehead
    p1 = landmarkCoords[0, :] - 10 * dorsumVect #
    p2 = landmarkCoords[3, :] # the opisthion

    v0 = p2 - p1
    v1 = p2 - p0
    n = np.cross(v0, v1)
    n = n / np.sqrt(np.sum(n**2))
    
    plane = vtk.vtkPlane()
    plane.SetNormal(-n)
    plane.SetOrigin(p2)

    # Sampling
    sphericalMesh = vtk.vtkPolyData()
    sphericalMesh.SetPoints(vtk.vtkPoints())

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(inputMesh)
    obbTree.BuildLocator()

    intersectionPoints = vtk.vtkPoints()
    intersectionCellIds = vtk.vtkIdList()

    thetaDistance = np.pi / (numberOfThetas - 1.0)

    rayEnd = np.zeros([3], dtype=np.float64)
    radius = np.sqrt( (meshBounds[1] - meshBounds[0])**2 + (meshBounds[3] - meshBounds[2])**2 + (meshBounds[5] - meshBounds[4])**2 )
    intersectedPoint = np.zeros([3], dtype=np.float64)

    ## Adding all arrays with one component
    arrayList = []
    for id in range(inputMesh.GetCellData().GetNumberOfArrays()):

        if inputMesh.GetCellData().GetArray(id).GetNumberOfComponents() == 1:
            newArray = vtk.vtkFloatArray()
            newArray.SetName(inputMesh.GetCellData().GetArray(id).GetName())
            newArray.SetNumberOfComponents(1)
            arrayList += [newArray]

    # Adding the array with reconstruction information
    newArray = vtk.vtkFloatArray()
    newArray.SetName('coords')
    newArray.SetNumberOfComponents(3)
    arrayList += [newArray]
    
    # This speeds up finding the intersecitons when doing ray-casting
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    triangle = vtk.vtkGenericCell() # Allocating memory

    # Spherical coordinates of the first (glabella or nasion) and fourth (opisthion) landmarks
    nasion = np.array(landmarks.GetPoint(0))
    dist = np.sqrt(np.sum(nasion**2))

    nasionSinTheta =  (nasion[2] - center[2])/dist
    nasionSinPhi = (nasion[1] - center[1]) / (dist * np.sqrt( (1 - nasionSinTheta**2) ) )

    opisthion = np.array(landmarks.GetPoint(3))
    dist = np.sqrt(np.sum(opisthion**2))

    opisthionSinTheta =  (opisthion[2] - center[2])/dist
    opisthionSinPhi = (opisthion[1] - center[1]) / (dist * np.sqrt( (1 - opisthionSinTheta**2) ) )

    for latitude in range(numberOfThetas): # elevation sampling

        theta = np.pi/2 - latitude*np.pi/(numberOfThetas-1) # Elevation angle

        thetaLongitude = 2.0*np.pi*np.cos(theta)
        nPointsAtTheta = int(np.floor(thetaLongitude / thetaDistance))

        for longitude in range(nPointsAtTheta): # Azimuth sampling
            if nPointsAtTheta != 1:
                phi = -np.pi + longitude*2*np.pi / (nPointsAtTheta - 1) # Azimuth angle
            else:
                phi = 0


            distToNasion = (nasionSinPhi-np.sin(phi))**2
            distToOpisthion = (opisthionSinPhi-np.sin(phi))**2
            
            limitTheta = nasionSinTheta * distToOpisthion/(distToNasion + distToOpisthion) + opisthionSinTheta * distToNasion/(distToNasion + distToOpisthion)


            if not maskResult or np.sin(theta) >= limitTheta:
            
                # Ray-casting
                rayEnd[0] = center[0] + radius*np.cos(phi)*np.cos(theta)
                rayEnd[1] = center[1] + radius*np.sin(phi)*np.cos(theta)
                rayEnd[2] = center[2] + radius*np.sin(theta)

                if obbTree.IntersectWithLine(center, rayEnd, intersectionPoints, intersectionCellIds):
                    closestDist = np.inf
                    closestId = 0
                    for p in range(intersectionPoints.GetNumberOfPoints()):

                        intersectionPoints.GetPoint(p, intersectedPoint)
                        dist = np.sqrt((center[0] - intersectedPoint[0])**2 + (center[1] - intersectedPoint[1])**2 + (center[2] - intersectedPoint[2])**2 )
                    
                        if dist < closestDist:
                            closestDist = dist
                            closestId = p

                    intersectionPoints.GetPoint(closestId, intersectedPoint)
                    # Intersected point has the Euclidean coordinates of the points at the specific longitude and latitude

                    # Finding the color
                    pCoords = np.zeros([3], dtype=np.float64)
                    weights = np.zeros([3], dtype=np.float64)
                    subId = vtk.reference(0)
                    cellId = cellLocator.FindCell(intersectedPoint, 0, triangle, pCoords, weights)

                    #Calculating the cartesian coordinates for the BullsEye
                    angle = phi
                    rho = float(latitude)/(numberOfThetas-1)

                    x = rho * np.cos(phi)
                    y = rho * np.sin(phi)

                    if maskResult:
                        toDraw = plane.EvaluateFunction(intersectedPoint) > 0.1
                    else:
                        toDraw = True

                    if toDraw:

                        sphericalMesh.GetPoints().InsertNextPoint(x, y, 0)

                        for arrayId in range(len(arrayList)-1):
                    
                            if cellId >= 0:
                                arrayList[arrayId].InsertNextTuple1(inputMesh.GetCellData().GetArray(arrayList[arrayId].GetName()).GetTuple1(cellId))
                            else:
                                arrayList[arrayId].InsertNextTuple1(0)

                        arrayList[-1].InsertNextTuple3(intersectedPoint[0], intersectedPoint[1], intersectedPoint[2])

    for thisArray in arrayList:
        sphericalMesh.GetPointData().AddArray(thisArray)

    # Triangulating the points
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(sphericalMesh)
    filter.Update()
    sphericalMesh = filter.GetOutput()

    return sphericalMesh

def CreateVectorImageFromBullsEyeMesh(inputMesh, arrayName='coords', imageSize=500, imageExtent=2.0):
    """
    Creates a 2D image of the spherical map model of the cranial bone surface, using a vector array from the model

    Parameters
    ----------
    inputMesh: vtkPolyData
        Spherical map model
    arrayName: string
        name of the array in the model to use to create the image 
    imageSize: int
        Resolution in pixels of the image created
    imageExtent: int
        Extent of the coordinates of the image. The image coordinates are [-imageExtent/2.0, imageExtent/2)
    Returns
    -------
    np.array
        An 2D image of the array with arrayName in the spherical map
        The background in the image is set to -1
    """

    colorArray = inputMesh.GetPointData().GetArray(arrayName)
    nComponents = colorArray.GetNumberOfComponents()
    
    image = sitk.Image([imageSize, imageSize], sitk.sitkVectorFloat32, nComponents)
    image.SetOrigin([-imageExtent/2.0, -imageExtent/2.0])
    image.SetSpacing([imageExtent/(imageSize-1), imageExtent/(imageSize-1)])

    # Setting -1 for areas without information
    zeroCoords = [-1] * nComponents    

    closestCoords = np.zeros([3], dtype=np.float32)
    pCoords = np.zeros([3], dtype=np.float32)
    w = np.zeros([3], dtype=np.float32)

    # Cell locator
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    triangle = vtk.vtkGenericCell()

    for x in range(imageSize):
        for y in range(imageSize):

            xCoords = image.GetOrigin()[0] + image.GetSpacing()[0] * x
            yCoords = image.GetOrigin()[1] + image.GetSpacing()[1] * y

            coords = (xCoords, yCoords, 0)

            #cellId = inputMesh.FindCell(coords, None, 0, 0.1, subId, pCoords, w)
            cellId = cellLocator.FindCell(coords, 0, triangle, pCoords, w)
            if cellId >= 0:
                pointId = inputMesh.FindPoint(coords)
                if pointId >= 0:
                    image[x,y] = colorArray.GetTuple(pointId)
                else:
                    image[x,y] = zeroCoords
            else:
                image[x,y] = zeroCoords

    return image

def GenerateSphericalMapOfData(externalSurface, subjectToTemplateTransform):
    '''
    Function to generate the spherical map of a surface mesh
    Will cut the mesh at the template space landmarks (tragion)
    '''
    l = vtk.vtkPolyData()
    l.DeepCopy(externalSurface)
    externalSurface = l

    template_space_photo = ApplyTransform(externalSurface, subjectToTemplateTransform)

    spherical_model = CreateSphericalMapFromSurfaceModel(template_space_photo, maskResult = False)
    coordsImage = CreateVectorImageFromBullsEyeMesh(spherical_model)
    return coordsImage

def RemoveScale(subjectData, indices):
    image_array = sitk.GetArrayFromImage(subjectData)[indices[:,0], indices[:,1],:]

    #mean euclidean distance of each point
    scale_factor = np.mean(np.sum(np.power(image_array, 2),axis = 1)**0.5)
    outimage = sitk.GetImageFromArray(sitk.GetArrayFromImage(subjectData)/scale_factor, isVector = True)
    outimage.CopyInformation(subjectData)
    return outimage, scale_factor

def CalculateFeaturesFromCoefficients(subject_data, coef, std_model, fun):
    '''
    Normalize the coefficients calculated from the PCA model
    Use these features to compute HSA index or fit a CRS model
    '''
    feature_data = pd.DataFrame(columns = subject_data.filter(like = 'PC').columns.values)
    for idx,column in enumerate(subject_data.filter(like = 'PC').columns.values):
        data = subject_data[column].astype(float).to_numpy()
        age = subject_data['Age'].astype(float).astype(int).to_numpy()
        sex = subject_data['Sex'].astype(float).astype(int).to_numpy()
        mean = fun(age, sex, *coef[idx, :])[0]
        std = std_model[idx].predict(np.stack([age,sex], axis = 1).reshape(-1,2))
        # feature_data = pd.concat([feature_data, pd.DataFrame(data = (data - mean) / std, columns = [column])],axis = 1, ignore_index=True)
        feature_data[column] = (data - mean) / std
    feature_data = feature_data.copy()
    feature_data[['Age','Sex']] = subject_data[['Age','Sex']].values
    return feature_data

def CalculateModelCoeffients(subjectData, model, constrainCoefficients=True, stdRange=3):
    """
    Creates a PCA-based statistical shape model. Data = average + coefficents @ components

    Parameters
    ----------
    subjectData: np.array (nFeatures), or (1, nFeatures)
        The data
    model: dict {} ->
        average: np.array (1, nFeatures)
            The model average
        components: np.array (nComponents, nFeatures)
            The principal components
        coefficients: np.array (nSamples, nComponents)
            The coefficients of each sample in the model space
        variance: np.array (nComponents)
            The % of the total variance explained by each component
    constrainCoefficients: bool
        Indicates if the coefficients of the subject will be constrained
    stdRange: float
        If constrainCoefficients is True, indicates how many standard deviations from the mean the coefficients will be constrained

    Returns
    -------
    np.array (nComponents)
        The coefficients calculated
    """
    subjectData = subjectData.reshape((1,-1))

    subjectCoefficients = (subjectData - model['average']) @ model['components'].T

    # Constraining coefficients
    if constrainCoefficients:
        std = np.std(model['coefficients'], axis=0)

        for c in range(model['variance'].size):
            subjectCoefficients[:, c] = np.clip(subjectCoefficients[:, c], -stdRange*std[c], stdRange*std[c])

    return subjectCoefficients.reshape((1,-1))

def ComputeResults(features, variance, num_comps = 35):
    #now compute the HSA index
    statisticalDistance = np.sum(np.abs(features.filter(like = 'PC').iloc[:,:num_comps].values * variance[:num_comps]), axis = 1) / np.sum(variance[:num_comps])

    with shelve.open(os.path.join(CLASSIFIER_DIR,'SVM_Trained_Data_Final'), 'r') as file:
        # selector= file['selector']
        feature_indices = np.array(file['feature_indices'])
        clf = file['clf']
        scaler = file['scaler']
        file.close()

    features_transformed = scaler.transform(features.iloc[:,feature_indices].values) 
    return 100*clf.predict_proba(features_transformed)[0][1], statisticalDistance[0]

def Arcsinh(age, sex, a, b, c, d, e, g):
    y = a + b*sex + (c)* np.arcsinh(d*age) + e*age + g*age*sex
    return y, [1, sex, np.arcsinh(d*age), c*(age+d*age**2/np.sqrt(1+d**2*age**2))/(d*age+np.sqrt(1+d**2*age**2)), age, age*sex]

def ComputeFromSphericalImage(coordsImage, age, sex, remove_scale = False):
    atlasPath = os.path.join(MODEL_DIR, 'PCAModel')
    with shelve.open(atlasPath, 'r') as atlasInformation:
        indices = atlasInformation['indices']
        model = atlasInformation['model']

    # extract subject data
    if remove_scale:
        coordsImage, _ = RemoveScale(coordsImage, indices)
    subjectData = sitk.GetArrayFromImage(coordsImage)[indices[:,0], indices[:,1],:].ravel()
    coefficients = CalculateModelCoeffients(subjectData, model, constrainCoefficients=False)

    ####now let's normalize the PC coefficients
    with shelve.open(os.path.join(MODEL_DIR, 'PCAModel')) as atlasInformation:
        indices = atlasInformation['indices']
        model = atlasInformation['model']

    with shelve.open(os.path.join(MODEL_DIR, 'PCAModel-Parameters')) as parametersModel:
        mean_coef = parametersModel['mean_params']
        std_model = parametersModel['std_model']
        variance_norm_only = parametersModel['variance']
    cols = ['PC%s'%x for x in np.arange(1,len(coefficients[0])+1)]
    df = pd.DataFrame(data = coefficients, columns = cols)
    if type(sex)==str:
        sex = int(sex=='M')
    df['Age'] = age
    df['Sex'] = sex
    df = CalculateFeaturesFromCoefficients(df, mean_coef, std_model, Arcsinh)

    return ComputeResults(df, variance_norm_only, num_comps = len(variance_norm_only))
