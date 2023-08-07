from tools.LandmarkingUtils import RunInference
from Analyze3DPhotogram import ReadImage
import time
from tools.PhotoAnalysisTools import AlignPatientToTemplate, GenerateSphericalMapOfData, ComputeFromSphericalImage
from Analyze3DPhotogram import ParseArguments

if __name__ == "__main__":
    #first, let's start with the landmarks
    #define the path to the data
    args = ParseArguments()
    age = args.age
    sex = args.sex
    #VTK xml reader
    start_time = time.time()
    image = ReadImage(args.input_filename)
    #run the inference
    landmarks = RunInference(image)
    landmark_time = time.time()
    output_mesh, transform = AlignPatientToTemplate(image, landmarks)
    alignment_time = time.time()
    spherical_image = GenerateSphericalMapOfData(output_mesh, transform)
    spherical_image_time = time.time()
    riskScore, HSA_index = ComputeFromSphericalImage(spherical_image, age, sex)
    metrics_time = time.time()

    #### Print all of our times! ####
    print(f'Started script at {time.strftime("%d %b %y %H:%M:%S", time.localtime(start_time))}')
    print(f'\tLandmarking took {landmark_time-start_time:0.4f} seconds.')
    print(f'\tAligning to the template took {alignment_time-landmark_time:0.4f} seconds.')
    print(f'\tGenerating the spherical model took {spherical_image_time-alignment_time:0.4f} seconds.')
    print(f'\tComputing the results took {metrics_time-spherical_image_time:0.4f} seconds.')
    print(f'Total time elapsed: {metrics_time-start_time:0.4f} seconds.')