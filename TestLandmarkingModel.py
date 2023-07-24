from ChebyModel.DataSetGraph import ReadPolyData, WritePolyData
from LandmarkingUtils import RunInference
from Analyze3DPhotogram import ValidVTP
import argparse
from os import path

def ConstructArguments():
    parser = argparse.ArgumentParser(description='Process 3D photogram')
    ## Required arguments
    parser.add_argument('--input_filename', metavar = 'input_filename', required = True, type = ValidVTP,
        help='Input data path')

    parser.add_argument('--crop_image',action='store_true', help = 'Option to crop the data to ensure the shoulders are not included in the photogram.')
    return parser

def ParseArguments():
    parser = ConstructArguments()
    return parser.parse_args()
    
if __name__ == "__main__":
    #define the path to the data
    args = ParseArguments()
    #VTK reader
    image = ReadPolyData(args.input_filename)
    #run the inference
    landmarks, cropped_image = RunInference(image, crop = args.crop_data, return_cropped_image=True)
    WritePolyData(landmarks,path.join(path.dirname(args.input_filename),'testlandmarks.vtp'))
    WritePolyData(cropped_image,path.join(path.dirname(args.input_filename),'testcropped_image.vtp'))