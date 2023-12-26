from training_curve_model import *
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Yolov7inSeg', 'Yolov7Obj'],
                        help="Input model type.")
    parser.add_argument('--result_path', type=str, required=True,
                        help="Enter the path to the results folder.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Save the training curve in this directory.")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    if args.model_type == 'Yolov7inSeg':
        trainCurve = yolov7_inSeg(result_path=args.result_path,
                                  output_path=args.output_path)
    elif args.model_type == 'Yolov7Obj':
        trainCurve = yolov7_Obj(result_path=args.result_path,
                                output_path=args.output_path)
    trainCurve.generate_curve()
