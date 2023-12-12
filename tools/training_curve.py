from training_curve_model import *
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['yolov7_inSeg', 'cascade_inSeg_mm'],
                        help="Input model type.")
    parser.add_argument('--result_dir', type=str, required=True,
                        help="Enter the path to the results folder.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Save the training curve in this directory.")
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    if args.model_type == 'yolov7_inSeg':
        trainCurve = yolov7_inSeg(result_dir=args.result_dir,
                                  output_dir=args.output_dir)
    # elif args.model_type == 'cascade_inSeg_mm':
    #     trainCurve = cascade_inSeg_mm(result_dir=args.result_dir,
    #                                   output_dir=args.output_dir)
    trainCurve.generate_curve()
