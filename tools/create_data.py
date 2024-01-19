from __future__ import annotations
from converter import *
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Dataset process script', add_help=False)

    parser.add_argument('--source_dir', type=str,
                        help="""The dataset's path includes image files and json files.
                                Enter the split test or train data set.""")
    parser.add_argument('--output_dir', type=str,
                        help="Save the result in this directory.")
    parser.add_argument('--classes_yaml', type=str, required=True,
                        help='The category of training needs a YAML file.')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'test'], required=True,
                        help='For training dataset or testing dataset.')
    parser.add_argument('--format', type=str, choices=['coco', 'cocoStuff', 'yoloSeg', 'yoloBbox', 'mvtec'],
                        required=True,
                        help='Which output format do you want?')
    parser.add_argument('--patch_size', type=int, choices=[256, 512, 1024],
                        help='The size of the patch needs to be divisible by width and height. '
                             'If you assign the value, the script will generate a patch dataset')
    parser.add_argument('--stride', type=int, choices=[1, 2], default='1',
                        help='A moving step, stride=1 means no overlap, stride=2 means overlap and step=1/2.')
    parser.add_argument('--store_none', action="store_true",
                        help="Whether to save none anomaly patch.")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset process script. ',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    if args.format == 'coco':
        conv = cocoConverter(source_dir=args.source_dir,
                             output_dir=args.output_dir,
                             classes_yaml=args.classes_yaml,
                             dataset_type=args.dataset_type,
                             format=args.format,
                             patch_size=args.patch_size,
                             stride=args.stride,
                             store_none=args.store_none)
    elif args.format == 'cocoStuff':
        conv = cocoStuffConverter(source_dir=args.source_dir,
                                  output_dir=args.output_dir,
                                  classes_yaml=args.classes_yaml,
                                  dataset_type=args.dataset_type,
                                  format=args.format,
                                  patch_size=args.patch_size,
                                  stride=args.stride,
                                  store_none=args.store_none)

    elif args.format == 'yoloSeg':
        conv = yoloSegConverter(source_dir=args.source_dir,
                                output_dir=args.output_dir,
                                classes_yaml=args.classes_yaml,
                                dataset_type=args.dataset_type,
                                format=args.format,
                                patch_size=args.patch_size,
                                stride=args.stride,
                                store_none=args.store_none)
    elif args.format == 'yoloBbox':
        conv = yoloBboxConverter(source_dir=args.source_dir,
                                 output_dir=args.output_dir,
                                 classes_yaml=args.classes_yaml,
                                 dataset_type=args.dataset_type,
                                 format=args.format,
                                 patch_size=args.patch_size,
                                 stride=args.stride,
                                 store_none=args.store_none)
    elif args.format == 'mvtec':
        conv = mvtecConverter(source_dir=args.source_dir,
                              output_dir=args.output_dir,
                              classes_yaml=args.classes_yaml,
                              dataset_type=args.dataset_type,
                              format=args.format,
                              patch_size=args.patch_size,
                              stride=args.stride,
                              store_none=args.store_none)
    else:
        raise ValueError("Can not find the converter of {}.".format(args.format))

    if args.patch_size == None:
        conv.generate_original()
    else:
        conv.generate_patch()
