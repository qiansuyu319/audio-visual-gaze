import argparse
import os
import opensmile
import soundfile as sf
import numpy as np

def extract_egemaps(input_wav, output_npy):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(input_wav)
    np.save(output_npy, features.values)
    print(f'eGeMAPS features saved to {output_npy}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--output_npy', required=True)
    args = parser.parse_args()
    extract_egemaps(args.input_wav, args.output_npy)

if __name__ == '__main__':
    main() 