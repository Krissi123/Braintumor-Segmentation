import glob
import os
import shutil
import pandas as pd

class EnvNotSetError(Exception):
    pass

try:
    from modules.scandata import (
        MriScan, 
        MriSlice, 
        TumourSegmentation, 
        ScanType, 
        ScanPlane, 
        PatientRecord,
    )
except: 
    raise EnvNotSetError(
        "Cannot import local modules.\n"
        " - Run `source prepare_enviromnent` in the top-level of this repository\n"
    )


def prepare_dataset():
    """
    Prepare simple dataset for only 20 patients
    Used for testing only
    Creates PNG files in data directories that separate MRI slices into tumour/healthy
    dependent on whether they contain tumour core within the segmentation file.
    Created files contain all 4 sequences in the RGBA channels of a 4 channel PNG.
    """
    base_dir = os.getenv('BASEDIR')
    if not base_dir:
        raise EnvNotSetError(
            "Environment variables not set.\n"
            " - Run `source prepare_enviromnent` in the top-level of this repository\n"
        )

    data_dir = os.path.join(base_dir,'data','mri-datasets','first-20-testset','dataset_multiseq')
    tumour_data_dir = os.path.join(data_dir, 'tumour')
    healthy_data_dir = os.path.join(data_dir, 'healthy')
    try:
        print(f'Creating directory:\n{data_dir}')
        os.mkdir(data_dir)
    except FileExistsError:
        print('Directory exists')
    try:
        os.mkdir(tumour_data_dir)
    except FileExistsError:
        print('Directory exists')
    try:
        os.mkdir(healthy_data_dir)
    except FileExistsError:
        print('Directory exists')

    for patient in range(1,21):

        T1_scan = MriScan(
            filename=f'data/mri-datasets/first-20-testset/images_structural/UPENN-GBM-{patient:05}_11/UPENN-GBM-{patient:05}_11_T1.nii.gz',
            sequence=ScanType.T1
        )
        T1CE_scan = MriScan(
            filename=f'data/mri-datasets/first-20-testset/images_structural/UPENN-GBM-{patient:05}_11/UPENN-GBM-{patient:05}_11_T1GD.nii.gz',
            sequence=ScanType.T1CE
        )
        T2_scan = MriScan(
            filename=f'data/mri-datasets/first-20-testset/images_structural/UPENN-GBM-{patient:05}_11/UPENN-GBM-{patient:05}_11_T2.nii.gz',
            sequence=ScanType.T2
        )
        FLAIR_scan = MriScan(
            filename=f'data/mri-datasets/first-20-testset/images_structural/UPENN-GBM-{patient:05}_11/UPENN-GBM-{patient:05}_11_FLAIR.nii.gz',
            sequence=ScanType.FLAIR
        )
        segmentation = TumourSegmentation(
            filename=f'data/mri-datasets/first-20-testset/automated_segm/UPENN-GBM-{patient:05}_11_automated_approx_segm.nii.gz',
            )

        patient_data = PatientRecord()
        patient_data.add_scan_data(T1_scan)
        patient_data.add_scan_data(T1CE_scan)
        patient_data.add_scan_data(T2_scan)
        patient_data.add_scan_data(FLAIR_scan)
        patient_data.add_segmentation(segmentation)

        patient_data.save_multi_channel_png(os.path.join(data_dir, f'UPENN-GBM-{patient:05}_11_multiseq'), [ScanType.T2,ScanType.T1CE, ScanType.FLAIR,ScanType.T1])


    data_list = []
    for auto_seg in glob.glob('data/mri-datasets/first-20-testset/automated_segm/UPENN-GBM-*_11_automated_approx_segm.nii.gz'):
        patient = auto_seg.removeprefix('data/mri-datasets/first-20-testset/automated_segm/UPENN-GBM-')[:5]
        segmentation = TumourSegmentation(auto_seg)
        for idx, slice in enumerate(segmentation.iterate_slices()):
            data_list.append([patient, idx, 1 if 1 in slice.slice_data  else 0])

    df = pd.DataFrame(data_list, columns=['patient', 'slice', 'tumour_present'])

    df.tumour_present.value_counts()

    # Arrange data set in directories for different classes
    for idx, row in df.iterrows():
        patient_number = row['patient']
        slice_number = row['slice']
        filename = f'UPENN-GBM-{int(patient_number):05}_11_multiseq_{int(slice_number):03}.png'
        original_file = os.path.join(data_dir, filename)
        if row['tumour_present']:
            new_file = os.path.join(tumour_data_dir, filename)
        else:
            new_file = os.path.join(healthy_data_dir, filename)
        shutil.move(original_file, new_file)
        

if __name__ == '__main__':
    prepare_dataset()
    