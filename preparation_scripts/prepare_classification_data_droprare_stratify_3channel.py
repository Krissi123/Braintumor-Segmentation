import glob
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


try:
    from modules.scandata import (
        MriScan, 
        TumourSegmentation, 
        ScanType, 
        PatientRecord,
    )
    from modules.exceptions import (
        ScanDataDirectoryNotFound, 
        ScanFileNotFound,
        EnvNotSetError
    )
except ModuleNotFoundError: 
    print(
        "Cannot import local modules.\n"
        " - Run `source prepare_enviromnent` in the top-level of this repository\n"
    )
    raise
 
# Seed for test/train split and dropping images for undersampling background cases
RSEED=78 
# Set fraction to keep for test set 
TEST_SIZE=0.1
 
def prepare_dataset():
    """
    Prepare dataset with patients stratified into train and test sets by age and gender
    
    Creates PNG files in data directories that separate MRI slices into types 
    dependent on what segmentation regions are present in the segmentation file.

    Slice classes with less than 100 observations are discarded from the analysis

    Created files contain all 4 sequences in the RGBA channels of a 4 channel PNG.
    """
    base_dir = os.getenv('BASEDIR')
    
    if not base_dir:
        raise EnvNotSetError(
            "Environment variables not set.\n"
            " - Run `source prepare_enviromnent` in the top-level of this repository\n"
        )
    
    # Create data frame of patients data files
    raw_data_dir = os.path.join(base_dir, 'data', 'UPENN-GBM')
    raw_scan_data_dir = os.path.join(raw_data_dir, 'images_structural')
    raw_segmentation_dir = os.path.join(raw_data_dir, 'automated_segm')
    print("Raw data directory: ", raw_data_dir)
    print("Raw scan data directory: ", raw_scan_data_dir)
    print("Raw segmentation directory: ", raw_segmentation_dir)
    print("\n")
     
    for dir in raw_scan_data_dir, raw_segmentation_dir:
        if not os.path.exists(dir):
            raise ScanDataDirectoryNotFound(f'{dir} does not exist')

    raw_files_list = []

    # Only work on preop patient data -- identifiers end *_11
    for patient_scan_dir in glob.glob(os.path.join(raw_scan_data_dir, 'UPENN*11')):
        patient_identifier = os.path.basename(patient_scan_dir)
        t1_filename = os.path.join(patient_scan_dir, f'{patient_identifier}_T1.nii.gz')
        t1ce_filename = os.path.join(patient_scan_dir, f'{patient_identifier}_T1GD.nii.gz')
        t2_filename = os.path.join(patient_scan_dir, f'{patient_identifier}_T2.nii.gz')
        FLAIR_filename = os.path.join(patient_scan_dir, f'{patient_identifier}_FLAIR.nii.gz')
        seg_filename = os.path.join(
            raw_segmentation_dir, f'{patient_identifier}_automated_approx_segm.nii.gz'
        )

        patient_raw_files = [ 
                patient_identifier, 
                patient_scan_dir,
                t1_filename,
                t1ce_filename,
                t2_filename,
                FLAIR_filename,
                seg_filename,
        ]
        
        for file in patient_raw_files[-5:]:
            if not os.path.exists(file):
                raise ScanFileNotFound(
                    f'{file} does not exist'
                )
        raw_files_list.append(patient_raw_files)

    df_patient_files = pd.DataFrame(raw_files_list, columns=[ 
                'patient_identifier', 
                'patient_scan_dir',
                't1_filename',
                't1ce_filename',
                't2_filename',
                'FLAIR_filename',
                'seg_filename',
    ])   


    # Find all possible slice segmentation types in all patients slices
    seg_volumes = {
        0: 'background',  # either healthy brain tissue or nothing
        1: 'tumour', 
        2: 'edema',
        4: 'contrast',
    }

    slice_list = []
    print("Scanning all segmentation files in dataset to find slice types")
    for idx, row in enumerate(df_patient_files.itertuples()):
    #for seg_filename in df_patient_files.seg_filename:
        segmentation = TumourSegmentation(row.seg_filename)
        for slice_number , seg_slice in enumerate(segmentation.iterate_slices()):
            slice_list.append([
                row.patient_identifier,
                slice_number,
                '_'.join(
                    [seg_volumes[x] for x in sorted(
                        list(set(seg_slice.slice_data.flatten()))
                    )]
                )
            ])
        if idx%50 == 0 and idx != 0:
            print(f"\t{idx} segmentation files complete")
    print(f"\t{idx+1} segmentation files complete")
            

    df_slices = pd.DataFrame(slice_list, columns=['patient_identifier', 'slice_number', 'slice_class'])

    print (
        '\nSlice classes found in dataset:\n',
        df_slices.slice_class.unique()
    )

    # Get clinical data to stratify by age and gender
    df_clinical = pd.read_csv(
        os.path.join(raw_data_dir,'table_data','UPENN-GBM_clinical_info_v1.0.csv')
    )
    df_clinical['age_bin'] = '<40'
    df_clinical.loc[df_clinical['Age_at_scan_years']>=40, 'age_bin'] = '40-50'
    df_clinical.loc[df_clinical['Age_at_scan_years']>=50, 'age_bin'] = '50-60'
    df_clinical.loc[df_clinical['Age_at_scan_years']>=60, 'age_bin'] = '60-70'
    df_clinical.loc[df_clinical['Age_at_scan_years']>=70, 'age_bin'] = '70-80'
    df_clinical.loc[df_clinical['Age_at_scan_years']>=80, 'age_bin'] = '>80'
    df_clinical['stratify_class'] = df_clinical['Gender'] + df_clinical['age_bin']
    df_patient_files = pd.merge(
        df_patient_files, 
        df_clinical[['ID','stratify_class']], 
        how='left', 
        left_on='patient_identifier', 
        right_on='ID'
    ).drop('ID', axis=1)

    # Allocate patient data to train or test set
    patients = df_patient_files.patient_identifier
    train_patients, test_patients = train_test_split(
        patients, 
        test_size=TEST_SIZE, 
        random_state=RSEED,
        stratify=df_patient_files.stratify_class,
    )
    df_train_patients = pd.DataFrame(train_patients)
    df_test_patients = pd.DataFrame(test_patients)
    df_train_patients['test_train_set'] = 'train'
    df_test_patients['test_train_set'] = 'test'

    df_slices = df_slices.merge(
        pd.concat([df_test_patients, df_train_patients]), 
        on='patient_identifier', 
        how='left',
    )


    # Create classification directories
    classification_data_dir = os.path.join(raw_data_dir, 'slice_classification_common_stratify_3channel')
    print(f"Classification data directory: {classification_data_dir}")
    try:
        os.mkdir(classification_data_dir)
    except:
        print(f'{classification_data_dir} already exists')

    # Split into test and train
    for data_set in 'test', 'train':
        data_set_dir = os.path.join(classification_data_dir, data_set)
        try:
            os.mkdir(data_set_dir)
        except:
            print(f'{data_set_dir} already exists')

        for slice_class in df_slices.slice_class.unique():
            class_dir = os.path.join(data_set_dir, slice_class)
            print(f"\tClass directory: {class_dir}")
            try:
                os.mkdir(class_dir)
            except:
                print(f'{class_dir} already exists')
                

    print("Creating slice PNG files")
    for idx, row in enumerate(df_patient_files.itertuples()):
        T1_scan = MriScan(
            filename=row.t1_filename,
            sequence=ScanType.T1
        )
        T1CE_scan = MriScan(
            filename=row.t1ce_filename,
            sequence=ScanType.T1CE
        )
        T2_scan = MriScan(
            filename=row.t2_filename,
            sequence=ScanType.T2
        )
        #FLAIR_scan = MriScan(
        #    filename=row.FLAIR_filename,
        #    sequence=ScanType.FLAIR
        #)
        segmentation = TumourSegmentation(
            row.seg_filename,
            )
        patient_data = PatientRecord()
        patient_data.add_scan_data(T1_scan)
        patient_data.add_scan_data(T1CE_scan)
        patient_data.add_scan_data(T2_scan)
        #patient_data.add_scan_data(FLAIR_scan)
        patient_data.add_segmentation(segmentation)
        png_basename_prefix = f'{row.patient_identifier}_allseq'
        patient_data.save_multi_channel_png(os.path.join(classification_data_dir,png_basename_prefix))
        if idx%50 == 0 and idx != 0:
            print(f"\t{idx} patient's files complete")
    print(f"\t{idx+1} patient's files complete")

    df_slice_class_counts = df_slices.query(
        'test_train_set == "train"'
    ).slice_class.value_counts().to_frame().reset_index().set_axis(
        ['slice_class', 'slice_class_count'], axis=1
    )
    df_slices =  pd.merge(df_slices,df_slice_class_counts, how='left')

    print(
        "Full dataset slice class observations:\n",
        df_slices.query('test_train_set == "train"').slice_class.value_counts()
    )
    # Background only slices most common and least useful for training. 
    # Take only 30 % in training sample data set and drop classes with less than 100 slices
    df_sliced_dropped_background =  df_slices.query(
            'test_train_set == "train" and slice_class == "background"'
    ).sample(frac=0.70, random_state=RSEED)
    df_sliced_dropped_rare = df_slices.query('slice_class_count < 100')
    df_slices_after_drop = df_slices.drop(df_sliced_dropped_background.index)
    df_slices_after_drop = df_slices_after_drop.drop(df_sliced_dropped_rare.index)
    print(
        "After dropping 30%% of background and classes with less than 100 observations:\n",
        df_slices_after_drop.query('test_train_set == "train"' ).slice_class.value_counts()
    )


    # Move data to classification directories
    for idx, row in enumerate(df_slices_after_drop.itertuples()):
        src_file = os.path.join(
            classification_data_dir,
            f'{row.patient_identifier}_allseq_{row.slice_number:03}.png'
        )
        dest_dir = os.path.join(classification_data_dir,row.test_train_set,row.slice_class)
        shutil.move(src_file,dest_dir)
    # Delete dropped  files to clean up
    for frame in df_sliced_dropped_background, df_sliced_dropped_rare:
        for idx, row in enumerate(frame.itertuples()):
            del_file = os.path.join(
                classification_data_dir,
                f'{row.patient_identifier}_allseq_{row.slice_number:03}.png'
            )
            os.remove(del_file)

    # Delete unused class directories
    for data_set in 'test', 'train':
        data_set_dir = os.path.join(classification_data_dir, data_set)
        for slice_class in df_sliced_dropped_rare.slice_class.unique():
            rm_dir = os.path.join(data_set_dir,slice_class)
            print(f"Removing {rm_dir} as classes dropped")
            try:
                os.rmdir(rm_dir)
            except:
                print(f'{rm_dir} does not exist')
            

if __name__ == '__main__':
    prepare_dataset()


