"""
Provides classes for dealing with data from MRI scans
"""
from __future__ import annotations
from enum import Enum, IntEnum, auto
from collections.abc import Iterator
from typing import Union
from PIL import Image
import numpy as np
import nibabel as nib # type: ignore
from matplotlib.axes import Axes as PltAxes # type: ignore

class ScanType(Enum):
    """ 
    Enumerator for defining MRI sequence types
    
    Members:
        T1: T1 sequence
        T1CE: T1 contrast-enhanced sequence
        T2: T2 sequence
        FLAIR: FLAIR sequence
    """
    T1 = auto() 
    T1CE = auto()
    T2 = auto()
    FLAIR = auto()

class ScanPlane(IntEnum):   
    """ 
    Enumerator for defining MRI scan planes. Integers correspond to axes of 
    data array in NIfTI file that is constant in this plane, e.g. a transversal
    plane slice is obtained by fixing a constant value in axis 2 of data array.

    Members:
        SAGITTAL: sagital plane, axis 0
        FRONTAL: frontal plane, axis 1
        TRANSVERSAL: transversal plane, 2
    """
    SAGITTAL = 0
    FRONTAL = 1
    TRANSVERSAL = 2
    

class BaseScanData:
    """ 
    Base class for storing 3D data

    Attributes:
        load_nifti_(): loads data from NIfTI file into class instance.
        get_slice(): returns a single slice from a 3D data volume.
        interate_slices(): iterates though all slices in a given axis.
        save_png(): saves all slices in a given axis to png files. 
    """    

    def __init__(self, filename : Union[str, None] = None) -> None:
        """
        Initialise base class for storing scan data

        Args:
            filename (str, optional): If given, read a datafile on 
            initialisation. Defaults to None.
        """
        if filename:
            self.load_nifti(filename=filename)

    def load_nifti(self, filename : str) -> None:
        """
        Load nifti file containing scan data

        Args:
            filename (str): Name of NIfTI file to be read
            reorder (bool, optional): If true reorder array containing scan
            data so transverse slices can be enumerated. Defaults to True.
        """        
        
        self.scan_file = nib.load(filename)
        # Get integer size of data in file from header
        self.scan_bit_depth = self.scan_file.header['bitpix'] # bit size as integer
        self.data_type = f'uint{self.scan_bit_depth}'
        self.scan_data = self.scan_file.get_fdata().astype(self.data_type)
        self.scan_max_value = self.scan_data.max()

        self.get_png_parameters()

    def get_png_parameters(self):
        """
        Calculate parameters needed for writing scan data to PNG files
        """
        self.scaling_factor = (2**self.scan_bit_depth - 1) / self.scan_max_value
        # PIL image mode for saving images
        if self.scan_bit_depth <= 8:
            self.png_mode = 'L'
        elif self.scan_bit_depth <=16:
            self.png_mode = 'I;16'
        else:
            self.png_mode = None
        
    
    def get_slice(
        self,
        slice_number : int, 
        plane : ScanPlane = ScanPlane.TRANSVERSAL
    ) -> MriSlice:
        """
        Returns a slice of an MRI scan

        Args:
            slice_number (int): _description_
            plane (ScanPlane, optional): plane in which the slice should be 
            taken. Defaults to ScanPlane.TRANSVERSAL.

        Returns:
            MriSlice: object containing the slice data in an array
        """
        return MriSlice(
            np.take(self.scan_data,slice_number, plane),
            data_type = self.data_type,
            max_value = self.scan_max_value,
            scaling_factor = self.scaling_factor,
            png_mode = self.png_mode,

        )
            
    def iterate_slices(
        self, plane : ScanPlane = ScanPlane.TRANSVERSAL
    ) -> Iterator[MriSlice]:
        """
        Generates iterations of MRI slices taken in a specific plane.

        Args:
            plane (ScanPlane, optional): plane in which the slices should be
            taken. Defaults to ScanPlane.TRANSVERSAL.

        Yields:
            Iterator[MriSlice]: Objects containing MRI slices in the chosen 
            plane that are stored in arrays.
        """
        reordered_data = np.moveaxis(self.scan_data, plane, 0)
        for mri_slice in reordered_data:
            yield MriSlice(
                mri_slice, 
                data_type = self.data_type,
                max_value = self.scan_max_value,
                scaling_factor = self.scaling_factor,
                png_mode = self.png_mode,
            )

    def save_nifti(self, filename : str) -> None:
        """TODO - if needed"""
        pass

    def save_png(
        self, 
        file_prefix : str, 
        start_index : int = 0,
        plane : ScanPlane = ScanPlane.TRANSVERSAL,
        u16bit : bool = False,
    ) -> None:
        """
        Saves a set of PNG images of MRI slices taken in a plane.

        Args:
            file_prefix (str): Prefix used for PNG files. Files will be appended
            with _xxxx.png, where x is the number of the slice the png represents.
            start_index (int): Index number to use for the png file of the 
            first slice. Defaults to 0.
            plane (ScanPlane, optional): plane in which the slices are made.
            Defaults to ScanPlane.TRANSVERSAL.
        """
        # Find width required for zero padding number of png file
        width = np.ceil(
            np.log10(len(list(self.iterate_slices(plane=plane))))
            )
        
        for idx, slice_data in enumerate(self.iterate_slices(plane=plane)):
            slice_data.save_png(
                f'{file_prefix}_{start_index+idx:0{int(width)}}.png', 
            )
            

class MriScan(BaseScanData):
    """ 
    Stores information relating to one MRI scan extending BaseScanData class.

    Attributes:
        sequence (ScanType): MRI sequence used for the scan.
        load_nifti_(): loads data from NIfTI file into class instance.
        get_slice(): returns a single slice from a 3D data volume.
        interate_slices(): iterates though all slices in a given axis.
        save_png(): saves all slices in a given axis to png files. 
    """

    def __init__(
        self, 
        filename : Union[str, None] = None,
        sequence : Union[ScanType, None] = None, 
    ) -> None:
        """
        Initialise instance.

        Args:
            filename (Union[str, None], optional): If given, reads data from
            file on initialisation. Defaults to None.
            sequence (Union[ScanType, None], optional): If given, sets the MRI
            sequence used in the scan. Defaults to None.
        """
    
        self.sequence = sequence
        super().__init__(filename=filename)


class TumourSegmentation(BaseScanData):
    """
    Initialise instance.

    Attributes:
        load_nifti_(): loads data from NIfTI file into class instance.
        get_slice(): returns a single slice from a 3D data volume.
        interate_slices(): iterates though all slices in a given axis.
        save_png(): saves all slices in a given axis to png files. 
    """

    def __init__(
        self, 
        filename :  Union[str, None] = None,
        scale_png_data : bool = True,
    ) -> None:
        """
        Initialise class for storing segmentation data

        Args:
            filename (Union[str, None], optional): If given, reads data from
            file on initialisation. Defaults to None. 
        """

        self.scale_png_data = scale_png_data

        super().__init__(filename=filename)

    def get_png_parameters(self) -> None:
        """
        Overrides base function to give the option not to scale segmentation data
        """
        if self.scale_png_data:
            super().get_png_parameters()
        else:
            self.scaling_factor = 1
            self.png_mode = 'L'
        

class MriSlice:
    """
    Class to store information from a single slice of an MRI scan

    Attributes:
        save_png(): Saves MRI slice data to a PNG image.
        add_plot_to_axis(): Adds a bitmap plot of the data stored in the image 
        to an instance of a matplotlib axis.
        
    """

    def __init__(
        self, 
        slice_data : np.ndarray,
        data_type : Union[str, None] = None,
        max_value : Union[int, None] = None,
        scaling_factor : Union[float, None] = None,
        png_mode : Union[str, None] = None, 
        
    ) -> None:
        """
        Initialise instance

        Args:
            slice_data (np.ndarray): Array containing data for the slice.
        """
        self.slice_data = np.array(slice_data)
        if data_type:
            self.data_type = data_type
        else:
            self.data_type = self.slice_data.dtype
        if max_value:
            self.max_value = max_value
        else:
            self.max_value = self.slice_data.max()
        if scaling_factor:
            self.scaling_factor = scaling_factor
        else:
            self.scaling_factor = self.max_value / np.dtype(self.data_type).itemsize
        if png_mode:
            self.png_mode = png_mode
        else:
            png_mode = None

    def save_png(
        self, 
        filename : str, 
    ) -> None:
        """
        Saves MRI slice data to a PNG image.

        Args:
            filename (str): Name to PNG file that is written.
            normalisation_factor (int, optional): factor that each data point is 
            integer-divided by to give values in 16-bit range. If None, values are 
            scaled relative to maximum value in the array. Defaults to None.
        """

        bitmap = Image.fromarray(
            (
                self.slice_data 
                * int(self.scaling_factor)
            ).astype(self.data_type),
            mode=self.png_mode,
        )
        bitmap.save(filename)

    def add_plot_to_axis(self, ax : PltAxes, cmap : str = 'gist_gray') -> None:
        """
        Adds a bitmap plot of the data stored in the image to an instance of a
        matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis object to which the image plot is
            added.
            cmap (str, optional): Matplotlib colormap used to create the 
            image. Defaults to 'gist_gray'.
        """
        ax.imshow(self.slice_data, cmap=cmap)

