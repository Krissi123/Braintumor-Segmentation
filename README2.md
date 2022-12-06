# Segmentation of Braintumors on MRI

### + + + +  WORK IN PROGRESS + + + +

## Introduction

In this project a convolutional neural network (CNN) is generated to segment malignant braintumors on MRI. 

In clinical routine it is still standard to measure diameters (length, width, height) to monitor the size of tumors, since a manual accurate segmentation would be too time-consuming and therefore cost-intensive. The result of the mesurement of the tumor is crucial for further decisions on patient's treatment. Hence it is absolutely essential to improve the mesurements by implementing automated segmentation tools. This would increase the quality tremendously while reducing the costs by saving a lot of radiologist's time.

## Methods

The model is trained on 611 datasets of patients of the UPENN-GBM Cohort. All images are pre-op scans. A U-NET CNN was trained on four standard MRI sequences (T1 native, T1 contrast enhanced, T2, FLAIR) and expert-verified segmentation maps.

See further information:
+ [Data](https://doi.org/10.7937/TCIA.709X-DN49) 

    Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., … Davatzikos, C. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. 

+ [Publication](https://doi.org/10.1038/s41597-022-01560-7) 

    Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., Lustig, R., Desai, A. S., Bagley, S. J., Mourelatos, Z., Morrissette, J., Watt, C. D., Brem, S., Wolf, R. L., Melhem, E. R., Nasrallah, M. P., Mohan, S., O’Rourke, D. M., Davatzikos, C. (2022). The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics. In Scientific Data (Vol. 9, Issue 1).

+ [The Cancer Imaging Archive (TCIA)](https://doi.org/10.1007/s10278-013-9622-7)
    
    Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057.

## .

## Requirements:

- pyenv with Python: 3.9.4

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
# NOT WORKING YET

make setup

# or also not finished

pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
