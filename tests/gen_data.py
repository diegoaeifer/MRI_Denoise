import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
import numpy as np
import os
import datetime

def create_dummy_dicom(filename, patient_id, series_uid, rows=256, cols=256):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    file_meta.ImplementationClassUID = '1.2.3.4'

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = f"Test^Patient^{patient_id}"
    ds.PatientID = patient_id
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = "Test Series"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.random.randint(0, 4096, (rows, cols), dtype=np.uint16).tobytes()
    
    # Needs valid date/time for some readers
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')
    
    ds.save_as(filename)

def generate_dataset(base_path, num_patients=3, series_per_pt=2, slices_per_series=5):
    os.makedirs(base_path, exist_ok=True)
    
    for p in range(num_patients):
        pid = f"P{p:03d}"
        for s in range(series_per_pt):
            sid = f"1.2.840.10008.5.{p}.{s}"
            series_dir = os.path.join(base_path, pid, f"Series_{s}")
            os.makedirs(series_dir, exist_ok=True)
            
            for sl in range(slices_per_series):
                fname = os.path.join(series_dir, f"slice_{sl:03d}.dcm")
                create_dummy_dicom(fname, pid, sid)
                
    print(f"Generated synthetic data in {base_path}")

if __name__ == "__main__":
    generate_dataset("FMImaging_MRI_Denoise/data/raw")
