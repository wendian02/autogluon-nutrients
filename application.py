

import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import re
from datetime import datetime
from autogluon.tabular import TabularPredictor
import torch
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from nutrient_utils.my_utils import read_sentinel_L2Acolite, save_Nutrients_nc_file, get_satellite_name_acolite
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor


# warnings.filterwarnings('ignore')  # filter warning

def extract_date(s):
    # extract date from string
    match = re.search(r'\d{4}_\d{2}_\d{2}', s)
    date_str = match.group(0)
    return pd.to_datetime(date_str, format='%Y_%m_%d')  # convert to datetime


def predict_batch(predictor, batch_data):
    return predictor.predict(batch_data)


def batch_predict_parallel(predictor, data, batch_size=256):
    batches = [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]

    with ProcessPoolExecutor() as executor, tqdm(total=len(batches), desc="Predicting") as pbar:
        futures = {executor.submit(predict_batch, predictor, batch): batch for batch in batches}
        predictions = []
        for future in futures:
            result = future.result()
            predictions.append(result)
            pbar.update(1)

    return pd.concat(predictions)


def convert_s2_s3_autogl(rhorc_2d, auto_model_dir='AutogluonModels'):
    s3_bands = [443, 490, 560, 620, 709, 754, 779, 865]  # s3ab bands
    s3_bands_labels = [f"s3rhorc_{band}" for band in s3_bands]

    s2_bands = [443, 492, 560, 665, 704, 740, 783, 865]  # s2a bands
    s2_bands_labels = [f"s2rhorc_{band}" for band in s2_bands]


    df_rhorc_2d = pd.DataFrame(rhorc_2d, columns=s2_bands_labels)


    original_index = df_rhorc_2d.index
    df_rhorc_2d_clean = df_rhorc_2d.dropna()
    df_rhorc_2d_clean = TabularDataset(df_rhorc_2d_clean)

    # predict
    y_pred_allband = []
    for s3_bands_label in s3_bands_labels:
        predictor_transfer = TabularPredictor.load(os.path.join(auto_model_dir, f"ag-{s3_bands_label}"))

        y_pred = batch_predict_parallel(predictor_transfer, df_rhorc_2d_clean, batch_size=10000)  # batch predict
        y_pred_allband.append(y_pred)

    y_pred_allband = pd.concat(y_pred_allband, axis=1)


    # align to original index
    y_pred_allband_reindexed = y_pred_allband.reindex(original_index)

    return y_pred_allband_reindexed

def apply_model(nc_file, outputdir, transfer_model_dir, predictor_DIN, predictor_DIP):
    fn = os.path.basename(nc_file)
    satellite_type = get_satellite_name_acolite(fn)
    coordinates = [117.46, 119.25,  24.28, 24.68]  # XMB

    if "sentinel-2" in satellite_type:

        lon, lat, rhorc_np_3d, l2_flags = read_sentinel_L2Acolite(nc_file, satellite_type,
                                                                  bands_type='s2s3 common band', cloud_mask=False,
                                                                  is_l2_flags=True,
                                                                  is_crop=True, coordinates=coordinates)  # rhorc
        rhorc_2d = rhorc_np_3d.reshape(-1, rhorc_np_3d.shape[2])
        rhorc_2d_convert = convert_s2_s3_autogl(rhorc_2d, transfer_model_dir)  # autogl model


    elif "sentinel-3" in satellite_type:
        lon, lat, rhorc_np_3d, l2_flags = read_sentinel_L2Acolite(nc_file, satellite_type, bands_type='s2s3 common band',
                                                        cloud_mask=False, is_l2_flags=True,
                                                        is_crop=True, coordinates=coordinates)  # rhorc
        rhorc_2d = rhorc_np_3d.reshape(-1, rhorc_np_3d.shape[2])
        rhorc_2d_convert = rhorc_2d  # if S3, no need to convert
    else:
        print("Data not match")
        return None

    df_rhorc_2d = pd.DataFrame(rhorc_2d_convert)
    df_rhorc_2d_ratio = df_rhorc_2d.div(df_rhorc_2d.iloc[:, 2], axis=0).drop(2, axis=1)  # cal ratio
    df_rhorc_2d = pd.concat([df_rhorc_2d, df_rhorc_2d_ratio], axis=1)

    img_date = extract_date(os.path.basename(nc_file))
    df_rhorc_2d['date'] = pd.to_datetime(img_date)

    df_rhorc_2d.columns = ['rhorc_443', 'rhorc_490', 'rhorc_560', 'rhorc_665', 'rhorc_709',
                           'rhorc_754', 'rhorc_779', 'rhorc_865', 'rhorc_443_ratio', 'rhorc_490_ratio',
                           'rhorc_665_ratio', 'rhorc_709_ratio', 'rhorc_754_ratio', 'rhorc_779_ratio',
                           'rhorc_865_ratio', 'date']

    # results blank df
    df_rs_DIN = pd.DataFrame(index=df_rhorc_2d.index)
    df_rs_DIP = pd.DataFrame(index=df_rhorc_2d.index)

    # delete nan and cloud mask
    df_rhorc_2d.dropna(inplace=True)
    cloud_threshold = 0.2  # s2 cloud mask
    df_rhorc_2d = df_rhorc_2d[df_rhorc_2d['rhorc_865'] < cloud_threshold]

    y_pre_DIN = np.exp(batch_predict_parallel(predictor_DIN, df_rhorc_2d, batch_size=10000))
    y_pre_DIP = np.exp(batch_predict_parallel(predictor_DIP, df_rhorc_2d, batch_size=10000))

    # reshape output
    df_y_pre_DIN = pd.DataFrame(y_pre_DIN, index=df_rhorc_2d.index)
    df_y_pre_DIP = pd.DataFrame(y_pre_DIP, index=df_rhorc_2d.index)

    df_y_pre_DIN_rawIndex = df_rs_DIN.combine_first(df_y_pre_DIN)
    df_y_pre_DIP_rawIndex = df_rs_DIP.combine_first(df_y_pre_DIP)

    DIN_arr_3d = df_y_pre_DIN_rawIndex.values.reshape(np.shape(lon))
    DIP_arr_3d = df_y_pre_DIP_rawIndex.values.reshape(np.shape(lon))

    savename = f'Nutrient_{os.path.basename(nc_file)[:3]}_{img_date.strftime("%Y%m%d")}.nc'
    save_Nutrients_nc_file(os.path.join(outputdir, savename), lon, lat, DIP_arr_3d, DIN_arr_3d)


if __name__ == '__main__':

    # model
    # transfer_model_dir = '/Users/wendy/Documents/Python/Nutrients/model_transfer/MLP'
    autogluon_model_dir = ''
    predictor_DIN = TabularPredictor.load("AutogluonModels/autogl-DIN-without-lon-lat-logoutput-final")
    predictor_DIP = TabularPredictor.load("AutogluonModels/autogl-DIP-without-lon-lat-logoutput-final")

    # dir
    data_dir = ''  # input dir
    output_dir = ''
    for nc_file in glob.glob(os.path.join(data_dir, '*.nc')):
        start = datetime.now()
        apply_model(nc_file, output_dir, autogluon_model_dir, predictor_DIN, predictor_DIP)
        print(datetime.now() - start)

    # one imges

    # nc_file = 'F:/Satellite Data/S2_transfer/matchup_imgs/S2/S2B_MSI_2019_10_21_02_50_57_T50RPN_L2W.nc'
    # output_dir = 'F:/Satellite Data/Nutrients'
    # start = datetime.now()
    # model_s2(nc_file, output_dir, transfer_model_dir, predictor_DIN, predictor_DIP)
    # print(datetime.now() - start)
