import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sunpy
from astropy.utils.data import get_pkg_data_filename

from solo_epd_loader import calc_electrons, combine_channels, create_multiindex, epd_load, resample_df

# omit Pandas' PerformanceWarning


def test_ept_l2_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df_p, df_e, meta = epd_load(sensor='ept', startdate=20220420, viewing='asun', autodownload=True)
    assert isinstance(df_p, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df_p.shape == (38809, 219)
    assert df_e.shape == (38809, 105)
    assert meta['Electron_Bins_Text'][0][0] == '0.0312 - 0.0354 MeV'
    assert df_p['Ion_Flux']['Ion_Flux_4'].sum() == np.float32(61307010.0)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df_e['Electron_Flux']['Electron_Flux_1'])) == 1
    # test combine_channels for ions
    df_p_new, chan_p_new = combine_channels(df=df_p, energies=meta, en_channel=[9, 12], sensor='ept')
    assert chan_p_new == '0.0809 - 0.1034 MeV'
    assert df_p_new.shape == (38809, 1)
    assert df_p_new['flux'].sum() == np.float32(28518708.0)
    # test combine_channels for electrons
    df_e_new, chan_e_new = combine_channels(df=df_e, energies=meta, en_channel=[1, 3], sensor='het')
    assert chan_e_new == '0.0334 - 0.0420 MeV'
    assert df_e_new.shape == (38809, 1)
    assert df_e_new['flux'].sum() == np.float32(49434200.0)
    # test resampling
    df_p_res = resample_df(df=df_p, resample='1h')
    assert df_p_res.shape == (11, 219)
    assert df_p_res.index.freqstr == 'H'
    assert df_p_res.index[0].ctime() == 'Wed Apr 20 00:30:00 2022'
    assert df_p_res['Ion_Flux']['Ion_Flux_1'].iloc[0] == np.float32(2832.2144)


def test_ept_l2_load_offline():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    # offline data files need to be replaced if data "version" is updated!
    fullpath = get_pkg_data_filename('data/test/l2/epd/ept/solo_L2_epd-ept-sun-rates_20200603_V02.cdf', package='solo_epd_loader')
    path = Path(fullpath).parent.parent.as_posix().split('/l2')[0]
    df_p, df_e, meta = epd_load(sensor='ept', startdate=20200603, viewing='asun', path=path)
    assert isinstance(df_p, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df_p.shape == (1595, 219)
    assert df_e.shape == (158, 105)
    assert meta['Electron_Bins_Text'][0][0] == '0.0312 - 0.0348 MeV'
    assert df_p['Ion_Flux']['Ion_Flux_4'].sum() == np.float32(177390.75)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df_e['Electron_Flux']['Electron_Flux_1'])) == 2


def test_ept_ll_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=sunpy.util.SunpyUserWarning)
    df_p, df_e, meta = epd_load(sensor='ept', startdate=20220420, level='ll', viewing='north', autodownload=True)
    assert isinstance(df_p, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df_p.shape == (749, 37)
    assert df_e.shape == (749, 17)
    assert meta['Ele_Bins_Text'][0][0] == '0.0329 - 0.0411 MeV'
    assert df_p['Ion_Flux']['Prot_Flux_4'].sum() == np.float32(372648.72)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df_e['Electron_Flux']['Ele_Flux_1'])) == 1


def test_het_l2_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df_p, df_e, meta = epd_load(sensor='het', startdate=20220420, viewing='north', autodownload=True)
    assert isinstance(df_p, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df_p.shape == (7767, 111)
    assert df_e.shape == (38839, 15)
    assert meta['Electron_Bins_Text'][0][0] == '0.4533 - 1.0380 MeV'
    assert df_p['H_Flux']['H_Flux_5'].sum() == np.float32(35.128803)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df_e['Electron_Flux']['Electron_Flux_1'])) == 5
    # test combine_channels
    df_p_new, chan_p_new = combine_channels(df=df_p, energies=meta, en_channel=[9, 12], sensor='het')
    assert chan_p_new == '11.8000 - 15.6500 MeV'
    assert df_p_new.shape == (7767, 1)
    assert df_p_new['flux'].sum() == np.float32(3.106687)
    df_e_new, chan_e_new = combine_channels(df=df_e, energies=meta, en_channel=[1, 3], sensor='het')
    assert chan_e_new == '1.0530 - 18.8300 MeV'
    assert df_e_new.shape == (38839, 1)
    assert df_e_new['flux'].sum() == np.float32(353.9847)


def test_het_ll_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=sunpy.util.SunpyUserWarning)
    df_p, df_e, meta = epd_load(sensor='het', startdate=20220420, level='ll', viewing='north', autodownload=True)
    assert isinstance(df_p, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df_p.shape == (749, 24)
    assert df_e.shape == (749, 9)
    assert meta['Ele_Bins_Text'][0][0] == '0.4533 - 1.0380 MeV'
    assert df_p['H_Flux']['H_Flux_5'].sum() == np.float32(0.14029181)
    # Check that fillvals are replaced by NaN
    assert np.sum(np.isnan(df_e['Electron_Flux']['Ele_Flux_1'])) == 1


def test_step_l2_old_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df, meta = epd_load(sensor='step', startdate=20200820, autodownload=True)
    df_e = calc_electrons(df, meta, contamination_threshold=2, only_averages=False, resample=False)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df.shape == (8640, 675)
    assert df_e.shape == (8640, 1011)
    assert meta['Electron_Avg_Bins_Text'][0][0] == '4.09 - 4.57 keV'
    assert df['Magnet_15_Flux_7'].sum() == np.float32(6526933.0)
    assert df_e['Magnet_15_Flux_7'].sum() == df['Magnet_15_Flux_7'].sum()
    assert df_e['Electron_11_Flux_0'].sum() == np.float32(1071644.4)
    assert np.sum(np.isnan(df_e['Electron_11_Flux_0'])) == 8639
    # test create_multiindex
    df_m = create_multiindex(df_e)
    assert df_e.shape == df_m.shape
    assert ('Electron_Avg_Uncertainty', 'Electron_Avg_Uncertainty_7') in df_m.keys()


def test_step_l2_old_only_averages_resample_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df, meta = epd_load(sensor='step', startdate=20200820, autodownload=True, only_averages=True)
    df_e = calc_electrons(df, meta, contamination_threshold=2, only_averages=True, resample='1h')
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df.shape == (8640, 195)
    assert df_e.shape == (24, 291)
    assert meta['Electron_Avg_Bins_Text'][0][0] == '4.09 - 4.57 keV'
    assert df['Magnet_Avg_Flux_7'].sum() == np.float32(235159520.0)
    assert df_e['Magnet_Avg_Flux_7'].sum() == np.float32(653220.9)
    assert df_e['Electron_Avg_Flux_0'].sum() == np.float32(309272.3)
    assert np.sum(np.isnan(df_e['Electron_Avg_Flux_0'])) == 6


def test_step_l2_new_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df, meta = epd_load(sensor='step', startdate=20220109, autodownload=True)
    df_e = calc_electrons(df, meta, contamination_threshold=2, only_averages=False, resample=False)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df.shape == (49097, 2052)
    assert df_e.shape == (49097, 3076)
    assert meta['Electron_Bins_Text'][0][0] == '0.0041 - 0.0046 MeV'
    assert df['Magnet_15_Flux_7'].sum() == np.float32(544824900.0)
    assert df_e['Magnet_15_Flux_7'].sum() == df['Magnet_15_Flux_7'].sum()
    assert df_e['Electron_03_Flux_1'].sum() == np.float32(30770176.0)
    assert np.sum(np.isnan(df_e['Electron_03_Flux_1'])) == 49096


def test_step_l2_new_only_averages_resample_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df, meta = epd_load(sensor='step', startdate=20220109, autodownload=True, only_averages=True)
    df_e = calc_electrons(df, meta, contamination_threshold=2, only_averages=True, resample='1h')
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_e, pd.DataFrame)
    assert isinstance(meta, dict)
    assert df.shape == (49097, 132)
    assert df_e.shape == (14, 196)
    assert meta['Electron_Bins_Text'][0][0] == '0.0041 - 0.0046 MeV'
    assert df['Magnet_Avg_Flux_7'].sum() == np.float32(1814261200.0)
    assert df_e['Magnet_Avg_Flux_7'].sum() == np.float32(509400.1)
    assert df_e['Electron_Avg_Flux_1'].sum() == np.float32(83453.23)
    assert np.sum(np.isnan(df_e['Electron_Avg_Flux_1'])) == 9
    # test create_multiindex
    df_m = create_multiindex(df_e)
    assert df_e.shape == df_m.shape
    assert ('Electron_Avg_Uncertainty', 'Electron_Avg_Uncertainty_7') in df_m.keys()


def test_step_ll_old_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    df, meta = epd_load(sensor='step', startdate=20200820, autodownload=True, level='ll')
    assert df == []
    assert meta == []


def test_step_ll_new_load_online():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    df, meta = epd_load(sensor='step', startdate=20200820, autodownload=True, level='ll')
    assert df == []
    assert meta == []
