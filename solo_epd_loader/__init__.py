# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

import glob
import itertools
import numpy as np
import os
import pandas as pd
import re
import urllib.request

from pathlib import Path
from astropy.io.votable import parse_single_table
import cdflib


"""
Example code that loads low latency (ll) electron and proton (+alphas) fluxes
(and errors) for 'ept' 'north' telescope from Apr 15 2021 to Apr 16 2021 into
two Pandas dataframes (one for protons & alphas, one for electrons). In general
available are 'sun', 'asun', 'north', and 'south' viewing directions for 'ept'
and 'het' telescopes of SolO/EPD.

from epd_loader import *

df_protons, df_electrons, energies = \
_read_epd_cdf('ept', 'north', 'll', 20210415, 20210416,
path='/home/userxyz/solo/data/')

# plot protons and alphas
ax = df_protons.plot(logy=True, subplots=True, figsize=(20,60))
plt.show()

# plot electrons
ax = df_electrons.plot(logy=True, subplots=True, figsize=(20,60))
plt.show()
"""

"""
Example code that loads level 2 (l2) electron and proton (+alphas) fluxes
(and errors) for 'het' 'sun' telescope from Aug 20 2020 to Aug 20 2020 into
two Pandas dataframes (one for protons & alphas, one for electrons).

from epd_loader import *

df_protons, df_electrons, energies = \
_read_epd_cdf('het', 'sun', 'l2', 20200820, 20200821,
path='/home/userxyz/solo/data/')

# plot protons and alphas
ax = df_protons.plot(logy=True, subplots=True, figsize=(20,60))
plt.show()

# plot electrons
ax = df_electrons.plot(logy=True, subplots=True, figsize=(20,60))
plt.show()

"""


def _check_duplicates(filelist, verbose=True):
    """
    Checks for duplicate file entries in filelist (that are only different by
    version number). Returns filelist with duplicates removed.
    """
    for _, g in itertools.groupby(filelist, lambda f: f.split('_')[:-1]):
        dups = list(g)
        if len(dups) > 1:
            dups.sort()
            if verbose:
                print('')
                print('WARNING: Following data files are duplicates with ' +
                      'different version numbers:')
            for i in dups:
                print(i)
            if verbose:
                print('')
                print('Removing following files from list that will be read: ')
            for n in range(len(dups)-1):
                print(dups[n])
                filelist.remove(dups[n])
            if verbose:
                print('You might want to delete these files in order to get ' +
                      'rid of this message.')
    return filelist


def _get_filename_url(cd):
    """
    Get download filename for a url from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0][1:-1]


def _load_tqdm(verbose=True):
    """
    Tries to load tqdm package for displaying download progress.
    Return True or False, depending of success state.
    If not available, returns False.
    """
    try:
        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        def download_url(url, output_path):
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                     desc=output_path.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=output_path,
                                           reporthook=t.update_to)
        tqdm_available = True
    except ModuleNotFoundError:
        if verbose:
            print("Module tqdm not installed, won't show progress bar.")
        tqdm_available = False
        download_url = None
    return tqdm_available, download_url


def _get_epd_filelist(sensor, level, startdate, enddate, path,
                      filenames_only=False):
    """
    INPUT:
        sensor: 'ept' or 'het'
        level: 'll', 'l2'
        startdate, enddate: YYYYMMDD
        path: directory in which the data is located;
              e.g. '/home/userxyz/uni/solo/data/l2/epd/ept/'
        filenames_only: if True only give the filenames, not the full path
    RETURNS:
        Dictionary with four entries for 'sun', 'asun', 'north', 'south';
        each containing a list of files matching selection criteria.
    """

    if level == 'll':
        l_str = 'LL02'
        t_str = 'T??????-????????T??????'
    if level == 'l2':
        l_str = 'L2'
        t_str = ''

    filelist_sun = []
    filelist_asun = []
    filelist_north = []
    filelist_south = []
    for i in range(startdate, enddate+1):
        filelist_sun = filelist_sun + \
            glob.glob(path+'solo_'+l_str+'_epd-'+sensor+'-sun-rates_' +
                      str(i) + t_str + '_V*.cdf')
        filelist_asun = filelist_asun + \
            glob.glob(path+'solo_'+l_str+'_epd-'+sensor+'-asun-rates_' +
                      str(i) + t_str + '_V*.cdf')
        filelist_north = filelist_north + \
            glob.glob(path+'solo_'+l_str+'_epd-'+sensor+'-north-rates_' +
                      str(i) + t_str + '_V*.cdf')
        filelist_south = filelist_south + \
            glob.glob(path+'solo_'+l_str+'_epd-'+sensor+'-south-rates_' +
                      str(i) + t_str + '_V*.cdf')

    if filenames_only:
        filelist_sun = [os.path.basename(x) for x in filelist_sun]
        filelist_asun = [os.path.basename(x) for x in filelist_asun]
        filelist_north = [os.path.basename(x) for x in filelist_north]
        filelist_south = [os.path.basename(x) for x in filelist_south]

    filelist = {'sun': filelist_sun,
                'asun': filelist_asun,
                'north': filelist_north,
                'south': filelist_south
                }
    return filelist


def _get_step_filelist(level, startdate, enddate, path,
                       filenames_only=False):
    """
    INPUT:
        level: 'll', 'l2'
        startdate, enddate: YYYYMMDD
        path: directory in which the data is located;
              e.g. '/home/userxyz/uni/solo/data/l2/epd/step/'
        filenames_only: if True only give the filenames, not the full path
    RETURNS:
        List of files matching selection criteria.
    """

    sensor = 'step'

    if level == 'll':
        l_str = 'LL02'
        t_str = 'T??????-????????T??????'
    if level == 'l2':
        l_str = 'L2'
        t_str = ''

    filelist = []
    for i in range(startdate, enddate+1):
        filelist = filelist + \
            glob.glob(path+'solo_'+l_str+'_epd-'+sensor+'-rates_' +
                      str(i) + t_str + '_V*.cdf')

    if filenames_only:
        filelist = [os.path.basename(x) for x in filelist]

    return filelist


def _epd_ll_download(date, path, sensor, viewing=None):
    """
    Download EPD low latency data from http://soar.esac.esa.int/soar
    One file/day per call.

    Note: for sensor 'step' the 'viewing' parameter is necessary, but it

    Example:
        _epd_ll_download(20210415,
                        '/home/userxyz/solo/data/low_latency/epd/ept/',
                        'ept', 'north')
        _epd_ll_download(20200820,
                        '/home/userxyz/solo/data/low_latency/epd/step/',
                        'step')
    """

    # try loading tqdm for download progress display
    tqdm_available, download_url = _load_tqdm(verbose=True)

    # get list of available data files, obtain corresponding start & end time
    fl = get_available_soar_files(date, date, sensor, 'll')
    # try:
    if sensor.lower() == 'step':
        stime = 'T'+fl[0].split('T')[1].split('-')[0]
        etime = 'T'+fl[0].split('T')[2].split('_')[0]
        url = 'http://soar.esac.esa.int/soar-sl-tap/data?' + \
            'retrieval_type=LAST_PRODUCT&data_item_id=solo_LL02_epd-' + \
            sensor.lower()+'-rates_'+str(date) + \
            stime+'-'+str(date+1)+etime+'&product_type=LOW_LATENCY'
    else:
        stime = 'T'+fl[0].split('T')[1].split('-')[0]  # fl[0][-32:-25]
        etime = 'T'+fl[0].split('T')[2].split('_')[0]  # fl[0][-16:-9]
        url = 'http://soar.esac.esa.int/soar-sl-tap/data?' + \
            'retrieval_type=LAST_PRODUCT&data_item_id=solo_LL02_epd-' + \
            sensor.lower()+'-'+viewing.lower()+'-rates_'+str(date) + \
            stime+'-'+str(date+1)+etime+'&product_type=LOW_LATENCY'

    # Get filename from url
    file_name = _get_filename_url(
        urllib.request.urlopen(url).headers['Content-Disposition'])

    if tqdm_available:
        download_url(url, path+file_name)
    else:
        urllib.request.urlretrieve(url, path+file_name)

    return path+file_name


def _epd_l2_download(date, path, sensor, viewing=None):
    """
    Download EPD level 2 data from http://soar.esac.esa.int/soar
    One file/day per call.

    Example:
        _epd_l2_download(20200820,
                        '/home/userxyz/solo/data/l2/epd/ept/',
                        'ept', 'north')
        _epd_l2_download(20200820,
                        '/home/userxyz/solo/data/l2/epd/step/',
                        'step')
    """

    # try loading tqdm for download progress display
    tqdm_available, download_url = _load_tqdm(verbose=True)

    if sensor.lower() == 'step':
        url = 'http://soar.esac.esa.int/soar-sl-tap/data?' + \
            'retrieval_type=LAST_PRODUCT&data_item_id=solo_L2_epd-' + \
            sensor.lower()+'-rates_'+str(date) + \
            '&product_type=SCIENCE'
    else:
        url = 'http://soar.esac.esa.int/soar-sl-tap/data?' + \
            'retrieval_type=LAST_PRODUCT&data_item_id=solo_L2_epd-' + \
            sensor.lower()+'-'+viewing.lower()+'-rates_'+str(date) + \
            '&product_type=SCIENCE'

    # Get filename from url
    file_name = _get_filename_url(
        urllib.request.urlopen(url).headers['Content-Disposition'])

    if tqdm_available:
        download_url(url, path+file_name)
    else:
        urllib.request.urlretrieve(url, path+file_name)

    return path+file_name


def get_available_soar_files(startdate, enddate, sensor, level='l2'):
    """
    Get list of files available at SOAR

    Check ESA's SOAR database for available Solar Orbiter/EPD files in date
    range for give sensor and data level. Returns list of file names.

    Parameters
    ----------
    startdate : yyyymmdd (int)
        Provides year (yyyy), month (mm) and day (dd) of the start date as one
        combined integer; fill empty positions with zeros, e.g. '20210415'
    enddate : yyyymmdd (int)
        Provides year (yyyy), month (mm) and day (dd) of the end date as one
        combined integer; fill empty positions with zeros, e.g. '20210415'
    sensor : {'ept', 'het', 'step'}
        Defines EPD sensor
    level : {'l2', 'll'}, optional
        Defines level of data product: level 2 ('l2') or low-latency ('ll');
        by default 'l2'

    Returns
    -------
    filelist : list of str
        List of corresponding files available at SOAR.
    """

    # add 1 day to enddate to better work with SOAR's API
    # enddate = (pd.to_datetime(str(enddate))+
    #            pd.to_timedelta('1d')).strftime('%Y%m%d')

    sy = str(startdate)[0:4]
    sm = str(startdate)[4:6]
    sd = str(startdate)[6:8]

    ey = str(enddate)[0:4]
    em = str(enddate)[4:6]
    ed = str(enddate)[6:8]

    if level.lower() == 'l2':
        p_level = 'L2'  # "processing_level"
    #     data_type = 'v_sc_data_item'
    if level.lower() == 'll':
        p_level = 'LL02'  # "processing_level"
    #     data_type = 'v_ll_data_item'
    data_type = 'v_public_files'

    url = "http://soar.esac.esa.int/soar-sl-tap/tap/sync?REQUEST=doQuery&" + \
          "LANG=ADQL&retrieval_type=LAST_PRODUCT&FORMAT=votable_plain&" + \
          "QUERY=SELECT+*+FROM+"+data_type + \
          "+WHERE+(instrument='EPD')+AND+((begin_time%3E%3D'"+sy+"-"+sm + \
          "-"+sd+"+00:00:00')+AND+(begin_time%3C%3D'"+ey+"-"+em+"-"+ed + \
          "+01:00:00'))"

    filelist = urllib.request.urlretrieve(url)

    # open VO table, convert to astropy table, convert to pandas dataframe
    df = parse_single_table(filelist[0]).to_table().to_pandas()

    # convert bytestrings to unicode, from stackoverflow.com/a/67051068/2336056
    for col, dtype in df.dtypes.items():
        if dtype == np.object:  # Only process object columns.
            # decode, or return original value if decode return Nan
            df[col] = df[col].str.decode('utf-8').fillna(df[col])

    # remove duplicates with older version number
    df = df.sort_values('file_name')
    df.drop_duplicates(subset=['item_id'], keep='last', inplace=True)

    # only use data level wanted; i.e., 'LL' or 'L2'
    df = df[df['processing_level'] == p_level]

    # list filenames for given telescope (e.g., 'HET')
    # filelist = df['filename'][df['sensor'] == sensor.upper()].sort_values()
    filelist = [s for s in df['file_name'].values if sensor.lower() in s]

    # list filenames for 'rates' type (i.e., remove 'hcad')
    filelist = [s for s in filelist if "rates" in s]

    # filelist.sort()
    if len(filelist) == 0:
        print('No corresponding data found at SOAR!')
    return filelist


def _autodownload_cdf(startdate, enddate, sensor, level, path):
    """
    Uses get_available_soar_files() to check which files for selection criteria
    are available online. Compares with locally available files at 'path', and
    downloads missing files to 'path' using epd_l*_download()
    """
    fls = get_available_soar_files(startdate, enddate, sensor, level)
    for i in fls:
        my_file = Path(path)/i
        if not my_file.is_file():
            print(i+' MISSING => DOWNLOADING...')
            tdate = int(i.split('_')[3].split('T')[0])
            tview = i.split('-')[2]
            if level.lower() == 'll':
                _ = _epd_ll_download(date=tdate, path=path, sensor=sensor,
                                     viewing=tview)
            if level.lower() == 'l2':
                _ = _epd_l2_download(date=tdate, path=path, sensor=sensor,
                                     viewing=tview)
    return


def epd_load(sensor, level, startdate, enddate=None, viewing=None, path=None,
             autodownload=False):
    """
    Load SolO/EPD data

    Load-in data for Solar Orbiter/EPD energetic charged particle sensors EPT,
    HET, and STEP. Supports level 2 and low latency data provided by ESA's
    Solar Orbiter Archive. Optionally downloads missing data directly. Returns
    data as Pandas dataframe.

    Parameters
    ----------
    sensor : {'ept', 'het', 'step'}
        Defines EPD sensor
    level : {'l2', 'll'}
        Defines level of data product: level 2 ('l2') or low-latency ('ll')
    startdate : yyyymmdd (int)
        Provides year (yyyy), month (mm) and day (dd) of the start date as one
        combined integer; fill empty positions with zeros, e.g. '20210415'
    enddate : yyyymmdd (int), optional
        Provides year (yyyy), month (mm) and day (dd) of the end date as one
        combined integer; fill empty positions with zeros, e.g. '20210415'.
        (if no enddate is given, 'enddate = startdate' will be set)
    viewing : {'sun', 'asun', 'north', 'south' or None}, optional
        Viewing direction of sensor. Needed for 'ept' or 'het'; for 'step'
        shoule be None. By default None
    path : str, optional
        User-specified directory in which Solar Orbiter data is/should be
        organized; e.g. '/home/userxyz/solo/data/', by default None
    autodownload : bool, optional
        If True, will try to download missing data files from SOAR, by default
        False.

    Returns
    -------
    For EPT & HET:
        1. Pandas dataframe with proton fluxes and errors (for EPT also alpha particles) in 'particles / (s cm^2 sr MeV)'
        2. Pandas dataframe with electron fluxes and errors in 'particles / (s cm^2 sr MeV)'
        3. Dictionary with energy information for all particles:
            - String with energy channel info
            - Value of lower energy bin edge in MeV
            - Value of energy bin width in MeV
    For STEP:
        1. Pandas dataframe with fluxes and errors in 'particles / (s cm^2 sr MeV)'
        2. Dictionary with energy information for all particles:
            - String with energy channel info
            - Value of lower energy bin edge in MeV
            - Value of energy bin width in MeV

    Raises
    ------
    Exception
        Sensors 'ept' or 'het' need a provided 'viewing' direction. If None is
        given, Exception is raised.

    Examples
    --------
    Load EPD/HET sun viewing direction low-latency data for Aug 20 to Aug 22,
    2020 from user-defined directory, downloading missing files from SOAR:

    >>> df_protons, df_electrons, energies = epd_load(sensor='het',
    ...     viewing='sun', level='ll', startdate=20200820, enddate=20200822,
    ...     autodownload=True)

    Load EPD/STEP level 2 data for Aug 20 to Aug 22, 2020 from user-defined
    directory, downloading missing files from SOAR:

    >>> df, energies = epd_load(sensor='step', level='l2', startdate=20200820,
    ... enddate=20200822, autodownload=True)
    """

    if sensor.lower() == 'step':
        datadf, energies_dict = \
            _read_step_cdf(level, startdate, enddate, path, autodownload)
        return datadf, energies_dict
    if sensor.lower() == 'ept' or sensor.lower() == 'het':
        if viewing is None:
            raise Exception("EPT and HET need a telescope 'viewing' " +
                            "direction! No data read!")
            df_epd_p = []
            df_epd_e = []
            energies_dict = []
        else:
            df_epd_p, df_epd_e, energies_dict = \
                _read_epd_cdf(sensor, viewing, level, startdate, enddate, path,
                              autodownload)
        return df_epd_p, df_epd_e, energies_dict


def _read_epd_cdf(sensor, viewing, level, startdate, enddate=None, path=None,
                  autodownload=False):
    """
    INPUT:
        sensor: 'ept' or 'het' (string)
        viewing: 'sun', 'asun', 'north', or 'south' (string)
        level: 'll' or 'l2' (string)
        startdate,
        enddate:    YYYYMMDD, e.g., 20210415 (integer)
                    (if no enddate is given, 'enddate = startdate' will be set)
        path: directory in which Solar Orbiter data is/should be organized;
              e.g. '/home/userxyz/uni/solo/data/' (string)
        autodownload: if True will try to download missing data files from SOAR
    RETURNS:
        1. Pandas dataframe with proton fluxes and errors (for EPT also alpha
           particles) in 'particles / (s cm^2 sr MeV)'
        2. Pandas dataframe with electron fluxes and errors in
           'particles / (s cm^2 sr MeV)'
        3. Dictionary with energy information for all particles:
            - String with energy channel info
            - Value of lower energy bin edge in MeV
            - Value of energy bin width in MeV
    """

    # if no path to data directory is given, use the current directory
    if path is None:
        path = os.getcwd()

    # select sub-directory for corresponding sensor (EPT, HET)
    if level.lower() == 'll':
        path = Path(path)/'low_latency'/'epd'/sensor.lower()
    if level.lower() == 'l2':
        path = Path(path)/'l2'/'epd'/sensor.lower()

    # add a OS-specific '/' to end end of 'path'
    path = f'{path}{os.sep}'

    # if no 'enddate' is given, get data only for single day of 'startdate'
    if enddate is None:
        enddate = startdate

    # if autodownload, check online available files and download if not locally
    if autodownload:
        _autodownload_cdf(startdate, enddate, sensor.lower(), level.lower(),
                          path)

    # get list of local files for date range
    filelist = _get_epd_filelist(sensor.lower(), level.lower(), startdate,
                                 enddate, path=path)[viewing.lower()]

    # check for duplicate files with different version numbers and remove them
    filelist = _check_duplicates(filelist, verbose=True)

    if len(filelist) == 0:
        raise Exception('WARNING: No corresponding data files found! ' +
                        'Try different settings, path or autodownload.')
        df_epd_p = []
        df_epd_e = []
        energies_dict = []
    else:

        """ <-- get column names of dataframe """
        if sensor.lower() == 'ept':
            if level.lower() == 'll':
                protons = 'Prot'
                electrons = 'Ele'
                e_epoch = 'EPOCH'
            if level.lower() == 'l2':
                protons = 'Ion'
                electrons = 'Electron'
                e_epoch = 'EPOCH_1'
        if sensor.lower() == 'het':
            if level.lower() == 'll':
                protons = 'H'
                electrons = 'Ele'
                e_epoch = 'EPOCH'
            if level.lower() == 'l2':
                protons = 'H'  # EPOCH
                electrons = 'Electron'  # EPOCH_4, QUALITY_FLAG_4
                e_epoch = 'EPOCH_4'

        # load cdf files
        t_cdf_file = cdflib.CDF(filelist[0])
        df_p = _cdf2df(t_cdf_file, "EPOCH")
        df_e = _cdf2df(t_cdf_file, e_epoch)

        if len(filelist) > 1:
            for f in filelist[1:]:
                t_cdf_file = cdflib.CDF(f)
                t_df_p = _cdf2df(t_cdf_file, "EPOCH")
                t_df_e = _cdf2df(t_cdf_file, e_epoch)
                df_p = pd.concat([df_p, t_df_p])
                df_e = pd.concat([df_e, t_df_e])

        # p intensities:
        flux_p_channels = \
            [protons+f'_Flux_{i}' for i in
             range(t_cdf_file.varinq(protons+'_Flux')['Dim_Sizes'][0])]
        # p errors:
        if level.lower() == 'll':
            flux_sigma_p_channels = \
                [protons+f'_Flux_Sigma_{i}' for i in
                 range(t_cdf_file.varinq(protons+'_Flux')['Dim_Sizes'][0])]
        if level.lower() == 'l2':
            flux_sigma_p_channels = \
                [protons+f'_Uncertainty_{i}' for i in
                 range(t_cdf_file.varinq(protons+'_Flux')['Dim_Sizes'][0])]
            # p rates:
            rate_p_channels = \
                [protons+f'_Rate_{i}' for i in
                 range(t_cdf_file.varinq(protons+'_Rate')['Dim_Sizes'][0])]

        if sensor.lower() == 'ept':
            # alpha intensities:
            flux_a_channels = \
                [f'Alpha_Flux_{i}' for i in
                 range(t_cdf_file.varinq("Alpha_Flux")['Dim_Sizes'][0])]
            # alpha errors:
            if level.lower() == 'll':
                flux_sigma_a_channels = \
                    [f'Alpha_Flux_Sigma_{i}' for i in
                     range(t_cdf_file.varinq("Alpha_Flux")['Dim_Sizes'][0])]
            if level.lower() == 'l2':
                flux_sigma_a_channels = \
                    [f'Alpha_Uncertainty_{i}' for i in
                     range(t_cdf_file.varinq("Alpha_Flux")['Dim_Sizes'][0])]
                # alpha rates:
                rate_a_channels = \
                    [f'Alpha_Rate_{i}' for i in
                     range(t_cdf_file.varinq("Alpha_Rate")['Dim_Sizes'][0])]

        # e intensities:
        flux_e_channels = \
            [electrons+f'_Flux_{i}' for i in
             range(t_cdf_file.varinq(electrons+'_Flux')['Dim_Sizes'][0])]
        # e errors:
        if level.lower() == 'll':
            flux_sigma_e_channels = \
                [f'Ele_Flux_Sigma_{i}' for i in
                 range(t_cdf_file.varinq(electrons+'_Flux')['Dim_Sizes'][0])]
        if level.lower() == 'l2':
            flux_sigma_e_channels = \
                [f'Electron_Uncertainty_{i}' for i in
                 range(t_cdf_file.varinq(electrons+'_Flux')['Dim_Sizes'][0])]
            # e rates:
            rate_e_channels = \
                [electrons+f'_Rate_{i}' for i in
                 range(t_cdf_file.varinq(electrons+'_Rate')['Dim_Sizes'][0])]

        if level.lower() == 'l2':
            if sensor.lower() == 'het':
                df_epd_p = pd.concat(
                    [df_p[flux_p_channels], df_p[flux_sigma_p_channels],
                     df_p[rate_p_channels], df_p['DELTA_EPOCH'],
                     df_p['QUALITY_FLAG'], df_p['QUALITY_BITMASK']],
                    axis=1,
                    keys=['H_Flux', 'H_Uncertainty', 'H_Rate',
                          'DELTA_EPOCH', 'QUALITY_FLAG', 'QUALITY_BITMASK'])

                df_epd_e = pd.concat([df_e[flux_e_channels],
                                      df_e[flux_sigma_e_channels],
                                      df_e[rate_e_channels],
                                      df_e['DELTA_EPOCH_4'],
                                      df_e['QUALITY_FLAG_4'],
                                      df_e['QUALITY_BITMASK_4']], axis=1,
                                     keys=['Electron_Flux',
                                           'Electron_Uncertainty',
                                           'Electron_Rate',
                                           'DELTA_EPOCH_4',
                                           'QUALITY_FLAG_4',
                                           'QUALITY_BITMASK_4'])

            if sensor.lower() == 'ept':
                df_epd_p = pd.concat(
                    [df_p[flux_p_channels], df_p[flux_sigma_p_channels],
                     df_p[rate_p_channels], df_p[flux_a_channels],
                     df_p[flux_sigma_a_channels], df_p[rate_a_channels],
                     df_p['DELTA_EPOCH'], df_p['QUALITY_FLAG'],
                     df_p['QUALITY_BITMASK']],
                    axis=1,
                    keys=['Ion_Flux', 'Ion_Uncertainty', 'Ion_Rate',
                          'Alpha_Flux', 'Alpha_Uncertainty', 'Alpha_Rate',
                          'DELTA_EPOCH', 'QUALITY_FLAG', 'QUALITY_BITMASK'])

                df_epd_e = pd.concat([df_e[flux_e_channels],
                                      df_e[flux_sigma_e_channels],
                                      df_e[rate_e_channels],
                                      df_e['DELTA_EPOCH_1'],
                                      df_e['QUALITY_FLAG_1'],
                                      df_e['QUALITY_BITMASK_1']], axis=1,
                                     keys=['Electron_Flux',
                                           'Electron_Uncertainty',
                                           'Electron_Rate',
                                           'DELTA_EPOCH_1',
                                           'QUALITY_FLAG_1',
                                           'QUALITY_BITMASK_1'])

        if level.lower() == 'll':
            if sensor.lower() == 'het':
                df_epd_p = pd.concat(
                    [df_p[flux_p_channels], df_p[flux_sigma_p_channels]],
                    axis=1, keys=['H_Flux', 'H_Uncertainty', 'QUALITY_FLAG'])

            if sensor.lower() == 'ept':
                df_epd_p = pd.concat(
                    [df_p[flux_p_channels], df_p[flux_sigma_p_channels],
                     df_p[flux_a_channels], df_p[flux_sigma_a_channels],
                     df_p['QUALITY_FLAG']],
                    axis=1, keys=['Ion_Flux', 'Ion_Uncertainty',
                                  'Alpha_Flux', 'Alpha_Uncertainty',
                                  'QUALITY_FLAG'])

            df_epd_e = pd.concat([df_e[flux_e_channels],
                                  df_e[flux_sigma_e_channels],
                                  df_e['QUALITY_FLAG']], axis=1,
                                 keys=['Electron_Flux',
                                       'Electron_Uncertainty',
                                       'QUALITY_FLAG'])

        # replace FILLVALUES in dataframes with np.nan
        # t_cdf_file.varattsget("Ion_Flux")["FILLVAL"][0] = -1e+31
        # same for l2 & ll and het & ept and e, p/ion, alpha
        df_epd_p = df_epd_p.replace(-1e+31, np.nan)
        df_epd_e = df_epd_e.replace(-1e+31, np.nan)
        # NB: t_cdf_file.varinq('Ion_Flux')['Pad'][0] = -1e+30

        energies_dict = {protons+"_Bins_Text":
                         t_cdf_file.varget(protons+'_Bins_Text'),
                         protons+"_Bins_Low_Energy":
                         t_cdf_file.varget(protons+'_Bins_Low_Energy'),
                         protons+"_Bins_Width":
                         t_cdf_file.varget(protons+'_Bins_Width'),
                         electrons+"_Bins_Text":
                         t_cdf_file.varget(electrons+'_Bins_Text'),
                         electrons+"_Bins_Low_Energy":
                         t_cdf_file.varget(electrons+'_Bins_Low_Energy'),
                         electrons+"_Bins_Width":
                         t_cdf_file.varget(electrons+'_Bins_Width')
                         }

        if sensor.lower() == 'ept':
            energies_dict["Alpha_Bins_Text"] = \
                t_cdf_file.varget('Alpha_Bins_Text')
            energies_dict["Alpha_Bins_Low_Energy"] = \
                t_cdf_file.varget('Alpha_Bins_Low_Energy')
            energies_dict["Alpha_Bins_Width"] = \
                t_cdf_file.varget('Alpha_Bins_Width')

    '''
    Careful if adding more species - they might have different EPOCH
    dependencies and cannot easily be put in the same dataframe!
    '''

    return df_epd_p, df_epd_e, energies_dict


def _read_step_cdf(level, startdate, enddate=None, path=None,
                   autodownload=False):
    """
    INPUT:
        level: 'll' or 'l2' (string)
        startdate,
        enddate:    YYYYMMDD, e.g., 20210415 (integer)
                    (if no enddate is given, 'enddate = startdate' will be set)
        path: directory in which Solar Orbiter data is/should be organized;
              e.g. '/home/userxyz/uni/solo/data/' (string)
        autodownload: if True will try to download missing data files from SOAR
    RETURNS:
        1. Pandas dataframe with fluxes and errors in
           'particles / (s cm^2 sr MeV)'
        2. Dictionary with energy information for all particles:
            - String with energy channel info
            - Value of lower energy bin edge in MeV
            - Value of energy bin width in MeV
    """
    sensor = 'step'

    # if no path to data directory is given, use the current directory
    if path is None:
        path = os.getcwd()

    # select sub-directory for corresponding sensor (in this case just 'step')
    if level.lower() == 'll':
        path = Path(path)/'low_latency'/'epd'/sensor.lower()
    if level.lower() == 'l2':
        path = Path(path)/'l2'/'epd'/sensor.lower()

    # add a OS-specific '/' to end end of 'path'
    path = f'{path}{os.sep}'

    # if no 'enddate' is given, get data only for single day of 'startdate'
    if enddate is None:
        enddate = startdate

    # if True, check online available files and download if not locally present
    if autodownload:
        _autodownload_cdf(startdate, enddate, sensor.lower(), level.lower(),
                          path)

    # get list of local files for date range
    filelist = _get_step_filelist(level.lower(), startdate, enddate, path=path)

    # check for duplicate files with different version numbers and remove them
    filelist = _check_duplicates(filelist, verbose=True)

    if len(filelist) == 0:
        raise Exception('WARNING: No corresponding data files found! ' +
                        'Try different settings, path or autodownload.')
        datadf = []
        energies_dict = []
    else:
        all_cdf = []
        for file in filelist:
            all_cdf.append(cdflib.cdfread.CDF(file))

        if level == 'l2':
            param_list = ['Integral_Flux', 'Magnet_Flux', 'Integral_Rate',
                          'Magnet_Rate', 'Magnet_Uncertainty',
                          'Integral_Uncertainty']
            # set up the dictionary:
            energies_dict = \
                {"Bins_Text": all_cdf[0]['Bins_Text'],
                 "Bins_Low_Energy": all_cdf[0]['Bins_Low_Energy'],
                 "Bins_Width": all_cdf[0]['Bins_Width'],
                 "Sector_Bins_Text": all_cdf[0]['Sector_Bins_Text'],
                 "Sector_Bins_Low_Energy": all_cdf[0]['Sector_Bins_Low_Energy'],
                 "Sector_Bins_Width": all_cdf[0]['Sector_Bins_Width']
                 }
        if level == 'll':
            param_list = ['Integral_Flux', 'Ion_Flux', 'Integral_Flux_Sigma',
                          'Ion_Flux_Sigma']
            # set up the dictionary:
            energies_dict = \
                {"Integral_Bins_Text": all_cdf[0]['Integral_Bins_Text'],
                 "Integral_Bins_Low_Energy": all_cdf[0]['Integral_Bins_Low_Energy'],
                 "Integral_Bins_Width": all_cdf[0]['Integral_Bins_Width'],
                 "Ion_Bins_Text": all_cdf[0]['Ion_Bins_Text'],
                 "Ion_Bins_Low_Energy": all_cdf[0]['Ion_Bins_Low_Energy'],
                 "Ion_Bins_Width": all_cdf[0]['Ion_Bins_Width']
                 }

        df_list = []
        for cdffile in all_cdf:
            col_list = []
            for key in param_list:
                try:
                    col_list.append(pd.DataFrame(cdffile[key],
                                    index=cdffile['EPOCH']))
                except TypeError:
                    print(' ')
                    print("WARNING: Gap in dataframe due to missing cdf file.")
                    break
            try:
                temp_df = pd.concat(col_list, axis=1, keys=param_list)
                df_list.append(temp_df)
            except ValueError:
                continue
        datadf = pd.concat(df_list)

        # transform the index of the dataframe into pd_datetime
        # notice the transform alldata.index -> np.int_ so that encode()
        # understands the format
        datetimes = cdflib.cdfepoch.encode(np.int_(datadf.index))
        datadf.index = pd.to_datetime(datetimes)

        # Finally make sure that bad values are set to nan:
        datadf = datadf.replace(-1e+31, np.nan)

    '''
    Careful if adding more species - they might have different EPOCH
    dependencies and cannot easily be put in the same dataframe!
    '''

    return datadf, energies_dict


def _cdf2df(cdf, index_key, dtimeindex=True, badvalues=None,
            ignore=None, include=None):
    """
    Converts a cdf file to a pandas dataframe.
    Note that this only works for 1 dimensional data, other data such as
    distribution functions or pitch angles will not work properly.
    Parameters
    ----------
    cdf : cdf
        Opened CDF file.
    index_key : str
        The CDF key to use as the index in the output DataFrame.
    dtimeindex : bool
        If ``True``, the DataFrame index is parsed as a datetime.
        Default is ``True``.
    badvalues : dict, list
        Deprecated.
    ignore : list
        In case a CDF file has columns that are unused / not required, then
        the column names can be passed as a list into the function.
    include : str, list
        If only specific columns of a CDF file are desired, then the column
        names can be passed as a list into the function. Should not be used
        with ``ignore``.
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Data frame with read in data.
    """
    if badvalues is not None:
        warnings.warn('The badvalues argument is decprecated, as bad values '
                      'are now automatically recognised using the FILLVAL CDF '
                      'attribute.', DeprecationWarning)
    if include is not None:
        if ignore is not None:
            raise ValueError('ignore and include are incompatible keywords')
        if isinstance(include, str):
            include = [include]
        if index_key not in include:
            include.append(index_key)

    # Extract index values
    index_info = cdf.varinq(index_key)
    if index_info['Last_Rec'] == -1:
        raise CDFEmptyError('No records present in CDF file')

    index = cdf.varget(index_key)
    try:
        # If there are multiple indexes, take the first one
        # TODO: this is just plain wrong, there should be a way to get all
        # the indexes out
        index = index[...][:, 0]
    except IndexError:
        pass

    if dtimeindex:
        index = cdflib.epochs.CDFepoch.breakdown(index, to_np=True)
        index_df = pd.DataFrame({'year': index[:, 0],
                                 'month': index[:, 1],
                                 'day': index[:, 2],
                                 'hour': index[:, 3],
                                 'minute': index[:, 4],
                                 'second': index[:, 5],
                                 'ms': index[:, 6],
                                 })
        # Not all CDFs store pass milliseconds
        try:
            index_df['us'] = index[:, 7]
            index_df['ns'] = index[:, 8]
        except IndexError:
            pass
        index = pd.DatetimeIndex(pd.to_datetime(index_df), name='Time')
    data_dict = {}
    npoints = len(index)

    var_list = _get_cdf_vars(cdf)
    keys = {}
    # Get mapping from each attr to sub-variables
    for cdf_key in var_list:
        if ignore:
            if cdf_key in ignore:
                continue
        elif include:
            if cdf_key not in include:
                continue
        if cdf_key == 'Epoch':
            keys[cdf_key] = 'Time'
        else:
            keys[cdf_key] = cdf_key
    # Remove index key, as we have already used it to create the index
    keys.pop(index_key)
    # Remove keys for data that doesn't have the right shape to load in CDF
    # Mapping of keys to variable data
    vars = {cdf_key: cdf.varget(cdf_key) for cdf_key in keys.copy()}
    for cdf_key in keys:
        var = vars[cdf_key]
        if type(var) is np.ndarray:
            key_shape = var.shape
            if len(key_shape) == 0 or key_shape[0] != npoints:
                vars.pop(cdf_key)
        else:
            vars.pop(cdf_key)

    # Loop through each key and put data into the dataframe
    for cdf_key in vars:
        df_key = keys[cdf_key]
        # Get fill value for this key
        try:
            fillval = float(cdf.varattsget(cdf_key)['FILLVAL'])
        except KeyError:
            fillval = np.nan

        if isinstance(df_key, list):
            for i, subkey in enumerate(df_key):
                data = vars[cdf_key][...][:, i]
                data = _fillval_nan(data, fillval)
                data_dict[subkey] = data
        else:
            # If ndims is 1, we just have a single column of data
            # If ndims is 2, have multiple columns of data under same key
            key_shape = vars[cdf_key].shape
            ndims = len(key_shape)
            if ndims == 1:
                data = vars[cdf_key][...]
                data = _fillval_nan(data, fillval)
                data_dict[df_key] = data
            elif ndims == 2:
                for i in range(key_shape[1]):
                    data = vars[cdf_key][...][:, i]
                    data = _fillval_nan(data, fillval)
                    data_dict[f'{df_key}_{i}'] = data

    return pd.DataFrame(index=index, data=data_dict)


def _get_cdf_vars(cdf):
    # Get list of all the variables in an open CDF file
    var_list = []
    cdf_info = cdf.cdf_info()
    for attr in list(cdf_info.keys()):
        if 'variable' in attr.lower() and len(cdf_info[attr]) > 0:
            for var in cdf_info[attr]:
                var_list += [var]

    return var_list


def _fillval_nan(data, fillval):
    try:
        data[data == fillval] = np.nan
    except ValueError:
        # This happens if we try and assign a NaN to an int type
        pass
    return data
