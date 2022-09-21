solo-epd-loader
===============

|pypi Version| |conda version| |license| |python version| |zenodo doi|

.. |pypi Version| image:: https://img.shields.io/pypi/v/solo-epd-loader?style=flat&logo=pypi
   :target: https://pypi.org/project/solo-epd-loader/
.. |conda version| image:: https://img.shields.io/conda/vn/conda-forge/solo-epd-loader?style=flat&logo=anaconda
   :target: https://anaconda.org/conda-forge/solo-epd-loader/
.. |license| image:: https://img.shields.io/conda/l/conda-forge/solo-epd-loader?style=flat
   :target: https://github.com/jgieseler/solo-epd-loader/blob/main/LICENSE.rst
.. |python version| image:: https://img.shields.io/pypi/pyversions/solo-epd-loader?style=flat&logo=python
.. |zenodo doi| image:: https://zenodo.org/badge/446889843.svg
   :target: https://zenodo.org/badge/latestdoi/446889843

Python data loader for Solar Orbiter's (SolO) `Energetic Particle Detector (EPD) <http://espada.uah.es/epd/>`_. At the moment provides level 2 (l2) and low latency (ll) data (`more details on data levels here <http://espada.uah.es/epd/EPD_data_overview.php>`_) obtained through CDF files from ESA's `Solar Orbiter Archive (SOAR) <http://soar.esac.esa.int/soar>`_ for the following sensors:

- Electron Proton Telescope (EPT)
- High Energy Telescope (HET)
- SupraThermal Electrons and Protons (STEP)

Current caveats:

- Only the standard ``rates`` data products are supported (i.e., no ``burst`` or ``high cadence`` data).
- Only electrons, protons and alpha particles are processed (i.e., for HET He3, He4, C, N, O, Fe are omitted at the moment).
- For STEP, the sectored data is not yet available, and data is only available until Oct 2021 due to the change of the data product (will be updated soon).
- The Suprathermal Ion Spectrograph (SIS) is not yet included. 

Disclaimer
----------
This software is provided "as is", with no guarantee. It is no official data source, and not officially endorsed by the corresponding instrument teams. **Please always refer to the** `official EPD data description <http://espada.uah.es/epd/EPD_data.php>`_ **before using the data!**

Installation
------------

solo_epd_loader requires python >= 3.6.

It can be installed either from `PyPI <https://pypi.org/project/solo-epd-loader/>`_ using:

.. code:: bash

    pip install solo-epd-loader

or from `Anaconda <https://anaconda.org/conda-forge/solo-epd-loader/>`_ using:

.. code:: bash

    conda install -c conda-forge solo-epd-loader

Usage
-----

The standard usecase is to utilize the ``epd_load`` function, which
returns Pandas dataframe(s) of the EPD measurements and a dictionary
containing information on the energy channels.

.. code:: python

   from solo_epd_loader import epd_load

   df_1, df_2, energies = \
       epd_load(sensor, level, startdate, enddate=None, viewing=None, path=None, autodownload=False)

Input
~~~~~

-  ``sensor``: ``'ept'``, ``'het'``, or ``'step'`` (string)
-  ``level``: ``'ll'`` or ``'l2'`` (string)
-  ``startdate``, ``enddate``: Datetime object (e.g., ``dt.date(2021,12,31)`` or ``dt.datetime(2021,4,15)``) or integer of the form yyyymmdd with empty positions filled with zeros, e.g. ``20210415`` (if no ``enddate`` is provided, ``enddate = startdate`` will be used)
-  ``viewing``: ``'sun'``, ``'asun'``, ``'north'``, ``'south'`` (string) or ``None``; not
   needed for ``sensor = 'step'``
-  ``path``: directory in which Solar Orbiter data is/should be
   organized; e.g. ``'/home/userxyz/solo/data/'`` (string). See `Data folder structure`_ for more details.
-  ``autodownload``: if ``True`` will try to download missing data files
   from SOAR (bolean)

Return
~~~~~~

-  For ``sensor`` = ``'ept'`` or ``'het'``:

   1. Pandas dataframe with proton fluxes and errors (for EPT also alpha
      particles) in ‘particles / (s cm^2 sr MeV)’
   2. Pandas dataframe with electron fluxes and errors in ‘particles /
      (s cm^2 sr MeV)’
   3. Dictionary with energy information for all particles:

      -  String with energy channel info
      -  Value of lower energy bin edge in MeV
      -  Value of energy bin width in MeV

-  For ``sensor`` = ``'step'``:

   1. Pandas dataframe with fluxes and errors in ‘particles / (s cm^2 sr
      MeV)’
   2. Dictionary with energy information for all particles:

      -  String with energy channel info
      -  Value of lower energy bin edge in MeV
      -  Value of energy bin width in MeV

Data folder structure
---------------------

The ``path`` variable provided to the module should be the base
directory where the corresponding cdf data files should be placed in
subdirectories. First subfolder defines the data product ``level``
(``l2`` or ``low_latency`` at the moment), the next one the
``instrument`` (so far only ``epd``), and finally the ``sensor``
(``ept``, ``het`` or ``step``).

For example, the folder structure could look like this:
``/home/userxyz/solo/data/l2/epd/het``. In this case, you should call
the loader with ``path='/home/userxyz/solo/data'``; i.e., the base
directory for the data.

You can use the (automatic) download function described in the following
section to let the subfolders be created initially automatically. NB: It might
be that you need to run the code with *sudo* or *admin* privileges in order to
be able to create new folders on your system.

Data download within Python
---------------------------

While using ``epd_load()`` to obtain the data, one can choose to automatically
download missing data files from `SOAR <http://soar.esac.esa.int/soar>`_
directly from within python. They are saved in the folder provided by the
``path`` argument (see above). For that, just add ``autodownload=True`` to the
function call:

.. code:: python

   from solo_epd_loader import epd_load

   df_protons, df_electrons, energies = \
       epd_load(sensor='het', level='l2', startdate=20200820,
                enddate=20200821, viewing='sun',
                path='/home/userxyz/solo/data/', autodownload=True)

   # plot protons and alphas
   ax = df_protons.plot(logy=True, subplots=True, figsize=(20,60))
   plt.show()

   # plot electrons
   ax = df_electrons.plot(logy=True, subplots=True, figsize=(20,60))
   plt.show()

Note: The code will always download the *latest version* of the file
available at SOAR. So in case a file ``V01.cdf`` is already locally
present, ``V02.cdf`` will be downloaded nonetheless.

Example 1 - low latency data
----------------------------

Example code that loads low latency (ll) electron and proton (+alphas)
fluxes (and errors) for EPT NORTH telescope from Apr 15 2021 to Apr 16
2021 into two Pandas dataframes (one for protons & alphas, one for
electrons). In general available are ‘sun’, ‘asun’, ‘north’, and ‘south’
viewing directions for ‘ept’ and ‘het’ telescopes of SolO/EPD.

.. code:: python

   from matplotlib import pyplot as plt
   from solo_epd_loader import epd_load

   df_protons, df_electrons, energies = \
       epd_load(sensor='ept', level='ll', startdate=20210415,
                enddate=20210416, viewing='north',
                path='/home/userxyz/solo/data/')

   # plot protons and alphas
   ax = df_protons.plot(logy=True, subplots=True, figsize=(20,60))
   plt.show()

   # plot electrons
   ax = df_electrons.plot(logy=True, subplots=True, figsize=(20,60))
   plt.show()

Example 2 - level 2 data
------------------------

Example code that loads level 2 (l2) electron and proton (+alphas)
fluxes (and errors) for HET SUN telescope from Aug 20 2020 to Aug 20
2020 into two Pandas dataframes (one for protons & alphas, one for
electrons).

.. code:: python

   from matplotlib import pyplot as plt
   from solo_epd_loader import epd_load

   df_protons, df_electrons, energies = \
       epd_load(sensor='het', level='l2', startdate=20200820,
                enddate=20200821, viewing='sun',
                path='/home/userxyz/solo/data/')

   # plot protons and alphas
   ax = df_protons.plot(logy=True, subplots=True, figsize=(20,60))
   plt.show()

   # plot electrons
   ax = df_electrons.plot(logy=True, subplots=True, figsize=(20,60))
   plt.show()

Example 3 - partly reproducing `Fig. 2 <https://www.aanda.org/articles/aa/full_html/2021/12/aa39883-20/F2.html>`_ from Gómez-Herrero et al. 2021 [#]_
-----------------------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

   from matplotlib import pyplot as plt
   from solo_epd_loader import epd_load
   import numpy as np

   # set your local path here
   lpath = '/home/userxyz/solo/data'

   # load ept sun viewing data
   df_protons_ept, df_electrons_ept, energies_ept = \
      epd_load(sensor='ept', level='l2', startdate=20200708, 
               enddate=20200724, viewing='sun', path=lpath, autodownload=True)

   # load step data             
   df_step, energies_step = \
      epd_load(sensor='step', level='l2', startdate=20200708,
               enddate=20200724, path=lpath, autodownload=True)

   # change time resolution to get smoother curve (resample with mean)
   resample = '60min'

   fig, axs = plt.subplots(2, sharex=True, figsize=(8, 10), dpi=200)
   axs[0].set_prop_cycle('color', plt.cm.Oranges_r(np.linspace(0,1,7)))
   axs[1].set_prop_cycle('color', plt.cm.winter(np.linspace(0,1,7)))

   # plot selection of ept electron channels
   for channel in [0, 8, 16, 26]:
      df_electrons_ept['Electron_Flux'][f'Electron_Flux_{channel}'].resample(resample).mean().plot(
         ax = axs[0], logy=True, label='EPT '+energies_ept["Electron_Bins_Text"][channel][0])

   # plot selection of step ion channels
   for channel in [8, 17, 33]:
      df_step['Magnet_Flux'][channel].resample(resample).mean().plot(
         ax = axs[1], logy=True, label='STEP '+energies_step["Bins_Text"][channel][0])

   # plot selection of ept ion channels
   for channel in [6, 22, 32, 48]:
      df_protons_ept['Ion_Flux'][f'Ion_Flux_{channel}'].resample(resample).mean().plot(
         ax = axs[1], logy=True, label='EPT '+energies_ept["Ion_Bins_Text"][channel][0])

   axs[0].set_ylim([0.3, 4e6])
   axs[1].set_ylim([0.01, 5e8])

   axs[0].set_ylabel("Electron flux\n"+r"(cm$^2$ sr s MeV)$^{-1}$")
   axs[1].set_ylabel("Ion flux\n"+r"(cm$^2$ sr s MeV)$^{-1}$")
   axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
   axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
   plt.subplots_adjust(hspace=0)
   fig.savefig("gh2021_fig_2.png", bbox_inches = "tight")
   plt.close('all')

**NB: This is just an approximate reproduction with different energy
channels, different time resolution, and different viewing direction!
Note also that the STEP data can not be used straightforwardly.**
|Figure|

Example 4 - partly reproducing `Fig. 2e <https://www.aanda.org/articles/aa/full_html/2021/12/aa40940-21/F2.html>`_ from Wimmer-Schweingruber et al. 2021 [#]_ 
-------------------------------------------------------------------------------------------------------------------------------------------------------------

.. code:: python

   from matplotlib import pyplot as plt
   from solo_epd_loader import epd_load
   import datetime
   import pandas as pd

   # set your local path here
   lpath = '/home/userxyz/solo/data'

   # load data
   df_protons_sun, df_electrons_sun, energies = \
       epd_load(sensor='ept', level='l2', startdate=20201210,
                enddate=20201211, viewing='sun',
                path=lpath, autodownload=True)
   df_protons_asun, df_electrons_asun, energies = \
       epd_load(sensor='ept', level='l2', startdate=20201210,
                enddate=20201211, viewing='asun',
                path=lpath, autodownload=True)
   df_protons_south, df_electrons_south, energies = \
       epd_load(sensor='ept', level='l2', startdate=20201210,
                enddate=20201211, viewing='south',
                path=lpath, autodownload=True)
   df_protons_north, df_electrons_north, energies = \
       epd_load(sensor='ept', level='l2', startdate=20201210,
                enddate=20201211, viewing='north',
                path=lpath, autodownload=True)

   # plot mean intensities of two energy channels; 'channel' defines the lower one
   channel = 6
   ax = pd.concat([df_electrons_sun['Electron_Flux'][f'Electron_Flux_{channel}'],
                   df_electrons_sun['Electron_Flux'][f'Electron_Flux_{channel+1}']],
                   axis=1).mean(axis=1).plot(logy=True, label='sun', color='#d62728')
   ax = pd.concat([df_electrons_asun['Electron_Flux'][f'Electron_Flux_{channel}'],
                   df_electrons_asun['Electron_Flux'][f'Electron_Flux_{channel+1}']],
                   axis=1).mean(axis=1).plot(logy=True, label='asun', color='#ff7f0e')
   ax = pd.concat([df_electrons_north['Electron_Flux'][f'Electron_Flux_{channel}'],
                   df_electrons_north['Electron_Flux'][f'Electron_Flux_{channel+1}']],
                   axis=1).mean(axis=1).plot(logy=True, label='north', color='#1f77b4')
   ax = pd.concat([df_electrons_south['Electron_Flux'][f'Electron_Flux_{channel}'],
                   df_electrons_south['Electron_Flux'][f'Electron_Flux_{channel+1}']],
                   axis=1).mean(axis=1).plot(logy=True, label='south', color='#2ca02c')

   plt.xlim([datetime.datetime(2020, 12, 10, 23, 0), 
             datetime.datetime(2020, 12, 11, 12, 0)])

   ax.set_ylabel("Electron flux\n"+r"(cm$^2$ sr s MeV)$^{-1}$")
   plt.title('EPT electrons ('+str(energies['Electron_Bins_Low_Energy'][channel])
             + '-' + str(energies['Electron_Bins_Low_Energy'][channel+2])+' MeV)')
   plt.legend()
   plt.show()

**NB: This is just an approximate reproduction; e.g., the channel
combination is a over-simplified approximation!** |image1|

References
----------

.. [#] First near-relativistic solar electron events observed by EPD onboard Solar Orbiter, Gómez-Herrero et al., A&A, 656 (2021) L3, https://doi.org/10.1051/0004-6361/202039883

.. [#] First year of energetic particle measurements in the inner heliosphere with Solar Orbiter’s Energetic Particle Detector, Wimmer-Schweingruber et al., A&A, 656 (2021) A22, https://doi.org/10.1051/0004-6361/202140940

.. |Figure| image:: https://github.com/jgieseler/solo-epd-loader/raw/main/examples/gh2021_fig_2.png
.. |image1| image:: https://github.com/jgieseler/solo-epd-loader/raw/main/examples/ws2021_fig_2d.png

License
-------

This project is Copyright (c) Jan Gieseler and licensed under
the terms of the BSD 3-clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.

Acknowledgements
----------------

The development of this software has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101004159 (SERPENTINE).
