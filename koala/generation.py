import click
import os
from astropy.table import Table
from astropy.io import fits
from grizli.fitting import compute_cdf_percentiles
import numpy as np
import pandas as pd
from grizli import utils
from grizli.aws import db


def dataframe_from_extraction(root, extraction_path, output=None):
    # Load the created table from grizli
    ex_fits_path = os.path.join(extraction_path, f"{root}_phot.fits")
    grizli_table = Table.read(ex_fits_path)

    # Create a temporary dictionary to hold the table values
    res = {'id': [], 'root': [], 'ra': [], 'dec': [], 'z_map': [], 'cdf_z': []}

    for i, gid in enumerate(grizli_table['id']):
        file_path = os.path.join('f{extraction_path}',
                                 f'{root}_{gid:05d}.full.fits')

        if not os.path.exists(file_path):
            continue

        with fits.open(file_path) as hdulist:
            res['id'].append(gid)
            res['root'].append(root)
            res['ra'].append(hdulist['PRIMARY'].header['RA'])
            res['dec'].append(hdulist['PRIMARY'].header['DEC'])
            res['z_map'].append(hdulist['ZFIT_STACK'].header['Z_MAP'])

            cdf_x, cdf_y = compute_cdf_percentiles(hdulist['ZFIT_STACK'].data,
                                                   cdf_sigmas=np.linspace(-5, 5, 51))

            res['cdf_z'].append(cdf_x)

    tab = pd.DataFrame(data=res)
    tab.to_pickle("proj_density_tab.pkl")

    return tab


def database_from_query(query):
    engine = db.get_db_engine()
    utils.set_warnings()

    result = db.from_sql(query, engine)

    return result
