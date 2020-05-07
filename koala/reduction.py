import os
import numpy as np
import logging.handlers

from astropy.table import QTable, Table
import drizzlepac

import grizli
from grizli.pipeline import photoz
from grizli import utils, prep, multifit, fitting

import eazy
import click


@click.command()
@click.argument('extraction_path', type=click.Path(exists=True))
def photometry_fitting(extraction_path):
    os.chdir(extraction_path)

    # Fetch 3D-HST catalogs
    if not os.path.exists('uds_3dhst.v4.2.cats.tar.gz'):
        logging.info("Retrieving 3d-HST catalog")
        os.system('curl -O https://archive.stsci.edu/missions/hlsp/3d-hst/RELEASE_V4.0/Photometry/UDS/uds_3dhst.v4.2.cats.tar.gz')
        os.system('tar xzvf uds_3dhst.v4.2.cats.tar.gz')
