import numpy as np
from scipy import stats
from .generate import dataframe_from_extraction, database_from_extraction

from grizli import utils
from grizli.aws import db


class ProjectedDensity:
    def __init__(self, query):
        self._result = db.from_sql(query)

    def rvs_from_cdf(self, s_grid, cdf_z, N=1000):
        s_grid = np.linspace(-5, 5, 51)
        cdf_grid = stats.norm.cdf(s_grid)

        dx_grid = np.diff(s_grid)
        pdf_grid = np.append([0], np.diff(cdf_grid)/dx_grid)

        # Redshift step sizes
        dz_grid = np.maximum(np.diff(cdf_z), 1.e-6)

        # PDF is dCDF/dz
        pdf_grid = np.append([0], np.diff(cdf_grid)/dz_grid)

        # Should be ~1
    #     print('Integral PDF: {0:.3f}'.format(np.trapz(pdf_grid, cdf_z)))

        # Step 1: draw uniform
        rv = np.random.rand(N)
        # rv = uniform.rvs(loc=0, scale=1, size=N)

        # Step 2: random z interpolated from CDF
        rv_z = np.interp(rv, cdf_grid, cdf_z)

        return pdf_grid, rv_z


