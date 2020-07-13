import random

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9
from astropy.table import Table
from grizli import utils
from grizli.aws import db
from grizli.fitting import compute_cdf_percentiles
from pylab import rcParams
from scipy import stats
from scipy.spatial import cKDTree
from tqdm import tqdm

from .plotting import Plotter

engine = db.get_db_engine()
utils.set_warnings()

rcParams['figure.figsize'] = 12, 8


class PDTable(Table):
    """
    Table populated by the values returned from a query of the Grizli database.
    Contains methods to calculate the projected density of the given field.
    """
    @classmethod
    def from_query(cls, query):
        """
        Generate a `PDTable` from a database query.

        Parameters
        ----------
        query : str
            The SQL query string to submit to the database.

        Returns
        -------
        instance : `PDTable`
            The `PDTable` instance with the results of the query as the
            columns.

        Examples
        --------
        >>> result_table = PDTable.from_query(
        ... "SELECT root, id, ra, dec, z_map, q_z, bic_diff, cdf_z"
        ... "FROM redshift_fit "
        ... "WHERE root = 'j021744m0346' AND status = 6 AND z_map > 0.5")
        """
        res = pd.read_sql_query(query, engine)
        instance = cls.from_pandas(res)

        return instance

    @classmethod
    def from_pickle(cls, path):
        """
        Loads a serialized `pd.DataFrame` into a `PDTable` instance.

        Parameters
        ----------
        path : str
            Path to the saved pickle file.

        Returns
        -------
        instance : `PDTable`
            The `PDTable` instance created from the pandas data frame.
        """
        res = pd.read_pickle(path)
        instance = cls.from_pandas(res)

        return instance

    @property
    def plot(self):
        """
        Convenience property to access the `Plotter` instance, which contains
        easy methods for basic plots.

        Returns
        -------
        `Plotter`
            The plotter class instance.
        """
        return Plotter(self)

    def randomize_redshift(self):
        """
        Simple implementation of some randomization for the redshift CDF.
        Essentially just randomizes the ``cdf_z`` column with the understanding
        that the redshift CDF values would then be randomly associated with
        different field objects.

        Returns
        -------
        new_table : ~`PDTable`
            New ~`PDTable` instance with randomized ``cdf_z`` column.
        """
        new_table = self.copy()
        random.shuffle(new_table['cdf_z'])

        return new_table

    def uniform_redshift(self):
        """
        Attempts to create a new CDF array of redshift based on a new
        distribution of the ``z_map`` column value. This uses the
        ~`compute_cdf_percentiles` from Grizli to generate the CDF
        distributions.

        Returns
        -------
        new_table : ~`PDTable`
            New table with randomized ``z_map`` values and new redshift CDF
            generated from the randomized distribution.
        """
        z_map = self['z_map']
        z_dist = np.linspace(min(z_map), max(z_map), len(z_map))

        new_z_map = np.zeros(len(z_map))

        new_table = self.copy()

        for i in range(len(new_z_map)):
            probs = np.random.sample(len(z_map))
            probs /= np.sum(probs)
            new_z_map[i] = np.random.choice(z_dist, p=probs)

        # Create random cdfs
        new_table['z_map'] = np.random.rand(len(z_map)) * np.max(z_map)
        cdf_x, cdf_y = compute_cdf_percentiles(
            self['fit_stack'].data,
            cdf_sigmas=np.linspace(-5, 5, 51))
        new_table['cdf_z'] = cdf_x

        return new_table

    def with_mask(self, mask):
        """
        Applies a mask to the table and returns the masked table. A simple
        convenience method for wrapping the regular mask creation for astropy
        table objects.

        Parameters
        ----------
        mask : ~`np.ndarray`
            The numpy masked array to be applied to the table.

        Returns
        -------
        ~`PDTable`
            Masked ~`PDTable` object.
        """
        return self[mask]

    def rvs_from_cdf(self, s_grid, cdf_z=None, num=1000):
        """
        Generate random variables from the CDF distribution. Based on the
        ``cdf_z`` values, create the CDF distribution from which they were
        extracted. Then, create random variable redshift values by
        interpolating an RV sample set of ``num`` values onto the CDF.

        Parameters
        ----------
        s_grid : array-like
            Range of values used in the generation of the CDF grid.
        cdf_z : array-like
            The redshift CDF values used in the final interpolation.
        num : float
            The number of random draws from the uniform distribution.

        Returns
        -------
        pdf_grid : array-like
            The PDF grid generated from the CDF.
        rv_z : array-like
            Random redshift variables interpolated from the CDF.
        """
        if cdf_z is None:
            cdf_z = self['cdf_z'][0]

        # Create the CDF related to the normal distribution centers on zero
        #  with a standard deviation of one.
        cdf_grid = stats.norm.cdf(s_grid)

        # Redshift step sizes
        dz_grid = np.maximum(np.diff(cdf_z), 1.e-6)

        # PDF is dCDF/dz
        pdf_grid = np.append([0], np.diff(cdf_grid)/dz_grid)

        # Should be ~1
        # print('Integral PDF: {0:.3f}'.format(np.trapz(pdf_grid, cdf_z)))

        # Step 1: draw uniform
        rv = np.random.rand(num)
        # rv = uniform.rvs(loc=0, scale=1, size=N)

        # Step 2: random z interpolated from CDF
        rv_z = np.interp(rv, cdf_grid, cdf_z)

        return pdf_grid, rv_z

    def redshift_draws(self, s_grid, num=1000):
        """
        For each object in the table, generate the RV redshift values.

        Parameters
        ----------
        s_grid : array-like
            Range of values used in the generation of the CDF grid.
        num : float
            The number of random draws from the uniform distribution.

        Returns
        -------
        z_draws : array-like
            A two-dimensional array containing the random redshift draws for
            each object in the table.
        """
        n_obj = len(self)
        z_draws = np.zeros((n_obj, num))
        # i_range = np.random.rand_int(0, n_obj, len(se))

        for i in tqdm(range(n_obj)):
            cdf_z = self['cdf_z'][i]
            _, z_draws[i, :] = self.rvs_from_cdf(s_grid, cdf_z, num=num)

        return z_draws

    def separation_radius(self, z_cm):
        """
        Generate the separation characteristics for a given redshift grid.

        Parameters
        ----------
        z_cm : array-like
            The log redshift grid of redshift values to be transformed to kpc
            comoving values per arc minute.

        Returns
        -------
        dr_cm : array-like
            Transverse comoving kpc values corresponding to an arcminute at the
            provided redshift values.
        dr_sep : float
            Separation radius for an annulus of area (1 Mpc)**2
        dr_area : float
            Area of a given separation radius.
        """
        # Calculate the separation given an array of redshift values
        # if z_cm is None:
        #     z_cm = utils.log_zgrid([0.1, 3.5], 0.01)

        dr_cm = WMAP9.kpc_comoving_per_arcmin(z_cm).to(u.Mpc/u.arcsec)

        # density
        # dz_thresh = 0.01  # separation threshold, dz*(1+z)

        # Separation radius
        dr_sep = np.sqrt(0.5 / np.pi) * u.Mpc
        dr_area = (np.pi * dr_sep.value ** 2)

        return dr_cm, dr_sep, dr_area

    def spatial_query(self, z_draws, z_cm, dz_thresh=0.01):
        """
        Perform a spatial query to calculate the projected density.

        Parameters
        ----------
        z_draws : array-like
            The redshift values drawn from the RV extractions.
        z_cm : array-like
            The log comoving redshift grid onto which the ``z_map`` values
            from Grizli will be interpolated, along with the transverse
            comoving kpc values. Used to calculate the scale at each redshift.
        dz_thresh : float
            The distance between objects in redshift space used to associate
            objects when calculating the projected densities.

        Returns
        -------
        scaled_proj_dens : array-like
            The scaled projected densities for each object in the table.
        dr : array-like
            The RA offset of each object from the center of the field, where
            the field center is calculated as the median of RA values.
        dd : array-like
            The DEC offset of each object from the center of the field, where
            the field center is calculated as the median of DEC values.
        """
        n_obj = len(self)

        dr = np.zeros_like(self['ra'])
        dd = dr * 0.
        aspect = dr * 0.

        projected_density = np.zeros(n_obj)

        roots = np.unique(self['root'])

        dr_cm, dr_sep, dr_area = self.separation_radius(z_cm)

        for root in tqdm(roots):
            # Get mask for unique object from root
            rsel = np.where(self['root'] == root)[0]

            # Find number of unique objects with this root
            n_r = len(rsel)

            # Find the center of the field for this root
            r0 = np.median(self['ra'][rsel])
            d0 = np.median(self['dec'][rsel])

            # Calculate offsets from the center for each object
            dr[rsel] = (self['ra'][rsel] - r0) * \
                np.cos(self['dec'][rsel] / 180 * np.pi) * 3600
            dd[rsel] = (self['dec'][rsel] - d0) * 3600
            aspect[rsel] = ((dr[rsel].max() - dr[rsel].min()) /
                            (dd[rsel].max() - dd[rsel].min()))

            points = np.array([dr[rsel], dd[rsel]]).T

            # Compose cKDTree
            tree = cKDTree(points)

            if not hasattr(dr_sep, 'unit'):
                # Find matches between the tree and itself if separation
                # defined in arcsec
                i0 = tree.query_ball_tree(tree, r=dr_sep)
            else:
                # Have to compute scale at each redshift
                scale = np.interp(self['z_map'][rsel], z_cm,
                                  dr_cm.to(dr_sep.unit / u.arcsec).value)

            for j in range(n_r):
                z_j = z_draws[rsel[j], :]

                if not(hasattr(dr_sep, 'unit')):
                    k = i0[j]
                else:
                    dr_arcsec = dr_sep.value / scale[j]
                    k = tree.query_ball_point(points[j], r=dr_arcsec)

                z_k = z_draws[rsel[k], :]

                dz_jk = (z_j - z_k)/(1 + z_j)
                w_k = (np.abs(dz_jk) < dz_thresh).sum(axis=1)/z_draws.shape[-1]

                # Summed weights, -1 because i0[j] includes j
                projected_density[rsel[j]] = (w_k.sum()-1)/dr_area

        # Now take out some redshift dependence since low-z things seem to
        # have higher densities but similar contrast
        # print(np.interp(1.8, z_cm, (dr_sep/dr_cm).value))

        rescale = np.interp(1.8, z_cm, (dr_sep/dr_cm).value) / \
            np.interp(self['z_map'], z_cm, (dr_sep/dr_cm).value)
        scaled_proj_dens = projected_density * rescale

        return scaled_proj_dens, dr, dd
