import matplotlib.pyplot as plt
import numpy as np
from grizli import utils


class Plotter:
    """
    Helper class that provides methods for plotting projected density-related
    statistical information.

    Parameters
    ----------
    table : ~`PDTable`
        The table instance containing the Grizli database information.
    """
    def __init__(self, table):
        self._table = table

    def redshift_map(self, ax=None):
        """
        Produces a histogram of redshift values stored in the ``z_map`` column
        of the Grizli database.

        Parameters
        ----------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`, optional
            The axis instance for this plot. If none is provided, creates a
            new axis object.

        Returns
        -------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance.
        """
        if ax is None:
            f, ax = plt.subplots()

        ax.hist(self._table['z_map'], bins=utils.log_zgrid([0.01, 3.5], 0.005))

        return ax

    def pdf_of_z(self, s_grid, num=1000, ax=None):
        """
        Create a histogram of the PDF generated from the RVs extracted from the
        ``cdf_z`` distribution.

        Parameters
        ----------
        s_grid :
            Range of values used in the generation of the CDF grid.
        num :
            The number of random draws from the uniform distribution.
        ax : ~`matplotlib.axes._subplots.AxesSubplot`, optional
            The axis instance for this plot. If none is provided, creates a
            new axis object.

        Returns
        -------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance.
        """
        pdf_grid, rv_z = self._table.rvs_from_cdf(s_grid)

        cdf_z = self._table['cdf_z'][0]

        zr = [cdf_z[0], cdf_z[-1]]
        zw = np.diff(zr)[0]

        if ax is None:
            f, ax = plt.subplots()

        ax.hist(rv_z, bins=100, range=zr, log=True, alpha=0.5)

        ax.plot(cdf_z, pdf_grid * num * zw / 100, linewidth=2)

        ax.set_ylim(0.1, pdf_grid.max() * 2 * num * zw / 100)
        ax.set_xlabel('z')
        ax.set_ylabel('PDF(z)')

        return ax

    def compare_redshift_draws(self, z_draws, mask=None, num=1000, ax=None):
        """
        Compares the RV draws generated from the redshift CDF distribution with
        the Grizli ``z_map`` values.

        Parameters
        ----------
        z_draws : array-like
            A two-dimensional array containing the random redshift draws for
            each object in the table.
        mask : ~`np.ndarray`
            Boolean mask to filter desired table values.
        num : float
            The number of random draws from the uniform distribution.
        ax : ~`matplotlib.axes._subplots.AxesSubplot`, optional
            The axis instance for this plot. If none is provided, creates a
            new axis object.

        Returns
        -------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance.
        """
        if ax is None:
            f, ax = plt.subplots()

        if mask is None:
            mask = np.ones(len(self._table)).astype(bool)

        ax.hist(self._table['z_map'][mask],
                bins=utils.log_zgrid([0.5, 2.5], 0.005),
                alpha=0.5)

        hx = np.histogram(z_draws[mask, :].flatten(),
                          bins=utils.log_zgrid([0.5, 2.5], 0.005))

        ax.plot(hx[1][1:], hx[0]/num, linestyle='steps-pre')

        return ax

    def transverse_plate_scale(self, z_cm, ax=None):
        """
        Plot the transverse plate scale.

        Parameters
        ----------
        z_cm : array-like
            The log comoving redshift grid onto which the ``z_map`` values
            from Grizli will be interpolated, along with the transverse
            comoving kpc values. Used to calculate the scale at each redshift.
        ax : ~`matplotlib.axes._subplots.AxesSubplot`, optional
            The axis instance for this plot. If none is provided, creates a
            new axis object.

        Returns
        -------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance.-
        """
        dr_cm, _, _ = self._table.separation_radius(z_cm)

        if ax is None:
            f, ax = plt.subplots()

        ax.plot(z_cm, dr_cm)
        ax.set_xlabel('z')
        ax.set_ylabel('transverse plate scale, comoving Mpc/arcsec')
        ax.grid()

        return ax

    def separation_radius(self, z_cm, ax=None):
        """
        Plot the separation radius as a function of the redshift comoving
        grid values.

        Parameters
        ----------
        z_cm : array-like
            The log comoving redshift grid onto which the ``z_map`` values
            from Grizli will be interpolated, along with the transverse
            comoving kpc values. Used to calculate the scale at each redshift.
        ax : ~`matplotlib.axes._subplots.AxesSubplot`, optional
            The axis instance for this plot. If none is provided, creates a
            new axis object.

        Returns
        -------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance.
        """

        dr_cm, dr_sep, dr_area = self._table.separation_radius(z_cm)

        if ax is None:
            f, ax = plt.subplots()

        ax.plot(z_cm, dr_sep/dr_cm)
        ax.set_xlabel('z')
        ax.set_ylabel('r, arcsec')
        ax.grid()
        ax.set_ylim(0, (dr_sep/dr_cm).value.max()*1.1)

        return ax

    def spatial_query(self, projected_densities, dz_thresh=0.01, z_cluster=0,
                      ax=None, **kwargs):
        """
        Plot the projected density of the field objects as a function of their
        redshift values, where the redshifts are taken from the ``z_map``
        Grizli values.

        Parameters
        ----------
        projected_densities : array-like
            The scaled projected densities for each object in the table.
        z_cluster : float
            The redshift value of the cluster, used with ``dz_thresh`` to
            color the range of objects within that redshift interval.
        ax : ~`matplotlib.axes._subplots.AxesSubplot`, optional
            The axis instance for this plot. If none is provided, creates a
            new axis object.
        kwargs : dict
            Key words arguments passed to the axis instance.

        Returns
        -------
        ax : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance.
        """
        dr_cm, dr_sep, dr_area = \
            self._table.separation_radius()

        h = np.histogram(self._table['z_map'],
                         bins=utils.log_zgrid([0.01, 3.4], dz_thresh*2))

        clu_sel = np.abs(self._table['z_map'] - z_cluster) \
                  < dz_thresh * (1 + z_cluster)

        if ax is None:
            f, ax = plt.subplots()

        ax.scatter(self._table['z_map'], projected_densities,
                   #    c=clu_sel,
                   **kwargs)

        ax.set_xlabel("$z$")
        ax.set_ylabel("Projected Density")

        return ax

    def sources(self, projected_densities, dr, dd, z_min=0.6, z_max=3,
                vm=(0, 1), min_density=8, cmap='magma_r', alpha=0.03,
                axes=None):
        """
        Plots the object field color coded by projected density values.

        Parameters
        ----------
        projected_densities : array-like
            The scaled projected densities for each object in the table.
        dr : array-like
            The RA offset of each object from the center of the field, where
            the field center is calculated as the median of RA values.
        dd : array-like
            The DEC offset of each object from the center of the field, where
            the field center is calculated as the median of DEC values.
        z_min : float
            Minimum redshift cutoff.
        z_max : float
            Maximum redshift cutoff.
        vm : tuple
            Value range for colored scatter marks.
        min_density : float
            Minimum projected density for color coding peaks.
        cmap : str
            The color scheme to use for plotting.
        alpha : float
            The alpha value for the scatter plot markers.
        axes : tuple, optional
            A two-tuple representing the axis objects for the upper and lower
            subplots.

        Returns
        -------
        ax1 : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance representing the
            2d plot of field objects.
        ax2 : ~`matplotlib.axes._subplots.AxesSubplot`
            The provided or newly instantiated axis instance representing the
            1d plot of field objects' projected densities.
        """
        peaks = (projected_densities > min_density) & \
                (self._table['z_map'] > z_min) & \
                (self._table['z_map'] < z_max)

        roots = np.unique(self._table['root'][peaks])

        if len(roots) == 0:
            return

        for root in roots:
            rsel = self._table['root'] == root

            sel = (self._table['z_map'] > z_min) & \
                  (self._table['z_map'] < z_max) & \
                  (projected_densities > 2)

            inds = np.argsort(projected_densities[sel])

            if axes is None:
                f, (ax1, ax2) = plt.subplots(2, 1)
            else:
                ax1, ax2 = axes

            ax1.scatter(dr[rsel & ~sel], dd[rsel & ~sel], color='k', alpha=alpha)
            ax1.scatter(dr[sel][inds], dd[sel][inds],
                        c=projected_densities[sel][inds],
                        alpha=0.9, s=200, cmap=cmap, vmin=vm[0], vmax=vm[1])

            ax1.grid()
            ax1.set_xlabel(r'$\Delta$RA, arcsec')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax1.set_ylabel(r'$\Delta$Dec, arcsec')

            ax1.set_title(root)

            ax2.scatter(self._table['z_map'][rsel & ~sel],
                        projected_densities[rsel & ~sel],
                        c='k', alpha=0.1, s=80)

            ax2.scatter(self._table['z_map'][rsel & sel],
                        projected_densities[rsel & sel],
                        c=projected_densities[rsel & sel],
                        vmin=vm[0], vmax=vm[1], alpha=0.9, s=100, cmap='magma_r')

            ax2.set_ylim(0, 20)
            ax2.set_xlim(0.2, 3.4)
            ax2.grid()

        return ax1, ax2
