from __future__ import division, print_function
import warnings
import numpy as np

key_wrap = lambda data: lambda f, idx: f(data, key=lambda x: x[idx])[idx]


class LogBin(object):
    def __init__(self, points, type="log10", resolution=None, autorun=True):
        """

        :param points: a list of (x,y) points to bin
        :param type: what scale to use.  "log10" for  log10 otherwise reverts to linear.
               other fxns manually fixable via trans, inv attributes
        """
        self.raw_data = sorted(points)
        self.data = None
        self.type = type
        self.warn_me = False
        self.clean_data()
        curry_minmax = key_wrap(self.data)
        self.minx = curry_minmax(min, 0)
        self.maxx = curry_minmax(max, 0)
        self.miny = curry_minmax(min, 1)
        self.maxy = curry_minmax(max, 1)
        self.bin_edges = None
        self.binned_data = []
        self.default_resolution = 0.09
        self.trans = np.log10 if type == "log10" else lambda x:x
        self.inv = lambda x :10**x if type == "log10" else lambda x:x
        self.xavg, self.yavg, self.xerr, self.yerr = [None, None, None, None]
        self.resolution=resolution if resolution else self.default_resolution
        if autorun:
            self.run()

    def clean_data(self):
        """
        Remove points for which the x value cannot be plotted (ie x<0)
        :return:
        """
        for idx, point in enumerate(self.raw_data):
            if point[0] > 0:
                break
        self.data = self.raw_data[idx:]

    def set_bins(self, left_edge=None, right_edge=None, nbins=None):
        """
        sets the bins.  tries to get bin spacing of 0.1.
        empty bins are automatically removed from final data so a couple extra is not so bad.
        :param left_edge: lowest value to begin bins
        :param right_edge: highest value to end bins
        :param nbins: number of bins
        :return:
        """
        if left_edge is None:
            left_edge = self.minx
        if right_edge is None:
            right_edge = self.maxx
        if nbins is None:
            nbins = (self.trans(right_edge) - self.trans(left_edge)) / self.resolution + 1
            self.bin_edges = np.linspace(self.trans(left_edge), self.trans(right_edge), num=nbins)

    def bin_data(self):
        if self.bin_edges is None:
            raise ValueError("You need to run set_bins before you can bin the data")
        bin_idx = 0
        self.binned_data = [[] for i in enumerate(self.bin_edges)]
        for point in self.data:
            while self.trans(point[0]) > self.bin_edges[bin_idx + 1]:
                bin_idx += 1
            self.binned_data[bin_idx].append(point)
        assert sum(map(len, self.binned_data)) == len(self.data)

        if self.warn_me:
            if not all(self.binned_data):
                for idx, bin in enumerate(self.binned_data[:-1]):
                    if not bin:
                        warnings.warn("No data in bin %i [%.4f,%.4f]  Consider using fewer bins."
                                      % (idx, pow(10, self.bin_edges[idx]), pow(10, self.bin_edges[idx + 1])))

    def average_bins(self):
        if not self.binned_data:
            raise ValueError("You need to run bin_data before you can average the bins")
        xyavg = [np.mean(i, axis=0) for i in self.binned_data if i]
        xyerr = [np.std(i, axis=0) for i in self.binned_data if i]
        self.xavg, self.yavg = list(zip(*xyavg))
        self.xerr, self.yerr = list(zip(*xyerr))
        self.calc_clipped_errors()

    def calc_clipped_errors(self):
        '''
        if mean-err<0 it screws up loglog plots
        :return:
        '''
        self.yerr_clip=[]
        self.xerr_clip=[]
        for x,xerr in zip(self.xavg,self.xerr):
            if x - xerr >0:
                self.xerr_clip.append(xerr)
            else:
                self.xerr_clip.append(x)
        for y,yerr in zip(self.yavg,self.yerr):
            if y - yerr >0:
                self.yerr_clip.append(yerr)
            else:
                self.yerr_clip.append(y)

    def run(self):
        self.set_bins()
        self.bin_data()
        self.average_bins()
