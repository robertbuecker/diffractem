
import yaml
import pprint
import json

class PreProcOpts:
    def __init__(self, fn=None):  

        self._filename = None
        self._help = {}

        # raw-image corrections
        self.verbose = True
        self.reference = 'Ref12_reference.tif'
        self.pxmask = 'Ref12_pxmask.tif'
        self.correct_saturation = True
        self.dead_time = 1.9e-3
        self.shutter_time = 2
        self.mask_gaps = True
        self.float = False
        self.cam_length = 2
        self.y_scale = 0.98 #TODO: extend to full ellipticity handling
        self.pixel_size = 55e-6
        self.com_threshold = 0.9
        self.com_xrng = 800
        self.lorentz_radius = 30
        self.lorentz_maxshift = 36
        self.xsize = 1556
        self.ysize = 616
        self.r_adf1 = (50, 100)
        self.r_adf2 = (100, 200)
        self.select_query = 'frame >= 0'
        self.agg_query = 'frame >= 0 and frame <= 5'
        self.agg_file_suffix = '_agg.h5'
        self.aggregate = True
        self.scratch_dir = '/scratch/diffractem'
        self.proc_dir = 'proc_data'
        self.rechunk = None
        self.peak_search_params = {'min-res': 5, 'max-res': 600,
                                   'local-bg-radius': 3, 'threshold': 8,
                                   'min-pix-count': 3,
                                   'min-snr': 3, 'int-radius': '3,4,5',
                                   'peaks': 'peakfinder8'}
        self.indexing_params = {'min-res': 0, 'max-res': 400, 'local-bg-radius': 4,
                                'threshold': 10, 'min-pix-count': 3, 'min-snr': 5,
                                'peaks': 'peakfinder8', 'indexing': 'none'}
        self.integration_params = {'min-res': 0, 'max-res': 400, 'local-bg-radius': 4,
                                   'threshold': 10, 'min-pix-count': 3, 'min-snr': 5,
                                   'peaks': 'peakfinder8', 'indexing': 'none'}
        self.peak_search_params.update({'temp-dir': self.scratch_dir})
        self.indexing_params.update({'temp-dir': self.scratch_dir})
        self.max_peaks = 500
        self.crystfel_procs = 40 # number of processes
        self.im_exc = 'indexamajig'
        self.geometry = 'calibrated.geom'
        self.peaks_cxi = True
        self.half_pixel_shift = False
        self.peaks_nexus = False
        self.friedel_refine = True
        self.min_peaks = 10
        self.peak_sigma = 2
        self.friedel_max_radius = None
        self.refined_file_suffix = '_ref.h5'
        self.center_stack = 'beam_center'
        self.broadcast_single = True
        self.broadcast_cumulative = True
        self.single_suffix = '_all.h5'
        self.idfields = ['file_raw', 'subset', 'sample', 'crystal_id', 'region', 'run']
        self.broadcast_peaks = True
        self.cum_file_suffix = '_cum.h5'
        self.cum_stacks = ['centered']
        self.cum_first_frame = 0
        self.rerun_peak_finder = False
        self.peak_radius = 4
        self.filter_len = 5
        self.nobg_file_suffix = '_nobg.h5'

        if fn is not None:
            self.load(fn)

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def load(self, fn=None):

        fn = self._filename if fn is None else fn
        if fn is None:
            raise ValueError('Please set the option file name first')

        if fn.endswith('json'):
            config = json.load(open(fn, 'r'))
        elif fn.endswith('yaml'):
            config = yaml.safe_load(open(fn, 'r'))
        else:
            raise ValueError('File extension must be .yaml or .json.')

        for k, v in config.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                print('Option', k, 'in', fn, 'unknown.')

        self._filename = fn

    def save(self, fn: str):
        if fn.endswith('json'):
            json.dump(self.__dict__, open(fn, 'w'), skipkeys=True, indent=4)
        elif fn.endswith('yaml'):
            yaml.dump(self.__dict__, open(fn, 'w'), sort_keys=False)