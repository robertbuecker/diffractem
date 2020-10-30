
import yaml
import pprint
import json
from typing import Union

#TODO consider to make this a types.SimpleNamespace or use dataclasses.dataclass
class PreProcOpts:
    def __init__(self, fn=None):  

        self._filename = None
        self._help = {}

       
        self.filter_len = 5
        self.nobg_file_suffix = '_nobg.h5' # raw-image corrections
        self.verbose: bool = True
        self.reference: str = 'Ref12_reference.tif'  #: Name of reference image for flat-field correction in TIF format
        self.pxmask: str = 'Ref12_pxmask.tif'        #: Name of pixelmask TIF image
        self.correct_saturation: bool = True          #: Correct for detector saturation using paralyzable model
        self.dead_time: float = 1.9e-3                 #: Dead time (in ms) for paralyzable detector model
        self.dead_time_gap_factor: float = 2           #: Factor for dead time in gap-pixels 
        self.shutter_time: float = 2                   #: Shutter time (in ms) for paralyzable detector model
        self.mask_gaps: bool = True                     #: Always mask detector gaps (regardless of dead pixel mask)
        self.interpolate_dead: bool = False             #: Interpolate dead pixels instead of masking
        self.float: bool = False                        #: Process images as floating-point
        self.find_peaks: bool = True                #: apply peakfinder during image analysis
        self.compression: Union[int, str] = 32004       #: standard HDF5 compression. Suggested values: gzip, none, 32004 (lz4)
        self.cam_length: float = 2                     #: Average camera length (in m).
        self.y_scale: float = 1                     #: Scaling of camera length along y. DEPRECATED!
        self.ellipse_ratio: float = 1                  #: ellipticity of camera lentgh along arbitrary axis
        self.ellipse_angle: float = 0                     #: cam length ellipticity angle. (e.g. 0 is x, pi/2 is y,...)
        self.pixel_size: float = 55e-6                 #: Pixel size (in m)
        self.wavelength: float = 0.0251                 #: Radiation wave length (in A)
        self.com_threshold:float = 0.9
        self.com_xrng: int = 800                     #: x range (px) around pattern center in which to look for center of mass
        self.com_yrng: int = 800                     #: y range (px) around pattern center in which to look for center of mass
        self.lorentz_radius: int= 30                #: radius (px) around center of mass for Lorentz fit of zero order
        self.lorentz_maxshift: float = 36              #: maximum shift (px) of Lorentz fit center from center of mass
        self.xsize: int = 1556                       #: x image size (px)
        self.ysize: int = 516                        #: y image size (px)
        self.r_adf1: tuple = (50, 100)                 #: inner/outer radii for virtual ADF 1 (px)
        self.r_adf2: tuple = (100, 200)                #: inner/outer radii for virtual ADF 2 (px)
        self.select_query: str = 'frame >= 0'        #: query string for selection of shots from raw data
        self.agg_query: str = 'frame >= 0 and frame <= 5'    #: query string for aggregation of patterns
        self.agg_file_suffix: str = '_agg.h5'        #: file suffix for aggregated patterns
        self.aggregate: bool = True                   #: calculate aggregated patterns
        self.scratch_dir: str = '/scratch/diffractem'#: scratch directory for temporary data
        self.proc_dir: str = 'proc_data'             #: directory for pre-processed data
        self.rechunk: bool = None
        self.peak_search_params: dict = \
            {'min-res': 5, 'max-res': 600,
            'local-bg-radius': 3, 'threshold': 8,
            'min-pix-count': 3,
            'min-snr': 3, 'int-radius': '3,4,5',
            'peaks': 'peakfinder8'}             #: parameters for peak finding using peakfinder8
        self.indexing_params: dict = \
            {'min-res': 0, 'max-res': 400, 'local-bg-radius': 4,
            'threshold': 10, 'min-pix-count': 3, 'min-snr': 5,
            'peaks': 'peakfinder8', 
            'indexing': 'none'}                 #: indexamajig parameters for indexing (of virtual files)
        self.integration_params: dict = \
            {'min-res': 0, 'max-res': 400, 'local-bg-radius': 4,
            'threshold': 10, 'min-pix-count': 3, 'min-snr': 5,
            'peaks': 'peakfinder8', 
            'indexing': 'none'}                 #: indexamajig parameters for integration
        self.peak_search_params.update({'temp-dir': self.scratch_dir})
        self.indexing_params.update({'temp-dir': self.scratch_dir})
        self.max_peaks: int = 500                    #: maximum number of peaks for peak finding
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