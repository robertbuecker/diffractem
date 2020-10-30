import warnings

import matplotlib as mpl
import numpy

import numpy as np
import pandas
import pandas as pd
from dask import array as da
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from diffractem.io import get_meta_lists, get_data_stacks, expand_files
from diffractem.nexus import get_meta_fields
from diffractem.proc2d import correct_dead_pixels, lorentz_fit_simple
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans


def get_data_stacks(filename, base_path='/%/data', labels=None):
    # Internally, this function is structured 99% as get_meta_lists, just operating on dask
    # arrays, not pandas frames
    fns = expand_files(filename)
    identifiers = base_path.rsplit('%', 1)
    stacks = defaultdict(list)

    for fn in fns:
        fh = h5py.File(fn)

        try:
            if len(identifiers) == 1:
                base_grp = {'': fh[identifiers[0]]}
            else:
                base_grp = fh[identifiers[0]]
            for subset, ssgrp in base_grp.items():

                if (len(identifiers) > 1) and identifiers[1]:
                    if identifiers[1].strip('/') in ssgrp.keys():
                        grp = ssgrp[identifiers[1].strip('/')]
                else:
                    grp = ssgrp  # subset identifier is on last level

                if isinstance(grp, h5py.Group):
                    for dsname, ds in grp.items():
                        if ds is None:
                            # can happen for dangling soft links
                            continue
                        if ((labels is None) or (dsname in labels)) \
                                and isinstance(ds, h5py.Dataset) \
                                and ('pandas_type' not in ds.attrs):
                            stacks[dsname].append(da.from_array(ds, chunks=ds.chunks))

        except Exception as err:
            fh.close()
            raise err

    stacks = {sn: da.concatenate(s, axis=0) for sn, s in stacks.items()}
    return stacks


def peak_finder(img_raw, xy, profile = None, pxmask = None, noise = np.array
                ([[False,True,False],[True,True,True],[False,True,False]]),
                 kernel_size = 9, threshold = 9, db_eps = 1.9,
                 db_samples = 9, bkg_remove = False, img_clean = True,
                 radial_min = 5):
    """
    1) (OPTIONAL) Construct background from xy and profile.
        Subtract from img_raw.
    2) Remove Impulse noise using morpholigical opening and closing with
        noise struc.
    3) Convolve Image with Gaussian kernel to find local background.
        Features are all pixels larger than the local background
        by a certain threshold.
    4) Pick peaks out of image features using DBSCAN (Clustering)
    5) Labelling of Peaks using ndimage
    6) (OPTIONAL) Clean image
    Simple peakfinder built upon scipy.ndimage.
    Uses a morpholgical opening to find features larger than size and higher
    than threshold. The features are then labelled and the center of mass and
    sum of each peak is returned in a numpy array.
    """
    ### 1) Strip Background
    if bkg_remove:
        ylen,xlen = img_raw.shape
        y,x = np.ogrid[0:ylen,0:xlen]
        radius = (np.rint(((x-xy[0])**2 + (y-xy[1])**2)**0.5)).astype(np.int32)
        prof = np.zeros(1+np.max(radius))
        np.copyto(prof[0:len(profile)], profile)
        bkg = prof[radius]
        img = img_raw - bkg
        img = correct_dead_pixels(img, pxmask, 'replace', replace_val=-1,
                              mask_gaps=True)
    else:
        img = img_raw
        bkg=None
    ### 2) Remove Impulse Noise
    img = ndimage.morphology.grey_opening(img, structure=noise)
    img = ndimage.morphology.grey_closing(img, structure=noise)
    ### 3) Feature detection NB: astropy.convolve is slow
    img_fil = np.where(img==-1,0,img)
    img_fil = ndimage.gaussian_filter(img_fil.astype(np.float),
                                      kernel_size,mode='constant',cval=0)
    img_norm = np.where(img==-1,0,1)
    img_norm = ndimage.gaussian_filter(img_norm.astype(np.float),
                                       kernel_size,mode='constant',cval=0)
    img_norm = np.where(img_norm==0.0,1,img_norm)
    img_fil = img_fil/img_norm
    img_feat = img - img_fil
    ### 4) Peak Picking
    loc = np.where(img_feat > threshold)
    X = np.transpose(np.stack([loc[1], loc[0] , np.log(img_feat[loc])]))
    db = DBSCAN(eps=db_eps, min_samples=db_samples, n_jobs=-1).fit(X)
    img_label = np.zeros(img.shape)
    img_label[loc] = 1 + db.labels_
    num_feat = len(set(db.labels_))
    ### 5) Peak Labelling
    com = ndimage.center_of_mass(img_feat,img_label,np.arange(1, num_feat))
    vol = ndimage.sum(img_feat,img_label,np.arange(1, num_feat))
    ### 6) Apply radial cut to peaks
    rad = np.sqrt(np.sum(np.square(np.array(com)-xy[::-1]),axis=1))
    rad_cut = rad > radial_min
    com = np.array(com)[rad_cut]
    vol = vol[rad_cut]
    img_label[img_label-1 == np.where(rad<radial_min)] = 0
    print("Found {} peaks".format(len(com)))
    ### 7) Clean Image
    if img_clean:
        img[img_label==0] = 0
    return np.column_stack((com,vol)), img


def profile_classify_ML(prof_stack, fit_record = None, show_plot = False,
                        refit = True, rescale = True, verbose = True,
                        bin_min = 5, bin_max = 40):
    """
    Takes the profile stack,
    refits a 1D Lorentz distribution(Optional),
    classifies the profiles with KMeans form sklearn.cluster,
    NB MinibatchKMeans is not parrallelized
    takes the mean, and returns them to the correct index in profile_ave.
    :param prof_stack   : stack of profiles
    :fit_record         : array of parameters returned form the 2D lorentz fit
    :show_plot          : switch for plotting the profiles
    :refit              : switch for refitting of the profiles
    :rescale            : if profiles should be scaled.
    """
    #scale,shape = np.transpose([rec[0][3:5] for rec in fit_record])
    n=prof_stack.shape[0]
    amp = np.zeros(n)
    scale = np.zeros(n)
    shape = np.zeros(n)
    prof_ave = np.zeros(prof_stack.shape)

    ###(Optional) Refitting of the profiles with a Lorentz distribution
    if refit or fit_record is None:
        prof_stack = prof_stack[:,:,np.newaxis]
        amp, scale, shape = (
                np.transpose(lorentz_fit_simple(prof_stack,bin_min,bin_max)))
        prof_stack = prof_stack.squeeze()
    else:
        amp, scale, shape = np.transpose(fit_record[:,[0,3,4]])

    if rescale:
        prof_stack = np.transpose(np.transpose(prof_stack)/amp)

    ###Grouping of profiles according to scale and shape
    X = np.transpose([100*scale,100*shape])
    # n_jobs=-1 Uses all cores
    cluster = KMeans(n_jobs=-1, n_clusters=int(np.sqrt(n))).fit(X)
    labels = cluster.labels_
    unique_labels = set(labels)
    print('Grouping the radial profiles into {} groups.'
          .format(len(unique_labels)))
    if show_plot :
        fig = plt.figure(0)
        fig.clf()
        plt.scatter(X[:, 0], X[:, 1], c = labels, marker = 'o')
        plt.title('Estimated number of clusters: {}'.
                  format(len(unique_labels)))
        plt.show()

    ###Taking Average of profiles
    prof_stack[prof_stack == -1] = np.nan
    for kk in unique_labels:
        cut = (kk==labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ave = np.nanmean(prof_stack[cut],axis=0)
        ave = np.nan_to_num(ave)
        prof_ave[cut] = ave

    if rescale:
        prof_ave = np.transpose(np.transpose(prof_ave)*amp)
    return prof_ave, labels


def profile_classify(prof_stack, fit_record = None, show_plot = False,
                     show_fit = True, refit = True, rescale = True,
                     verbose = True, bin_min = 5, bin_max = 40):
    """
    Takes the profile stack,
    refits a 1D Lorentz distribution(Optional),
    groups the profiles with the same shape and scale together,
    takes the mean and std of the group,
    plots all the profiles in the group on the same figure(Optional)
    and returns them to the correct index in profile_ave and profile_std.
    :param prof_stack   : stack of profiles
    :fit_record         : array of parameters returned form the 2D lorentz fit
    :show_plot          : switch for plotting the profiles
    :refit              : switch for refitting of the profiles
    :rescale            : if profiles should be scaled.
    """
    #scale,shape = np.transpose([rec[0][3:5] for rec in fit_record])
    n=prof_stack.shape[0]
    amp = np.zeros(n)
    scale = np.zeros(n)
    shape = np.zeros(n)
    prof_ave = np.zeros(prof_stack.shape)
    prof_std = np.zeros(prof_stack.shape)

    ###(Optional) Refitting of the profiles with a Lorentz distribution
    if refit or fit_record is None:
        prof_stack = prof_stack[:,:,np.newaxis]
        amp, scale, shape = (
                np.transpose(lorentz_fit_simple(prof_stack,bin_min,bin_max)))
        prof_stack = prof_stack.squeeze()
    else:
        amp, scale, shape = np.transpose(fit_record[:,[0,3,4]])

    if rescale:
        prof_stack = np.transpose(np.transpose(prof_stack)/amp)

    ###Grouping of profiles according to scale and shape
    keys = np.rint(scale*10)*100 + np.rint(shape*10)
    unique_keys, inverse, counts = np.unique(keys.astype(np.int32),
                                             return_inverse = True,
                                             return_counts = True)
    prof_stack[prof_stack == -1] = np.nan
    print('Grouping the radial profiles into {} groups:'
          .format(len(unique_keys)))
    print('Categories: {}'.format(unique_keys))
    print('    Counts: {}'.format(counts))
    for kk in unique_keys:
        cut = (kk==keys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ave = np.nanmean(prof_stack[cut],axis=0)
            std = np.nanstd(prof_stack[cut],axis=0)
        ave = np.nan_to_num(ave)
        std = np.nan_to_num(std)
        if show_plot:
            fig = plt.figure(kk)
            fig.clf()
            plt.plot(np.transpose(prof_stack[keys==kk,0:20]))
        ###(Optional) Plotting of the first 20 bins of the profiles
        prof_ave[cut] = ave
        prof_std[cut] = std

    if rescale:
        prof_ave = np.transpose(np.transpose(prof_ave)*amp)
        prof_std = np.transpose(np.transpose(prof_std)*amp)
    return prof_ave, prof_std, inverse, counts


def process_stack(imgs, ops, execution='threads', **kwargs):
    """
    Applies correction function, or a pipeline thereof, to a stack of images, using different ways of execution.
    Essentially very similar to map function of hyperspy.
    :param imgs: 3D image stack or 2D single image
    :param ops: function handle to processing function, or list/tuple thereof.
    :param execution: method of execution. Options are
        'threads': process each image in parallel using threads
        'processes': process each image in parallel using processes
        'stack': pass the entire stack to the ops functions. All ops must be able to handle stack inputs
        'loop': loop through the images. Use for debugging only!
    :param kwargs: keyword arguments to be passed to the correction functions
    :return: the processed stack/image
    """

    assert execution.lower() in ('threads', 'processes', 'stack', 'loop')

    is_2D = len(imgs.shape) < 3

    if is_2D:
        imgs.shape = (1,imgs.shape[0],imgs.shape[1])

    imgs_out = imgs.copy()

    if not (isinstance(ops, list) or isinstance(ops, tuple)):
        ops = (ops,)

    if execution.lower() in ['threads', 'processes']:
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        if execution.lower=='processes':
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor
        for func in ops:
            imgs_out = Executor().map(lambda img: func(img, **kwargs), imgs_out)
            imgs_out = np.stack(imgs_out)

    elif execution.lower=='loop':
        for ii, img in enumerate(imgs_out):
            for func in ops:
                imgs_out[ii,:,:] = func(img, **kwargs)

    else:
        for func in ops:
            imgs_out = func(imgs_out, **kwargs)

    if is_2D:
        imgs_out.shape = (imgs_out.shape[1],imgs_out.shape[2])

    return imgs_out


def diff_plot(filename, idcs, setname='centered', beamdiam=100e-9,
              rings=(10, 5, 2.5), radii=(3, 4, 6), show_map=True, show_peaks=True, show_predict=False,
              figsize=(15, 10), dpi=300, cutoff=99.5,
              width=616, xoff=0, yoff=0, ellipticity=0,
              map_px=None, clen=None, det_px=None, wavelength=None,
              stacks=None, shots=None, peaks=None, predict=None,
              base_path='/%/data', map_path='/%/map/image', results_path='/%/results',
              pre_compute=True, store_to=None, cmap='gray', ringcolor='y', **kwargs):
    """

    """

    if shots is None:
        shots = get_meta_lists(filename, base_path, ['shots'])['shots']
    if show_peaks and peaks is None:
        peaks = get_meta_lists(filename, results_path, ['peaks'])['peaks']
    if show_predict and predict is None:
        predict = get_meta_lists(filename, results_path, ['predict'])['predict']
    if stacks is None:
        stacks = get_data_stacks(filename, base_path, [setname])

    # TODO: replace all the following defaults by proper reading from NeXus and assigning to shots
    shotsel = shots.loc[idcs, :].copy()

    if map_px is None:
        map_px = shotsel['map_px'] = 17e-9
    else:
        shotsel['map_px'] = map_px
    if clen is None:
        shotsel['clen'] = 1.57
    else:
        shotsel['clen'] = clen
    if det_px is None:
        shotsel['det_px'] = 55e-6
    else:
        shotsel['det_px'] = det_px
    if wavelength is None:
        shotsel['wavelength'] = 2.5e-12
    else:
        shotsel['wavelength'] = wavelength

    shotsel['recpx'] = shotsel['wavelength'] / (shotsel['det_px'] / shotsel['clen']) * 1e10

    imgs = stacks[setname][shotsel.index.values, ...]
    if pre_compute:
        imgs = imgs.compute()

    if show_map:
        map_path = map_path.replace('%', 'entry')
        map_imgs = meta_from_nxs(list(shotsel['file'].unique()), map_path)[map_path]

    figs = []

    for ii, ((idx, shot), img) in enumerate(zip(shotsel.iterrows(), imgs)):

        if not pre_compute:
            img = img.compute()

        figh = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        figs.append(figh)
        #figh = figs[ii]

        img_ax = figh.add_axes([0, 0, 0.66, 0.95])
        img_ax.imshow(img, vmin=0, vmax=np.quantile(img, cutoff/100), cmap=cmap, label='diff')

        if show_peaks:
            coords = peaks.loc[peaks['serial'] == idx, :]
        else:
            coords = pd.DataFrame()

        if show_predict:
            pred_coords = peaks.loc[predict['serial'] == idx, :]
        else:
            pred_coords = pd.DataFrame()

        img_ax.set_xlim((778 - width/2, 778 + width/2))
        try:
            img_ax.set_title(
                'Set: {}, Shot: {}, Region: {}, Run: {}, Frame: {} \n (#{} in file: {}) PEAKS: {}'.format(shot['subset'],
                                                                                                      idx,
                                                                                                      shot['region'],
                                                                                                      shot['run'],
                                                                                                      shot['frame'],
                                                                                                      shot['shot'],
                                                                                                      shot['file'],
                                                                                                      len(coords), 3))
        except:
            'Shot {}: (file: {}) PEAKS: {}'.format(shot['subset'], shot['file'], len(coords), 3)

        #print(shot['recpx'])
        for res in rings:
            img_ax.add_artist(mpl.patches.Ellipse((img.shape[1] / 2 + xoff, img.shape[0] / 2 + yoff),
                                             width=2*(shot['recpx'] / res), height=2*(shot['recpx'] / res * (1+ellipticity)), edgecolor=ringcolor, fill=False))
            img_ax.text(img.shape[1] / 2 + shot['recpx'] / res / 1.4, img.shape[0] / 2 - shot['recpx'] / res / 1.4, '{} A'.format(res),
                    color=ringcolor)

        for _, c in coords.iterrows():
            img_ax.add_artist(plt.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[0], fill=True, color='r', alpha=0.15))
            img_ax.add_artist(plt.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[1], fill=False, color='y', alpha=0.2))
            img_ax.add_artist(plt.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[2], fill=False, color='y', alpha=0.3))

        for _, c in pred_coords.iterrows():
            img_ax.add_artist(plt.Rectangle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             width=radii[-1], height=radii[-1], fill=False, color='b'))

        img_ax.axis('off')

        if not show_map:
            continue

        #map_px = shotsel['map_px']
        #print(map_px)

        map_ax = figh.add_axes([0.6, 0.5, 0.45, 0.45])
        feat_ax = figh.add_axes([0.6, 0, 0.45, 0.45])

        map_ax.imshow(map_imgs[shot['file']], cmap='gray')
        map_ax.add_artist(plt.Circle((shot['crystal_x'], shot['crystal_y']), facecolor='r'))
        map_ax.add_artist(AnchoredSizeBar(map_ax.transData, 5e-6 / map_px, '5 um', 'lower right'))
        map_ax.axis('off')

        feat_ax.imshow(map_imgs[shot['file']], cmap='gray')
        #feat_ax.add_artist(AnchoredSizeBar(feat_ax.transData, 0.1e-6 / map_px, '100 nm', 'lower right'))
        feat_ax.add_artist(plt.Circle((shot['crystal_x'], shot['crystal_y']), radius=beamdiam/2/map_px, color='r', fill=False))
        if not np.isnan(shot['crystal_x']):
            feat_ax.set_xlim(shot['crystal_x'] + np.array([-20, 20]))
            feat_ax.set_ylim(shot['crystal_y'] + np.array([-20, 20]))
        else:
            feat_ax.set_xlim(shot['pos_x'] + np.array([-20, 20]))
            feat_ax.set_ylim(shot['pos_y'] + np.array([-20, 20]))
        feat_ax.axis('off')

        if store_to is not None:
            plt.savefig('{}/{}_{:04d}'.format(store_to, filename.rsplit('.', 1)[0].rsplit('/', 1)[-1], idx))
            plt.close(plt.gcf())

    return figs


def region_plot(file_name, regions=None, crystal_pos=True, peak_ct=True, beamdiam=100e-9, scanpx=2e-8, figsize=(10, 10),
                **kwargs):
    meta = get_meta_lists(file_name)

    cmap = plt.cm.jet
    fhs = []

    if regions is None:
        regions = meta['shots']['region'].drop_duplicates().values

    if not hasattr(regions, '__iter__'):
        regions = (regions,)

    for reg in regions:

        shots = meta['shots'].loc[meta['shots']['region'] == reg, :]

        if not len(shots):
            print('Region {} does not exist. Skipping.'.format(reg))
            continue

        shot = shots.iloc[0, :]

        fh = plt.figure(figsize=figsize, **kwargs)
        fhs.append(fh)

        ax = plt.axes()
        ax.set_title('Set: {}, Region: {}, Run: {}, # Crystals: {}'.format(shot['subset'], shot['region'], shot['run'],
                                                                           shots['crystal_id'].max()))

        stem = get_meta_array(file_name, 'stem', shot)
        ax.imshow(stem, cmap='gray')

        if 'acqdata' in meta.keys():
            acqdata = meta['acquisition_data'].loc[shot['file']]
            pxs = float(acqdata['Scanning_Pixel_size_x'])
        else:
            pxs = scanpx * 1e9

        if crystal_pos and peak_ct:
            norm = int(shots['peak_count'].quantile(0.99))

            def ncmap(x):
                return cmap(x / norm)

            for idx, cr in shots.loc[:, ['crystal_x', 'crystal_y', 'peak_count']].drop_duplicates().iterrows():
                ax.add_artist(plt.Circle((cr['crystal_x'], cr['crystal_y']), radius=beamdiam * 1e9 / 2 / pxs,
                                         facecolor=ncmap(cr['peak_count']), alpha=1))
            # some gymnastics to get a colorbar
            Z = [[0, 0], [0, 0]]
            levels = range(0, norm, 1)
            CS3 = plt.contourf(Z, levels, cmap=plt.cm.jet)
            plt.colorbar(CS3, fraction=0.046, pad=0.04)
            del (CS3)

        elif crystal_pos:
            for idx, cr in shots.loc[:, ['crystal_x', 'crystal_y']].drop_duplicates().iterrows():
                ax.add_artist(
                    plt.Circle((cr['crystal_x'], cr['crystal_y']), radius=beamdiam * 1e9 / 2 / pxs, facecolor='r',
                               alpha=1))

        ax.add_artist(AnchoredSizeBar(ax.transData, 5000 / pxs, '5 um', 'lower right', pad=0.3, size_vertical=1))
        ax.axis('off')

        return fhs


def reduce_stack(filename, first_frame=1, last_frame=-1, aggregate='sum', suffix=None, exclude=(),
                 threads=True, instrument_data=True, label='raw_counts', **kwargs):
    exp_time_field = '/entry/instrument/detector/collection/shutter_time'
    tilt_field = '/entry/instrument/Stage/A'
    if last_frame == -1:
        last_frame = 16383
    exclude = [f'frame < {first_frame}', f'frame > {last_frame}'] + list(exclude)

    from multiprocessing.pool import ThreadPool

    def make_stacks(raw_name):
        sr0 = get_nxs_list(raw_name, 'shots')

        if instrument_data:
            instrument_data = pd.DataFrame(get_meta_fields(raw_name, [exp_time_field, tilt_field])).astype(float)
            instrument_data.columns = ['exp_time', 'tilt_angle']
            instrument_data['tilt_angle'] = (instrument_data['tilt_angle'] * 180 / np.pi).round(1)
            sr0 = sr0.merge(instrument_data, right_index=True, left_on='file')

        sr0['selected'] = True
        for ex in exclude:
            sr0.loc[sr0.eval(ex), 'selected'] = False

        return modify_stack(raw_name, sr0, aggregate=aggregate, labels=label, **kwargs)

    if threads:
        with ThreadPool() as pool:
            stkdat = pool.map(make_stacks, expand_files(filename))
    else:
        stkdat = make_stacks(expand_files(filename))

    shots = pd.concat([s[1] for s in stkdat if s[1].shape[0]], ignore_index=True)
    stack_raw = da.concatenate([s[0][label] for s in stkdat if s[0][label].shape[0]], axis=0)
    shots['file_raw'] = shots['file']

    return None


def get_meta_lists(filename, base_path='/%/data', labels=None):
    warnings.warn('Please use get_meta_list instead if you know what you\'re looking for. It is WAY faster.',
                  DeprecationWarning)
    fns = expand_files(filename)
    identifiers = base_path.rsplit('%', 1)
    lists = defaultdict(list)
    # print(fns)

    for fn in fns:
        # print(fn)
        with h5py.File(fn) as fh:

            if len(identifiers) == 1:
                base_grp = {'': fh[identifiers[0]]}
            else:
                base_grp = fh[identifiers[0]]
            # print(base_grp)
            for subset, ssgrp in base_grp.items():
                # print(list(ssgrp.keys()))
                if (len(identifiers) > 1) and identifiers[1]:
                    if identifiers[1].strip('/') in ssgrp.keys():
                        grp = ssgrp[identifiers[1].strip('/')]
                    else:
                        continue
                else:
                    grp = ssgrp  # subset identifier is on last level

                if isinstance(grp, h5py.Group):
                    # print(grp)
                    for tname, tgrp in grp.items():
                        # print(tname, tgrp)
                        if tgrp is None:
                            # can happen for dangling soft links
                            continue
                        if ((labels is None) or (tname in labels)) and ('table_type' in tgrp.attrs):
                            newlist = pd.read_hdf(fn, tgrp.name)
                            newlist['subset'] = subset
                            newlist['file'] = fn
                            # newlist['shot_in_subset'] = range(newlist.shape[0])

                            lists[tname].append(newlist)
                            # print(f'Appended {len(newlist)} items from {fn}: {subset} -> list {tname}')

    lists = {tn: pd.concat(t, axis=0, ignore_index=True) for tn, t in lists.items()}
    return lists
