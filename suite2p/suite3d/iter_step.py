
from cmath import log
import os
from turtle import ht
import numpy as n
import copy
from multiprocessing import shared_memory, Pool
from scipy.ndimage import uniform_filter
from dask import array as darr
import time
from ..io import lbm as lbmio 
from ..registration import register
from . import utils
# from . import deepinterp as dp

from ..detection import utils as det_utils
from ..detection import detection3d as det3d
from ..detection import svd_utils as svu

import tracemalloc
import traceback
import gc
import threading
import psutil

def default_log(string, *args, **kwargs): 
    print(string)


def init_batches(tifs, batch_size, max_tifs_to_analyze=None):
    if max_tifs_to_analyze is not None and max_tifs_to_analyze > 0:
        tifs = tifs[:max_tifs_to_analyze]
    n_tifs = len(tifs)
    n_batches = int(n.ceil(n_tifs / batch_size))

    batches = []
    for i in range(n_batches):
        batches.append(tifs[i*batch_size : (i+1) * batch_size])

    return batches


def calculate_corrmap(mov, params, dirs, log_cb = default_log, save=True, return_mov_filt=False,iter_limit=None,
                      iter_dir_tag = 'iters', mov_sub_dir_tag = 'mov_sub'):
    # TODO This can be accelerated 
    # np sub and convolution takes about ~1/2 of the time, that is parallelized
    # reminaing 1/2 of runtime is single-core, so overall improvement capped around 2x speed 
    # might be worth it

    t_batch_size = params['t_batch_size']
    temporal_hpf = params['temporal_hpf']
    npil_filt_xy = params['npil_filt_xy']
    npil_filt_z = params['npil_filt_z']
    conv_filt_xy = params['conv_filt_xy']
    conv_filt_z = params['conv_filt_z']
    intensity_thresh = params.get('intensity_thresh', 0)
    dtype = params['dtype']
    n_proc_corr = params['n_proc_corr']
    mproc_batchsize = params['mproc_batchsize'] 
    npil_filt_size = (npil_filt_z, npil_filt_xy, npil_filt_xy)
    unif_filt_size = (conv_filt_z, conv_filt_xy, conv_filt_xy)
    do_sdnorm = params.get('do_sdnorm','True')
    fix_vmap_edges = params.get('fix_vmap_edges','True')

    conv_filt_type = params.get('conv_filt_type', 'unif')
    npil_filt_type = params.get('npil_filt_type', 'unif')
    log_cb("Using conv_filt: %s, %.2f, %.2f" % (conv_filt_type, conv_filt_z, conv_filt_xy), 1)
    log_cb("Using np_filt: %s, %.2f, %.2f" % (npil_filt_type, npil_filt_z, npil_filt_xy), 1)

    reconstruct_svd = False
    if type(mov) == dict:
        svd_info = mov
        svd_root = '\\'.join(svd_info['svd_dirs'][0].split('\\')[:-2])
        n_comps = params.get('n_svd_comps', svd_info['n_comps'])
        crop = params.get('svd_crop_z', None)
        log_cb("Will reconstruct SVD movie on-the-fly from %s with %d components" % (svd_root, n_comps))
        nz, nt, ny, nx = svd_info['mov_shape']
        if crop is not None: 
            nz = crop[1] - crop[0]
            log_cb("Cropping z from %d to %d" % crop, 3)
        reconstruct_svd = True
    else:
        nz, nt, ny, nx = mov.shape
        flip_shape = True
        if nt < nz:
            nt, nz, ny, nx = mov.shape
            flip_shape = False
            log_cb("Shape is unexpected (%s). Modifying such that nt: %d and nz: %d" % (str(mov.shape), nt, nz))

    n_batches = int(n.ceil(nt / t_batch_size))
    if save:
        batch_dirs, __ = init_batch_files(dirs[iter_dir_tag], makedirs=True, n_batches=n_batches)
        __, mov_sub_paths = init_batch_files(None, dirs[mov_sub_dir_tag], makedirs=False, n_batches=n_batches, filename='mov_sub')
        # print(mov_sub_paths)
        log_cb("Created files and dirs for %d batches" % n_batches, 1)
    else: mov_sub_paths = [None] * n_batches

    vmap2 = n.zeros((nz,ny,nx))
    mean_img = n.zeros((nz,ny,nx))
    max_img = n.zeros((nz,ny,nx))
    sdmov2 = n.zeros((nz,ny,nx))
    n_frames_proc = 0 
    for batch_idx in range(n_batches):
        if iter_limit is not None and batch_idx == iter_limit:
            break
        log_cb("Running batch %d of %d" % (batch_idx + 1, n_batches), 2)
        st_idx = batch_idx * t_batch_size
        end_idx = min(nt, st_idx + t_batch_size)
        n_frames_proc += end_idx - st_idx
        if reconstruct_svd:
            recon_tic = time.time()
            log_cb("Reconstructing from svd", 3)
            movx = svu.reconstruct_overlapping_movie(svd_info, t_indices = (st_idx, end_idx),n_comps=n_comps, crop_z = crop)
            log_cb("Reconstructed in %.2f seconds" % (time.time() - recon_tic),3 )
        else:
            if flip_shape:
                movx = mov[:,st_idx:end_idx]
                movx = darr.swapaxes(movx, 0, 1).compute().astype(dtype)
            else:
                movx = mov[st_idx:end_idx]
                try:
                    movx = movx.compute()
                except:
                    log_cb("Not a dask array", 3)
                movx = movx.astype(dtype)
        log_cb("Loaded and swapped, idx %d to %d" % (st_idx, end_idx), 2)
        log_cb("Calculating corr map",2)
        mov_filt = calculate_corrmap_for_batch(movx, sdmov2, vmap2, mean_img, max_img, temporal_hpf, npil_filt_size, unif_filt_size, intensity_thresh,
                                    n_frames_proc, n_proc_corr, mproc_batchsize, mov_sub_save_path=mov_sub_paths[batch_idx],do_sdnorm=do_sdnorm,
                                    log_cb=log_cb, return_mov_filt=return_mov_filt, fix_vmap_edges=fix_vmap_edges,
                                               conv_filt_type=conv_filt_type, np_filt_type=npil_filt_type, dtype=dtype)
        if save:
            log_cb("Saving to %s" % batch_dirs[batch_idx],2)
            n.save(os.path.join(batch_dirs[batch_idx], 'vmap2.npy'), vmap2)
            n.save(os.path.join(batch_dirs[batch_idx], 'mean_img.npy'), mean_img)
            n.save(os.path.join(batch_dirs[batch_idx], 'max_img.npy'), max_img)
        gc.collect()
    vmap = vmap2 ** 0.5
    if fix_vmap_edges and nz > 1:
        vmap[0] = vmap[0] * vmap[1].mean() / vmap[0].mean()
        vmap[-1] = vmap[-1] * vmap[-2].mean() / vmap[-1].mean()
    if save: n.save(os.path.join(batch_dirs[batch_idx], 'vmap.npy'), vmap)
    if return_mov_filt:
        return mov_filt, vmap
    return vmap


    
def calculate_corrmap_for_batch(mov, sdmov2, vmap2, mean_img, max_img, temporal_hpf, npil_filt_size, unif_filt_size, intensity_thresh, n_frames_proc=0,n_proc=12, mproc_batchsize = 50, mov_sub_save_path=None, log_cb=default_log, return_mov_filt=False, do_sdnorm=True, np_filt_type='unif', conv_filt_type = 'unif' , fix_vmap_edges=True, dtype=None):
    if dtype is None: dtype = n.float32
    nt, nz, ny, nx = mov.shape
    log_cb("Rolling mean filter", 3)
    mean_img[:] = mean_img * (n_frames_proc - nt) / n_frames_proc + mov.mean(axis=0) * nt / n_frames_proc
    max_img[:] = n.maximum(max_img, mov.max(axis=0))

    # shmem_mov_sub, shmem_par_mov_sub, mov_sub = utils.create_shmem_from_arr(mov, copy=True)
    # del mov
    mov = det_utils.hp_rolling_mean_filter(mov, temporal_hpf, copy=False)
    # det3d.hp_rolling_mean_filter_mp(
        # shmem_par_mov_sub, temporal_hpf, nz=nz, n_proc=n_proc)
    # print(mov_sub.std())
    # return
    log_cb("Stdev over time",3)
    if do_sdnorm:
        sdmov2 += det3d.standard_deviation_over_time(mov, batch_size=nt, sqrt=False)
        sdmov = n.sqrt(n.maximum(1e-10, sdmov2 / n_frames_proc))
    else:
        log_cb("Skipping sdnorm", 3)
        sdmov = 1
    mov[:] = mov[:] / sdmov
    if return_mov_filt:
        sdnorm_mov = mov.copy()
    log_cb("Sharr creation",3)
    shmem_mov_sub, shmem_par_mov_sub, mov_sub = utils.create_shmem_from_arr(mov, copy=True)
    del mov
    shmem_mov_filt, shmem_par_mov_filt, mov_filt = utils.create_shmem_from_arr(
        mov_sub, copy=False)
    log_cb("Sub and conv", 3)
    det3d.np_sub_and_conv3d_split_shmem(
        shmem_par_mov_sub, shmem_par_mov_filt, npil_filt_size, unif_filt_size, n_proc=n_proc, batch_size=mproc_batchsize,
        np_filt_type=np_filt_type, conv_filt_type = conv_filt_type)
    if mov_sub_save_path is not None:
        n.save(mov_sub_save_path, mov_sub.astype(dtype))
    log_cb("Vmap", 3)
    vmap2 += det3d.get_vmap3d(mov_filt, intensity_thresh,
                              sqrt=False, mean_subtract=False, fix_edges=fix_vmap_edges)
    if return_mov_filt:
        retfilt = mov_filt.copy()
        retsub = mov_sub.copy()
    shmem_mov_sub.close(); shmem_mov_sub.unlink()
    shmem_mov_filt.close(); shmem_mov_filt.unlink()
    if return_mov_filt:
        return (sdnorm_mov, retfilt, retsub)
    else:
        return None


def run_detection(mov3d_in, vmap2, sdmov2,n_frames_proc, temporal_high_pass_width, do_running_sdmov,
                    npil_hpf_xy, npil_hpf_z, unif_filter_xy, unif_filter_z, intensity_thresh,
                    log_cb = default_log, n_proc=10, batch_size=10):
    nt, nz, ny, nx = mov3d_in.shape
    n_frames_proc_new = n_frames_proc + nt
    log_cb("Temporal high-pass filtering",2)
    log_cb("Begin Detect", level=4, log_mem_usage=True)
    if n_proc > 1:
        # this is the cause of a memory leak, it seems? probably fixed
        shmem, shmem_par, mov3d = utils.create_shmem_from_arr(mov3d_in, copy=True)
    else:
        mov3d = mov3d_in
    log_cb("After shmem", level=4, log_mem_usage=True)
    mov3d = det_utils.temporal_high_pass_filter(
        mov3d, temporal_high_pass_width, copy=False)
    log_cb("After hpf", level=4, log_mem_usage=True)

    log_cb("Computing standard deviation of pixels over time",2)
    if do_running_sdmov:
        sdmov2 += det3d.standard_deviation_over_time(mov3d, batch_size=mov3d.shape[0], sqrt=False)
        sdmov = n.sqrt(n.maximum(1e-10, sdmov2 / n_frames_proc_new))
    else:
        sdmov = det3d.standard_deviation_over_time(mov3d, batch_size=mov3d.shape[0], sqrt=True)
    log_cb("After sdmov", level=4, log_mem_usage=True)
    
    if n_proc == 1:
        log_cb("Neuropil subtraction",2)
        mov3d = det3d.neuropil_subtraction(mov3d / sdmov, npil_hpf_xy, npil_hpf_z)
        log_cb("Square convolution",2)
        mov_u0 = det3d.square_convolution_2d(
            mov3d, filter_size=unif_filter_xy, filter_size_z=unif_filter_z)
        log_cb("Vmap calculation", 2)
        vmap2 += det3d.get_vmap3d(mov_u0, intensity_thresh, sqrt=False)
    else:
        log_cb("Why is the scale different between n_proc=1 and n_proc > 1")
        log_cb("Neuropil subtraction and convolution", 2)
        mov3d[:] = mov3d[:] / sdmov
        log_cb("After Norm", level=4, log_mem_usage=True)
        filt_size = (npil_hpf_z, npil_hpf_xy, npil_hpf_xy)
        conv_filt_size = (unif_filter_z, unif_filter_xy, unif_filter_xy)
        log_cb("Before 3D filter", level=4, log_mem_usage=True)
        det3d.np_sub_and_conv3d_shmem(
            shmem_par, filt_size, conv_filt_size, n_proc=n_proc, batch_size=batch_size)
        log_cb("After 3D filter", level=4, log_mem_usage=True)
        log_cb("Vmap calculation", 2)
        vmap2 += det3d.get_vmap3d(mov3d, intensity_thresh, sqrt=False)
        log_cb("After vmap", level=4, log_mem_usage=True)
    if n_proc > 1:
        mov3d_in[:] = mov3d[:]
        log_cb("Before unlink", level=4, log_mem_usage=True)
        shmem.close()
        shmem.unlink()
        del mov3d; del shmem
        log_cb("After unlink", level=4, log_mem_usage=True)
    else:
        mov3d_in[:] = mov3d[:]
    log_cb("Before return", level=4, log_mem_usage=True)
    return mov3d_in

def register_mov(mov3d, refs_and_masks, all_ops, log_cb = default_log, convolve_method='fast_cpu', do_rigid=True):
    nz, nt, ny, nx = mov3d.shape
    all_offsets = {'xms' : [],
                   'yms' : [],
                   'cms' : [],
                   'xm1s': [],
                   'ym1s': [],
                   'cm1s': []}
    for plane_idx in range(nz):
        log_cb("Registering plane %d" % plane_idx, 2)
        mov3d[plane_idx], ym, xm, cm, ym1, xm1, cm1 = register.register_frames(
            refAndMasks = refs_and_masks[plane_idx],
            frames = mov3d[plane_idx],
            ops = all_ops[plane_idx], convolve_method=convolve_method, do_rigid=do_rigid)
        all_offsets['xms'].append(xm); all_offsets['yms'].append(ym); all_offsets['cms'].append(cm)
        all_offsets['xm1s'].append(xm1); all_offsets['ym1s'].append(ym1); all_offsets['cm1s'].append(cm1)
    return all_offsets
     

def fuse_movie(mov, n_skip, centers, shift_xs):
    n_skip_l = n_skip // 2
    n_skip_r = n_skip - n_skip_l
    nz, nt, ny, nx = mov.shape

    centers = n.concatenate([centers , [nx]])
    # print(centers)
    n_seams = len(centers)
    nxnew = nx - (n_skip) * (n_seams )
    # print(nxnew)
    mov_fused = n.zeros((nz, nt, ny, nxnew), dtype=mov.dtype)

    for zidx in range(nz):
        curr_x = 0
        curr_x_new = 0
        for i in range(n_seams):
            wid = (centers[i] + shift_xs[zidx]) - curr_x
            mov_fused[zidx, :, :, curr_x_new: curr_x_new + wid - n_skip] = \
                mov[zidx, :, :, curr_x + n_skip_l: curr_x + wid - n_skip_r]
            curr_x_new += wid - n_skip
            curr_x += wid

    return mov_fused


def fuse_and_save_reg_file(reg_file, reg_fused_dir, centers, shift_xs, n_skip, crops=None, mov=None, save=True, delete_original=False):
    file_name = reg_file.split(os.sep)[-1]
    fused_file_name = os.path.join(reg_fused_dir, 'fused_' + file_name)
    if mov is None: 
        print("Loading")
        mov = n.load(reg_file)
        print("Loaded")

    mov_fused = fuse_movie(mov, n_skip, centers, shift_xs)
    
    if crops is not None:
        mov_fused = mov_fused[crops[0][0]:crops[0][1], :, crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
    if delete_original:
        print("Delelting file: %s" % reg_file)
        os.remove(reg_file)
    if save: 
        n.save(fused_file_name, mov_fused)
        return fused_file_name
    else: return mov_fused



def fuse_and_save_reg_file_old(reg_file, reg_fused_dir, centers, shift_xs, nshift, nbuf, crops=None, mov=None, save=True, delete_original=False):
    file_name = reg_file.split(os.sep)[-1]
    fused_file_name = os.path.join(reg_fused_dir, 'fused_' + file_name)
    if mov is None: 
        print("Loading")
        mov = n.load(reg_file)
        print("Loaded")
    nz, nt, ny, nx = mov.shape
    weights = n.linspace(0, 1, nshift)
    n_seams = len(centers)
    nxnew = nx - (nshift + nbuf) * n_seams
    mov_fused = n.zeros((nz, nt, ny, nxnew), dtype=mov.dtype)
    print("Looping")
    for zidx in range(nz):
        print(zidx)
        curr_x = 0
        curr_x_new = 0
        for i in range(n_seams):
            wid = (centers[i] + shift_xs[zidx]) - curr_x

            mov_fused[zidx, :, :, curr_x_new: curr_x_new + wid -
                        nshift] = mov[zidx, :, :, curr_x: curr_x + wid - nshift]
            mov_fused[zidx, :, :, curr_x_new + wid - nshift: curr_x_new + wid] =\
                (mov[zidx, :, :, curr_x + wid - nshift: curr_x + wid]
                    * (1 - weights)).astype(n.int16)
            mov_fused[zidx, :, :, curr_x_new + wid - nshift: curr_x_new + wid] +=\
                (mov[zidx, :, :, curr_x + wid + nbuf: curr_x +
                    wid + nbuf + nshift] * (weights)).astype(n.int16)

            curr_x_new += wid
            curr_x += wid + nbuf + nshift
        mov_fused[zidx, :, :, curr_x_new:] = mov[zidx, :, :, curr_x:]
    if crops is not None:
        mov_fused = mov_fused[crops[0][0]:crops[0][1], :, crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
    if delete_original:
        print("Delelting file: %s" % reg_file)
        os.remove(reg_file)
    if save: 
        n.save(fused_file_name, mov_fused)
        return fused_file_name
    else: return mov_fused


def init_batch_files(job_iter_dir=None, job_reg_data_dir=None, n_batches=1, makedirs = True, filename='reg_data', dirname='batch'):
    reg_data_paths = []
    batch_dirs = []
    for batch_idx in range(n_batches):
        if job_reg_data_dir is not None:
            reg_data_filename = filename+'%04d.npy' % batch_idx
            reg_data_path = os.path.join(job_reg_data_dir, reg_data_filename)
            reg_data_paths.append(reg_data_path)

        if makedirs:
            assert job_iter_dir is not None
            batch_dir = os.path.join(job_iter_dir, dirname + '%04d' % batch_idx)
            os.makedirs(batch_dir, exist_ok=True)
            batch_dirs.append(batch_dir)

    return batch_dirs, reg_data_paths

        

def subtract_crosstalk(shmem_params, coeff = None, planes = None, n_procs=15, log_cb = default_log):

    assert coeff is not None

    if planes is None:
        pairs = [(i, i+15) for i in range(15)]
    else:
        pairs = []
        for plane_idx in planes:
            if plane_idx > 15:
                if plane_idx - 15 in planes:
                    pairs.append((plane_idx-15, plane_idx))
                    log_cb("Subtracting plane %d from %d" % (pairs[-1][0], pairs[-1][1]), 2)
                else:
                    log_cb("Plane %d does not have its pair %d" % (plane_idx, plane_idx-15),0)
    # print(pairs)
    p = Pool(n_procs)
    p.starmap(subtract_crosstalk_worker, [(shmem_params, coeff, pair[0], pair[1]) \
                                                    for pair in pairs])

    return coeff

def subtract_crosstalk_worker(shmem_params, coeff, deep_plane_idx, shallow_plane_idx):
    shmem, mov3d = utils.load_shmem(shmem_params)
    # print(mov3d.shape, shallow_plane_idx, deep_plane_idx)
    mov3d[shallow_plane_idx] = mov3d[shallow_plane_idx] - coeff * mov3d[deep_plane_idx]
    utils.close_shmem(shmem_params)
    
    


def register_dataset(tifs, params, dirs, summary, log_cb = default_log,
                    start_batch_idx = 0):

    ref_img_3d = summary['ref_img_3d']
    crosstalk_coeff = summary['crosstalk_coeff']
    refs_and_masks = summary.get('refs_and_masks', None)
    all_ops = summary.get('all_ops',None)
    job_iter_dir = dirs['iters']
    job_reg_data_dir = dirs['registered_data']
    n_tifs_to_analyze = params.get('total_tifs_to_analyze', len(tifs))
    tif_batch_size = params['tif_batch_size']
    planes = params['planes']
    notch_filt = params['notch_filt']
    do_subtract_crosstalk = params['subtract_crosstalk']
    mov_dtype = params['dtype']

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    log_cb("Will analyze %d tifs in %d batches" % (len(n.concatenate(batches)), len(batches)),0)

    # init accumulators
    nz, ny, nx = ref_img_3d.shape
    n_frames_proc = 0
    
    __, reg_data_paths = init_batch_files(job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename='reg_data')
    __, offset_paths = init_batch_files(job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename='offsets')

    loaded_movs = [0]

    def io_thread_loader(tifs, batch_idx):
        log_cb("   [Thread] Loading batch %d \n" % batch_idx, 20)
        loaded_mov = lbmio.load_and_stitch_tifs(tifs, planes, filt = notch_filt, concat=True, log_cb=log_cb)
        log_cb("   [Thread] Loaded batch %d \n" % batch_idx, 20)
        loaded_movs[0] = loaded_mov
        
        log_cb("   [Thread] Thread for batch %d ready to join \n" % batch_idx, 20)
    log_cb("Launching IO thread")
    io_thread = threading.Thread(target=io_thread_loader, args=(batches[start_batch_idx], start_batch_idx))
    io_thread.start()

    for batch_idx in range(start_batch_idx, n_batches):
        try:
            log_cb("Start Batch: ", level=3,log_mem_usage=True )
            reg_data_path = reg_data_paths[batch_idx]
            offset_path = offset_paths[batch_idx]
            log_cb("Loading Batch %d of %d" % (batch_idx+1, n_batches), 0)
            io_thread.join()
            log_cb("Batch %d IO thread joined" % (batch_idx))
            log_cb('After IO thread join', level=3,log_mem_usage=True )
            shmem_mov,shmem_mov_params, mov = utils.create_shmem_from_arr(loaded_movs[0], copy=True)
            log_cb("After Sharr creation:", level=3,log_mem_usage=True )
            if batch_idx + 1 < n_batches:
                log_cb("Launching IO thread for next batch")
                io_thread = threading.Thread(target=io_thread_loader, args=(batches[batch_idx+1], batch_idx+1))
                io_thread.start()
                log_cb("After IO thread launch:", level=3,log_mem_usage=True )
            if do_subtract_crosstalk:
                __ = subtract_crosstalk(shmem_mov_params, crosstalk_coeff, planes = planes, log_cb = log_cb)
            log_cb("Registering Batch %d" % batch_idx, 1)
            
            log_cb("Before Reg:", level=3,log_mem_usage=True )
            log_cb()
            all_offsets = register_mov(mov,refs_and_masks, all_ops, log_cb)
            log_cb("Saving registered file to %s" % reg_data_path, 2)
            n.save(reg_data_path, mov)
            n.save(offset_path, all_offsets)
            log_cb("After reg:", level=3,log_mem_usage=True )

            nz, nt, ny, nx = mov.shape
            n_frames_proc_new = n_frames_proc + nt

            n_cleared = gc.collect()
            log_cb("Garbage collected %d items" %n_cleared, 2)
            log_cb("After gc collect: ", level=3,log_mem_usage=True )
        except Exception as exc:
            log_cb("Error occured in iteration %d" % batch_idx, 0 )
            tb = traceback.format_exc()
            log_cb(tb, 0)
            break
