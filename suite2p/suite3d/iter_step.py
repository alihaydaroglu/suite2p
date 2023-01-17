

from cmath import log
import os
from turtle import ht
import numpy as n
import copy
from multiprocessing import shared_memory, Pool

from ..io import lbm as lbmio 
from ..registration import register
from . import utils
from . import deepinterp as dp

from ..detection import utils as det_utils
from ..detection import detection3d as det3d

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
     

def fuse_and_save_reg_file(reg_file, reg_fused_dir, centers, shift_xs, nshift, nbuf, crops=None, mov=None, save=True):
    file_name = reg_file.split('\\')[-1]
    fused_file_name = os.path.join(reg_fused_dir, 'fused_' + file_name)
    if mov is None: 
        mov = n.load(reg_file)
    nz, nt, ny, nx = mov.shape
    weights = n.linspace(0, 1, nshift)
    n_seams = len(centers)
    nxnew = nx - (nshift + nbuf) * n_seams
    mov_fused = n.zeros((nz, nt, ny, nxnew), dtype=mov.dtype)

    for zidx in range(nz):
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
    if save: 
        n.save(fused_file_name, mov_fused)
        return fused_file_name
    else: return mov_fused


def init_batch_files(job_iter_dir, job_reg_data_dir, n_batches):
    reg_data_paths = []
    batch_dirs = []
    for batch_idx in range(n_batches):
        reg_data_filename = 'reg_data%04d.npy' % batch_idx
        reg_data_path = os.path.join(job_reg_data_dir, reg_data_filename)
        reg_data_paths.append(reg_data_path)

        batch_dir = os.path.join(job_iter_dir, 'batch%04d' % batch_idx)
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
    n_tifs_to_analyze = params['total_tifs_to_analyze']
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
    example_arr = ref_img_3d
    shmem_sum_img, shmem_sum_img_params, sum_img = utils.create_shmem_from_arr(example_arr)
    shmem_mean_img, shmem_mean_img_params, mean_img = utils.create_shmem_from_arr(example_arr.astype(mov_dtype))
    shmem_max_img,shmem_max_img_params, max_img = utils.create_shmem_from_arr(example_arr)
    n_frames_proc = 0
    
    batch_dirs, reg_data_paths = init_batch_files(job_iter_dir, job_reg_data_dir, n_batches)

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
            iter_dir = batch_dirs[batch_idx]
            reg_data_path = reg_data_paths[batch_idx]
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
            all_offsets = register_mov(mov,refs_and_masks, all_ops, log_cb)
            
            log_cb("After reg:", level=3,log_mem_usage=True )

            n.save(os.path.join(iter_dir, 'all_offsets.npy'), all_offsets)

            nz, nt, ny, nx = mov.shape
            n_frames_proc_new = n_frames_proc + nt
            
            sum_img_batch = mov.sum(axis=1)
            mean_img_batch = sum_img_batch / nt
            max_img_batch = mov.max(axis=1)
            max_img = n.max([max_img, max_img_batch], axis=0)
            sum_img += sum_img_batch
            mean_img = mean_img * (n_frames_proc / n_frames_proc_new) + mean_img_batch * (nt / n_frames_proc_new)
            n_frames_proc = n_frames_proc_new

            log_cb("Iteration complete, saving mean/max images to %s" % iter_dir,1)
            n.save(os.path.join(iter_dir, 'max_img.npy'), max_img)
            n.save(os.path.join(iter_dir, 'mean_img.npy'), mean_img)
            n.save(os.path.join(iter_dir, 'sum_img.npy'), sum_img)
            del sum_img_batch
            del mean_img_batch
            del max_img_batch
            n_cleared = gc.collect()
            log_cb("Garbage collected %d items" %n_cleared, 2)
            log_cb("After gc collect: ", level=3,log_mem_usage=True )
        except Exception as exc:
            log_cb("Error occured in iteration %d" % batch_idx, 0 )
            tb = traceback.format_exc()
            log_cb(tb, 0)
            break


def register_dataset_old(tifs, params, dirs, summary, log_cb = default_log,
                    override_input_reg_bins=None, save_output=True, n_proc_detection=10,
                    mem_profile=False, mem_profile_save_dir=None, debug_on_ones=False, do_detection=False, start_batch_idx = 0):
    if not save_output:
        log_cb("Not saving outputs to file",0)
    if mem_profile:
        log_cb("Profiling memory",0)

    ref_img_3d = summary['ref_img_3d']
    crosstalk_coeff = summary['crosstalk_coeff']
    refs_and_masks = summary.get('refs_and_masks', None)
    all_ops = summary.get('all_ops',None)
    plane_means = summary.get('plane_mean', None)
    plane_stds = summary.get('plane_std', None)

    job_iter_dir = dirs['iters']
    job_reg_data_dir = dirs['registered_data']
    job_deepinterp_dir = dirs.get('deepinterp', None)

    n_tifs_to_analyze = params['total_tifs_to_analyze']
    tif_batch_size = params['tif_batch_size']
    planes = params['planes']
    notch_filt = params['notch_filt']
    load_from_binary = params['load_from_binary']
    do_subtract_crosstalk = params['subtract_crosstalk']
    mov_dtype = params['dtype']
    temporal_high_pass_width = params['temporal_high_pass_width']
    do_running_sdmov = params['running_sdmov']
    npil_hpf_xy = params['npil_high_pass_xy']
    npil_hpf_z = params['npil_high_pass_z']
    unif_filter_xy = params['detection_unif_filter_xy']
    unif_filter_z = params['detection_unif_filter_z']
    intensity_thresh = params['intensity_threshold']
    fuse = params.get('fuse_after_registration',False)

    do_deepinterp = params.get('do_deepinterp', False)
    model_path = params.get('model_path', None)
    n_files_before = params.get('n_files_before', 30)
    n_files_after = params.get('n_files_after', 30)
    crop_size = params.get('crop_size', (1024,1024))
    batch_size = params.get('batch_size', 4)
    do_norm_per_batch = params.get('do_norm_per_batch', False)
    centroid = params.get("centroid", None)

    if mem_profile:
        mallocs = []
        malloc_labels = []
        tracemalloc.start()
        mallocs.append(tracemalloc.take_snapshot())
        malloc_labels.append('before initialization')

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    log_cb("Will analyze %d tifs in %d batches" % (len(n.concatenate(batches)), len(batches)),0)
    if load_from_binary:
        log_cb("Loading from saved binaries, will skip registration and crosstalk subtraction", 0)
    
    # init accumulators
    nz, ny, nx = ref_img_3d.shape
    if crop_size is not None and do_deepinterp:
        ny, nx = crop_size
        if do_deepinterp:
            example_arr = n.zeros((nz,ny,nx), mov_dtype)
        else: 
            example_arr = n.zeros((nz,ny,nx), ref_img_3d.dtype)
    else: example_arr = ref_img_3d
    shmem_sum_img, shmem_sum_img_params, sum_img = utils.create_shmem_from_arr(example_arr)
    shmem_mean_img, shmem_mean_img_params, mean_img = utils.create_shmem_from_arr(example_arr.astype(mov_dtype))
    shmem_max_img,shmem_max_img_params, max_img = utils.create_shmem_from_arr(example_arr)
    n_frames_proc = 0
    
    batch_dirs, reg_data_paths = init_batch_files(job_iter_dir, job_reg_data_dir, n_batches)
    if override_input_reg_bins is not None:
        reg_data_paths = override_input_reg_bins
        n_batches = len(reg_data_paths)
        log_cb("Reading files from user-provided location, only %d batches" % n_batches, 0)
    if mem_profile:
        mallocs.append(tracemalloc.take_snapshot())
        malloc_labels.append("Before first batch")

    if fuse:
        __, fuse_xs = lbmio.load_and_stitch_full_tif_mp(tifs[0], channels=n.arange(1), get_roi_start_pix=True)
        fuse_centers = n.sort(fuse_xs)[1:]
        fuse_shift_xs = summary['plane_shifts'][:,1].astype(int)
        fuse_nshift = params['fuse_nshift']
        fuse_nbuf = params['fuse_nbuf']
    loaded_movs = [0]

    if not load_from_binary:
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
            if mem_profile:
                cur, peak = tracemalloc.get_traced_memory()
                # log_cb("Start of Batch %03d. Current Memory: %09.6f GB, Peak: %09.6f GB" % (batch_idx, cur/(1024**3), peak/(1024**3)), 0)
                vm = psutil.virtual_memory()
            log_cb("Start Batch: ", level=3,log_mem_usage=True )
            iter_dir = batch_dirs[batch_idx]
            reg_data_path = reg_data_paths[batch_idx]
            log_cb("Loading Batch %d of %d" % (batch_idx+1, n_batches), 0)
            if not load_from_binary:
                io_thread.join()
                log_cb("Batch %d IO thread joined" % (batch_idx))
                if mem_profile:
                    cur, peak = tracemalloc.get_traced_memory()
                    # log_cb("After IO thread join on Batch %03d. Current Memory: %09.6f GB, Peak: %09.6f GB" % (batch_idx, cur/(1024**3), peak/(1024**3)), 0)
                log_cb('After IO thread join', level=3,log_mem_usage=True )
                shmem_mov,shmem_mov_params, mov = utils.create_shmem_from_arr(loaded_movs[0], copy=True)
                
                log_cb("After Sharr creation:", level=3,log_mem_usage=True )
                if batch_idx + 1 < n_batches:
                    io_thread = threading.Thread(target=io_thread_loader, args=(batches[batch_idx+1], batch_idx+1))
                    io_thread.start()
                    log_cb("After IO thread launch:", level=3,log_mem_usage=True )
                if do_subtract_crosstalk:
                    __ = subtract_crosstalk(shmem_mov_params, crosstalk_coeff, planes = planes, log_cb = log_cb)
                log_cb("Registering Batch %d" % batch_idx, 1)
                
                log_cb("Before Reg:", level=3,log_mem_usage=True )
                all_offsets = register_mov(mov,refs_and_masks, all_ops, log_cb)
                
                
                log_cb("After reg:", level=3,log_mem_usage=True )
                
                if fuse:
                    fuse_and_save_reg_file(reg_data_path,'\\'.join(reg_data_path.split('\\')[:-1]), mov=mov,
                    centers=fuse_centers, shift_xs=fuse_shift_xs, nshift=fuse_nshift, nbuf=fuse_nbuf)
                else:
                    n.save(reg_data_path, mov)

                n.save(os.path.join(iter_dir, 'all_offsets.npy'), all_offsets)
                del all_offsets
                
                log_cb("After deleting offsets:", level=3,log_mem_usage=True )
            else:
                log_cb("Reading registered binaries from %s" % reg_data_path)
                log_cb("Before Load: ", level=3,log_mem_usage=True )
                mov = n.load(reg_data_path)
                
                log_cb("After load:", level=3,log_mem_usage=True ) 
                shmem_mov_params = {}
                shmem_mov = None

            nz, nt, ny, nx = mov.shape
            n_frames_proc_new = n_frames_proc + nt
            if do_deepinterp:
                model = dp.init_deepinterp_model(model_path)
                n_buffer = n_files_before + n_files_after
                buffer_full = n.zeros((nz, n_buffer, crop_size[0], crop_size[1]), mov.dtype)
                if centroid is None:
                    centroid = utils.get_centroid(ref_img_3d)
                if crop_size is not None:
                    log_cb("Cropping around " + str(centroid) )
                    mov = utils.pad_crop_movie(mov, centroid, crop_size)
                    
                mov_out = dp.denoise_mov3d(model, mov, buffer_full, plane_means=plane_means,
                                        plane_stds=plane_stds, do_norm_per_batch=do_norm_per_batch, log=log_cb)
                log_cb("DP output shape: " + str(mov_out.shape), 2)
                if batch_idx == 0:
                    mov_out = mov_out[:, n_files_before:]
                if batch_idx == n_batches-1:
                    mov_out = mov_out[:, :-n_files_before]
                log_cb("Saved output shape: " + str(mov_out.shape), 2)
                mov_out = mov_out.astype(mov_dtype)
                denoised_file = os.path.join(job_deepinterp_dir, 'dp_batch%04d_offset%02d.npy' % (batch_idx, n_files_before))
                log_cb("Saving denoised movie to %s" % denoised_file, 1)
                n.save(denoised_file, mov_out)
                log_cb("Denoising done")
                mov = mov_out
                nz, nt, ny, nx = mov.shape
            
            sum_img_batch = mov.sum(axis=1)
            mean_img_batch = sum_img_batch / nt
            max_img_batch = mov.max(axis=1)

            max_img = n.max([max_img, max_img_batch], axis=0)
            log_cb("Before Swap:", level=3,log_mem_usage=True )

            if debug_on_ones:
                mov3d = n.ones((nt, nz, ny, nx), mov_dtype)
            else:
                temp3d = mov.swapaxes(0,1)
                mov3d = temp3d.astype(mov_dtype)
                del temp3d
            log_cb("After Swap:", level=3,log_mem_usage=True )
            log_cb("Freeing shared memory")
            if shmem_mov is not None:
                shmem_mov.close(); shmem_mov.unlink()
            log_cb("After freeing:", level=3,log_mem_usage=True )
            del mov; del shmem_mov

            log_cb("Running detection", 1)
            log_cb("Before detection: ", level=3,log_mem_usage=True )
            if do_detection:
                mov3d = run_detection(mov3d, vmap2, sdmov2, n_frames_proc, temporal_high_pass_width, do_running_sdmov,
                            npil_hpf_xy, npil_hpf_z, unif_filter_xy, unif_filter_z, intensity_thresh,
                            log_cb = log_cb, n_proc=n_proc_detection)
            log_cb("After detection: ", level=3,log_mem_usage=True )
            
            log_cb("Update accumulators",2)
            sum_img += sum_img_batch
            mean_img = mean_img * (n_frames_proc / n_frames_proc_new) + mean_img_batch * (nt / n_frames_proc_new)

            if batch_idx > 0:
                del vmap
            vmap = vmap2 ** .5
            n_frames_proc = n_frames_proc_new

            log_cb("Iteration complete",1)
            if save_output and do_detection:
                log_cb("Saving outputs to %s" % iter_dir, 1)
                n.save(os.path.join(iter_dir, 'vmap.npy'), vmap)
                # n.save(os.path.join(iter_dir, 'mov_sub.npy'), mov3d)
                n.save(os.path.join(iter_dir, 'max_img.npy'), max_img)
                n.save(os.path.join(iter_dir, 'mean_img.npy'), mean_img)
                n.save(os.path.join(iter_dir, 'sum_img.npy'), sum_img)
            del mov3d
            del sum_img_batch
            del mean_img_batch
            del max_img_batch
            if mem_profile:
                mallocs.append(tracemalloc.take_snapshot())
                malloc_labels.append("End of batch %d" % batch_idx)
                n.save(os.path.join(mem_profile_save_dir, 'mem_profiles_end_of_batch%05d.npy'  % batch_idx), (mallocs, malloc_labels))
                mallocs = []; malloc_labels = [];
            log_cb("After saving: ", level=3,log_mem_usage=True )
            n_cleared = gc.collect()
            log_cb("Garbage collected %d items" %n_cleared, 2)
            log_cb("After gc collect: ", level=3,log_mem_usage=True )

        except Exception as exc:
            log_cb("Error occured in iteration %d" % batch_idx, 0 )
            tb = traceback.format_exc()
            log_cb(tb, 0)
            break


    output = {
        'vmap' : vmap,
        'max_img' :max_img,
        'mean_img' : mean_img,
        'sum_img' : sum_img
    }

    if mem_profile:
        output['mallocs'] = mallocs
        output['malloc_labels'] = malloc_labels
        n.save(os.path.join(mem_profile_save_dir, 'mem_profile_out.npy'), output)


    # return output


# def iter_dataset_slow(tifs, params, dirs, summary, log_cb=default_log,
#                       override_input_reg_bins=None, save_output=True,
#                       mem_profile=False, mem_profile_save_dir=None):
#     if not save_output:
#         log_cb("Not saving outputs to file", 0)
#     if mem_profile:
#         log_cb("Profiling memory", 0)

#     ref_img_3d = summary['ref_img_3d']
#     crosstalk_coeff = summary['crosstalk_coeff']
#     refs_and_masks = summary.get('refs_and_masks', None)
#     all_ops = summary.get('all_ops', None)
#     plane_means = summary.get('plane_mean', None)
#     plane_stds = summary.get('plane_std', None)

#     job_iter_dir = dirs['iters']
#     job_reg_data_dir = dirs['registered_data']
#     job_deepinterp_dir = dirs['deepinterp']

#     n_tifs_to_analyze = params['total_tifs_to_analyze']
#     tif_batch_size = params['tif_batch_size']
#     planes = params['planes']
#     notch_filt = params['notch_filt']
#     load_from_binary = params['load_from_binary']
#     do_subtract_crosstalk = params['subtract_crosstalk']
#     mov_dtype = params['dtype']
#     temporal_high_pass_width = params['temporal_high_pass_width']
#     do_running_sdmov = params['running_sdmov']
#     npil_hpf_xy = params['npil_high_pass_xy']
#     npil_hpf_z = params['npil_high_pass_z']
#     unif_filter_xy = params['detection_unif_filter_xy']
#     unif_filter_z = params['detection_unif_filter_z']
#     intensity_thresh = params['intensity_threshold']

#     do_deepinterp = params.get('do_deepinterp', False)
#     model_path = params.get('model_path', None)
#     n_files_before = params.get('n_files_before', 30)
#     n_files_after = params.get('n_files_after', 30)
#     crop_size = params.get('crop_size', (1024, 1024))
#     batch_size = params.get('batch_size', 4)
#     do_norm_per_batch = params.get('do_norm_per_batch', False)
#     centroid = params.get("centroid", None)

#     if mem_profile:
#         mallocs = []
#         malloc_labels = []
#         tracemalloc.start()
#         mallocs.append(tracemalloc.take_snapshot())
#         malloc_labels.append('before initialization')

#     batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
#     n_batches = len(batches)
#     log_cb("Will analyze %d tifs in %d batches" %
#            (len(n.concatenate(batches)), len(batches)), 0)
#     if load_from_binary:
#         log_cb(
#             "Loading from saved binaries, will skip registration and crosstalk subtraction", 0)

#     # init accumulators
#     nz, ny, nx = ref_img_3d.shape
#     if crop_size is not None and do_deepinterp:
#         ny, nx = crop_size
#         if do_deepinterp:
#             example_arr = n.zeros((nz, ny, nx), mov_dtype)
#         else:
#             example_arr = n.zeros((nz, ny, nx), ref_img_3d.dtype)
#     else:
#         example_arr = ref_img_3d
#     shmem_sum_img, shmem_sum_img_params, sum_img = utils.create_shmem_from_arr(
#         example_arr)
#     shmem_mean_img, shmem_mean_img_params, mean_img = utils.create_shmem_from_arr(
#         example_arr.astype(mov_dtype))
#     shmem_max_img, shmem_max_img_params, max_img = utils.create_shmem_from_arr(
#         example_arr)
#     shmem_vmap2, shmem_vmap2_params, vmap2 = utils.create_shmem_from_arr(
#         example_arr.astype(mov_dtype))
#     shmem_sdmov2, shmem_sdmov2_params, sdmov2 = utils.create_shmem_from_arr(
#         example_arr.astype(mov_dtype))
#     n_frames_proc = 0

#     batch_dirs, reg_data_paths = init_batch_files(
#         job_iter_dir, job_reg_data_dir, n_batches)
#     if override_input_reg_bins is not None:
#         reg_data_paths = override_input_reg_bins
#         n_batches = len(reg_data_paths)
#         log_cb("Reading files from user-provided location, only %d batches" %
#                n_batches, 0)
#     if mem_profile:
#         mallocs.append(tracemalloc.take_snapshot())
#         malloc_labels.append("Before first batch")

#     for batch_idx in range(n_batches):
#         try:
#             iter_dir = batch_dirs[batch_idx]
#             reg_data_path = reg_data_paths[batch_idx]
#             # log_cb("Loading Batch %d of %d" % (batch_idx, n_batches), 0)
#             if not load_from_binary:

#                 tifs = batches[batch_idx]
#                 mov = lbmio.load_and_stitch_tifs(
#                     tifs, planes, filt=notch_filt, concat=True, log_cb=log_cb)
#                 shmem_mov, shmem_mov_params, mov = utils.create_shmem_from_arr(
#                     mov, copy=True)
#                 if do_subtract_crosstalk:
#                     __ = subtract_crosstalk(
#                         shmem_mov_params, crosstalk_coeff, planes=planes, log_cb=log_cb)
#                 log_cb("Registering Batch %d" % batch_idx, 1)
#                 all_offsets = register_mov(
#                     mov, refs_and_masks, all_ops, log_cb)
#                 if save_output:
#                     log_cb("Saving registered file to %s" % reg_data_path, 2)
#                     n.save(reg_data_path, mov)
#                     n.save(os.path.join(iter_dir, 'all_offsets.npy'), all_offsets)
#             else:
#                 log_cb("Reading registered binaries from %s" % reg_data_path)
#                 mov = n.load(reg_data_path)
#                 shmem_mov_params = {}

#             nz, nt, ny, nx = mov.shape
#             n_frames_proc_new = n_frames_proc + nt
#             if do_deepinterp:
#                 model = dp.init_deepinterp_model(model_path)
#                 n_buffer = n_files_before + n_files_after
#                 buffer_full = n.zeros(
#                     (nz, n_buffer, crop_size[0], crop_size[1]), mov.dtype)
#                 if centroid is None:
#                     centroid = utils.get_centroid(ref_img_3d)
#                 if crop_size is not None:
#                     log_cb("Cropping around " + str(centroid))
#                     mov = utils.pad_crop_movie(mov, centroid, crop_size)

#                 mov_out = dp.denoise_mov3d(model, mov, buffer_full, plane_means=plane_means,
#                                            plane_stds=plane_stds, do_norm_per_batch=do_norm_per_batch, log=log_cb)
#                 log_cb("DP output shape: " + str(mov_out.shape), 2)
#                 if batch_idx == 0:
#                     mov_out = mov_out[:, n_files_before:]
#                 if batch_idx == n_batches-1:
#                     mov_out = mov_out[:, :-n_files_before]
#                 log_cb("Saved output shape: " + str(mov_out.shape), 2)
#                 mov_out = mov_out.astype(mov_dtype)
#                 denoised_file = os.path.join(
#                     job_deepinterp_dir, 'dp_batch%04d_offset%02d.npy' % (batch_idx, n_files_before))
#                 log_cb("Saving denoised movie to %s" % denoised_file, 1)
#                 n.save(denoised_file, mov_out)
#                 log_cb("Denoising done")
#                 mov = mov_out
#                 nz, nt, ny, nx = mov.shape

#             sum_img_batch = mov.sum(axis=1)
#             mean_img_batch = sum_img_batch / nt
#             max_img_batch = mov.max(axis=1)

#             max_img = n.max([max_img, max_img_batch], axis=0)
#             mov3d = mov.swapaxes(0, 1).astype(mov_dtype)
#             log_cb("Freeing shared memory")
#             utils.close_and_unlink_shmem(shmem_mov_params)

#             log_cb("Running detection", 1)
#             run_detection(mov3d, vmap2, sdmov2, n_frames_proc, temporal_high_pass_width, do_running_sdmov,
#                           npil_hpf_xy, npil_hpf_z, unif_filter_xy, unif_filter_z, intensity_thresh,
#                           log_cb=log_cb)

#             log_cb("Update accumulators", 2)
#             sum_img += sum_img_batch
#             mean_img = mean_img_batch * \
#                 (n_frames_proc / n_frames_proc_new) + \
#                 mean_img_batch * (nt / n_frames_proc_new)

#             vmap = vmap2 ** .5
#             n_frames_proc = n_frames_proc_new

#             log_cb("Iteration complete", 1)
#             if save_output:
#                 log_cb("Saving outputs to %s" % iter_dir, 1)
#                 n.save(os.path.join(iter_dir, 'vmap.npy'), vmap)
#                 n.save(os.path.join(iter_dir, 'max_img.npy'), max_img)
#                 n.save(os.path.join(iter_dir, 'mean_img.npy'), mean_img)
#                 n.save(os.path.join(iter_dir, 'sum_img.npy'), sum_img)

#             if mem_profile:
#                 mallocs.append(tracemalloc.take_snapshot())
#                 malloc_labels.append("End of batch %d" % batch_idx)
#         except Exception as exc:
#             log_cb("Error occured in iteration %d" % batch_idx, 0)
#             tb = traceback.format_exc()
#             log_cb(tb, 0)
#             break

#     output = {
#         'vmap': vmap,
#         'max_img': max_img,
#         'mean_img': mean_img,
#         'sum_img': sum_img
#     }

#     if mem_profile:
#         output['mallocs'] = mallocs
#         output['malloc_labels'] = malloc_labels
#         n.save(os.path.join(mem_profile_save_dir, 'mem_profile_out.npy'), output)

#     return output
