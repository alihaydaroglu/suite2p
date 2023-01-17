import tifffile
from cmath import log
import datetime
import os
import numpy as n
from . import init_pass
from . import utils as u3d 
import psutil
from suite2p.io import lbm as lbmio
from multiprocessing import Pool
from suite2p.suite3d.iter_step import fuse_and_save_reg_file

class Job:
    def __init__(self, root_dir, job_id, params=None, tifs=None, exist_ok=False, verbosity=10, create=True):
        """Create a Job object that is a wrapper to manage files, current state, log etc.

        Args:
            root_dir (str): Root directory in which job directory will be created
            job_id (str): Unique name for the job directory
            params (dict): Job parameters (see examples)
            tifs (list) : list of full paths to tif files to be used
            exist_ok (bool, optional): If False, will throw error if job_dir exists. Defaults to False.
            verbosity (int, optional): Verbosity level. 0: critical only, 1: info, 2: debug. Defaults to 1.
        """

        self.verbosity = verbosity
        if create:
            self.params = params
            self.params['tifs'] = tifs
            self.tifs = tifs
            self.init_job_dir(root_dir, job_id, exist_ok=exist_ok)
        else:
            self.init_job_dir(root_dir, job_id, exist_ok=True, update_params=False)
            self.load_params()
            self.tifs = self.params.get('tifs', [])
            
        self.save_params()
               
    def fuse_registered_movie(self, n_shift, n_buf, files=None, save=False, n_proc=4):
        '''
        Fuse the strips of the stitched tiff together, interpolating at the borders

        Args:
            n_shift (int): number of pixels of overlap (how many pixels on the right border of strip 1 are repeated on the left border of strip 2, ignoring the black pixels between the strips)
            n_buf (int): number of black pixels between strips
            files (list, optional): List of file paths for registered binaries. Defaults to None, which means all   registered binaries in the job.
            save (bool, optional): Whether to save results in registered_fused_data in the jobdir. Defaults to False.
            n_proc (int, optional): Number of processors. Each processor loads a whole file, so be mindful of ram. Defaults to 4.

        Returns:
            fused_files: A list of filepaths is save=True, a concatenated, fused movie if save=False
        '''
        if files is None:
            files = self.get_registered_files()
        __, xs = lbmio.load_and_stitch_full_tif_mp(self.tifs[0], channels=n.arange(1), get_roi_start_pix=True)
        centers = n.sort(xs)[1:]
        shift_xs = n.round(self.load_summary()['plane_shifts'][:,1]).astype(int)
        if save:
            reg_fused_dir = self.make_new_dir('registered_fused_data')
        else: reg_fused_dir = ''
        with Pool(n_proc) as p:
            fused_files = p.starmap(fuse_and_save_reg_file, [(file, reg_fused_dir, centers,  shift_xs, n_shift, n_buf, None, None, save) for file in files])
        if not save:
            fused_files = n.concatenate(fused_files, axis=1)
        return fused_files

    def log(self, string='', level=1, logfile=True, log_mem_usage=False):
        """Print messages based on current verbosity level

        Args:
            string (str): String to be printed
            level (int, optional): Level equal or below self.verbosity will be printed. Defaults to 1.
        """
        if log_mem_usage:
            
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            total = sm.used + vm.used
            string = "{:<20}".format(string)
            string += ("Total Used: %07.3f GB, Virtual: %07.3f GB, Swap: %07.3f" %
                       ((total/(1024**3), vm.used/(1024**3), sm.used/(1024**3))))
            
        if level <= self.verbosity:
            # print('xxx')
            print(("   " * level) + string)
        if logfile:
            logfile = os.path.join(self.job_dir, 'log.txt')
            self.logfile = logfile
            with open(logfile, 'a+') as f:
                datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = '\n[%s][%02d] ' % (datetime_string, level)
                f.write(header + '   ' * level + string)

    def save_params(self, new_params=None):
        """Update saved params in job_dir/params.npy
        """
        params_path = os.path.join(self.dirs['job_dir'], 'params.npy')
        if new_params is not None:
            self.params.update(new_params)
        n.save(params_path, self.params)
        self.log("Updated params file: %s" % params_path, 2)

    def load_params(self):
        params_path = os.path.join(self.dirs['job_dir'], 'params.npy')
        self.params = n.load(params_path, allow_pickle=True).item()
        self.log("Found and loaded params from %s" % params_path)

    def load_summary(self):
        summary_path = os.path.join(self.dirs['summary'], 'summary.npy')
        summary = n.load(summary_path,  allow_pickle=True).item()
        return summary

    def make_svd_dirs(self, n_blocks=None):
        self.make_new_dir('svd')
        self.make_new_dir('blocks', 'svd', dir_tag='svd_blocks')
        block_dirs = []
        if n_blocks is not None:
            for i in range(n_blocks):
                block_dirs.append(self.make_new_dir('%03d' % i, 'svd_blocks', dir_tag = 'svd_blocks_%03d' % i))
            return block_dirs

    def make_stack_dirs(self, n_stacks):
        stack_dirs = []
        self.make_new_dir('stacks', 'svd', dir_tag='svd_stacks')
        for i in range(n_stacks):
            stack_dirs.append(self.make_new_dir('%03d' % i, 'svd_stacks', dir_tag = 'svd_stacks_%03d' % i))
        return stack_dirs

    def make_new_dir(self, dir_name, parent_dir_name = None, exist_ok=True, dir_tag = None):
        if parent_dir_name is None:
            parent_dir = self.job_dir
        else: 
            parent_dir = self.dirs[parent_dir_name]
        if dir_tag is None:
            dir_tag = dir_name
        
        dir_path = os.path.join(parent_dir, dir_name)
        os.makedirs(dir_path, exist_ok = exist_ok)
        self.dirs[dir_tag] = dir_path
        n.save(os.path.join(self.job_dir, 'dirs.npy'), self.dirs)
        return dir_path

    def init_job_dir(self, root_dir, job_id, exist_ok=False, update_params=True):
        """Create a job directory and nested dirs

        Args:
            root_dir (str): Root directory to create job_dir in
            job_id (str): Unique name for job
            exist_ok (bool, optional): If False, throws error if job_dir exists. Defaults to False.
        """

        job_dir = os.path.join(root_dir,'s3d-%s' % job_id)
        self.job_dir = job_dir
        if os.path.isdir(job_dir):
            self.log("Job directory %s already exists" % job_dir, 0)
            assert exist_ok, "Manually delete job_dir, or set exist_ok=True"
        else:
            os.makedirs(job_dir, exist_ok=True)

        self.log("Creating job directory for %s in %s" %
                    (job_id, root_dir), 0)
        if 'dirs.npy' in os.listdir(job_dir):
            self.log("Loading dirs ")
            self.dirs = n.load(os.path.join(job_dir, 'dirs.npy'),allow_pickle=True).item()
        else:
            self.dirs = {'job_dir' : job_dir}

        if job_dir not in self.dirs.keys():
            self.dirs['job_dir'] = job_dir

        for dir_name in ['registered_data', 'summary', 'iters']:
            dir_key = dir_name
            if dir_key not in self.dirs.keys():
                new_dir = os.path.join(job_dir, dir_name) 
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir, exist_ok=True)
                    self.log("Created dir %s" % new_dir,2)
                else:
                    self.log("Found dir %s" % new_dir,2)
                self.dirs[dir_key] = new_dir
                
            else:
                self.log("Found dir %s" % dir_name,2)
        n.save(os.path.join(job_dir, 'dirs.npy'), self.dirs)

    def run_init_pass(self):
        self.log("Launching initial pass", 0)
        init_pass.run_init_pass(self)
    def copy_init_pass(self,summary_old_job):
        n.save(os.path.join(self.dirs['summary'],
               'summary.npy'), summary_old_job)
    
    def get_registered_files(self, key='registered_data', filename_filter='reg_data'):
        all_files = n.os.listdir(self.dirs[key])
        reg_files = [os.path.join(self.dirs[key],x) for x in all_files if x.startswith(filename_filter)]
        return reg_files
    def get_denoised_files(self):
        all_files = n.os.listdir(self.dirs['deepinterp'])
        reg_files = [os.path.join(self.dirs['deepinterp'],x) for x in all_files if x.startswith('dp')]
        return reg_files

    def get_iter_dirs(self):
        iters_dir = self.dirs['iters']
        iter_dirs = [os.path.join(iters_dir, dir) for dir in os.listdir(iters_dir)]
        ret = []
        for dir in iter_dirs:
            if 'vmap.npy' in os.listdir(dir):
                ret.append(dir)
        return ret
    
    def load_iter_results(self, iter_idx):
        iter_dir = self.get_iter_dirs()[iter_idx]
        self.log("Loading from %s" % iter_dir)
        res = {
            'vmap' : n.load(os.path.join(iter_dir, 'vmap.npy')),
            'max_img' : n.load(os.path.join(iter_dir, 'max_img.npy')),
            'mean_img' : n.load(os.path.join(iter_dir, 'mean_img.npy')),
            'sum_img' : n.load(os.path.join(iter_dir, 'sum_img.npy')),
        }
        return res


    def get_subtracted_movie(self):
        mov_sub_paths = []
        for d in self.get_iter_dirs():
            if 'mov_sub.npy' in os.listdir(d):
                mov_sub_paths.append(os.path.join(d, 'mov_sub.npy'))

        mov_sub = u3d.npy_to_dask(mov_sub_paths, axis=0)
        return mov_sub

    def get_registered_movie(self, key='registered_data', filename_filter='reg_data'):
            paths = self.get_registered_files(key, filename_filter)
            mov_reg = u3d.npy_to_dask(paths, axis=1)
            return mov_reg

    def load_frame_counts(self):
        return n.load(os.path.join(self.dirs['job_dir'],'frames.npy'), allow_pickle=True).item()


    def save_frame_counts(self):
        size_to_frames = {}
        nframes = []
        jobids = []
        for tif in self.tifs:
            jobids.append(int(tif.split('\\')[-2]))
            tifsize = int(os.path.getsize(tif))
            if tifsize in size_to_frames.keys():
                nframes.append(size_to_frames[tifsize])
            else:
                tf = tifffile.TiffFile(tif)
                nf = len(tf.pages) // 30
                nframes.append(nf)
                size_to_frames[tifsize] = nf
                self.log(tif +  ' is %d frames and %d bytes' % (nf, tifsize))

        nframes = n.array(nframes)
        jobids = n.array(jobids)

        tosave = {'nframes' : nframes, 'jobids' : jobids}
        self.frames = tosave
        n.save(os.path.join(self.dirs['job_dir'],'frames.npy'), tosave)

        return nframes, jobids