/*
Copyright 2017-2018 Peter Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use ::proc::*;

//use cray_shmem::*;
#[cfg(feature = "gpu")] use gpudevicemem::*;
use mpich::*;

use std::sync::{ONCE_INIT, Once};

static DIST_PROC_GROUP_ONCE: Once = ONCE_INIT;

pub struct DistProcGroup {
  closed:   bool,
}

impl Drop for DistProcGroup {
  fn drop(&mut self) {
    // TODO: should make sure that the proc group outlives procs.
    //assert!(Shmem::finalize().is_ok());
    assert!(mpi_finalize().is_ok());
  }
}

impl Default for DistProcGroup {
  fn default() -> Self {
    DIST_PROC_GROUP_ONCE.call_once(|| {
      #[cfg(feature = "gpu")]
      {
        // TODO: do this to initialize CUDA before MPI.
        println!("DEBUG: DistProcGroup: num gpu devices: {}", GPUDeviceId::count());
      }
      assert!(mpi_init_multithreaded().is_ok());
      //assert!(Shmem::init_multithreaded().is_ok());
    });
    DistProcGroup{
      closed:   false,
    }
  }
}

impl DistProcGroup {
}

impl Iterator for DistProcGroup {
  type Item = DistProcSpawner;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    // TODO: fork/join-over-SPMD impl.
    self.closed = true;
    let srank = MPIComm::world().rank() as usize;
    let nsranks = MPIComm::world().num_ranks() as usize;
    //let shmem_rank = Shmem::rank() as usize;
    //let shmem_nranks = Shmem::num_ranks() as usize;
    //assert_eq!(rank, shmem_rank);
    //assert_eq!(nranks, shmem_nranks);
    let proc = DistProc{
      srank, nsranks,
    };
    Some(DistProcSpawner{proc})
  }
}

pub struct DistProcSpawner {
  proc: DistProc,
}

impl DistProcSpawner {
  pub fn spawn<F>(self, f: F) -> Result<DistProcJoinHandle, ()> where F: FnOnce(DistProc) + Send + 'static {
    // TODO
    f(self.proc);
    Ok(DistProcJoinHandle{})
  }
}

pub struct DistProcJoinHandle {
}

impl Drop for DistProcJoinHandle {
  fn drop(&mut self) {
    // TODO
  }
}

impl DistProcJoinHandle {
  pub fn join(self) {
  }
}

#[derive(Clone)]
pub struct DistProc {
  srank:    usize,
  nsranks:  usize,
}

impl FlatProc for DistProc {
  fn flat_rank(&self) -> usize {
    self.srank
  }

  fn num_ranks(&self) -> usize {
    self.nsranks
  }
}

impl Proc<usize> for DistProc {
  fn rank(&self) -> usize {
    self.srank
  }

  fn sup_rank(&self) -> usize {
    self.nsranks
  }

  fn wait_barrier(&self) -> bool {
    mpi_barrier(&mut MPIComm::world()).unwrap();
    self.srank == 0
  }
}

impl<T> ProcSyncIO<T> for DistProc where T: MPIDataTypeExt + Copy {
  fn sync_allreduce_sum_inplace(&self, buf: &mut [T]) {
    let ptr = buf.as_mut_ptr();
    let len = buf.len();
    unsafe { mpi_allreduce(
        ptr, len,
        ptr, len,
        MPIReduceOp::Sum,
        &mut MPIComm::world(),
    ) }.unwrap();
  }
}
