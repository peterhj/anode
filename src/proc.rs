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

use mpich::*;

use std::sync::{ONCE_INIT, Once};

static DIST_PROC_GROUP_ONCE: Once = ONCE_INIT;

pub struct DistProcGroup {
  closed:   bool,
}

impl Drop for DistProcGroup {
  fn drop(&mut self) {
    // TODO: should make sure that the proc group outlives procs.
    assert!(mpi_finalize().is_ok());
  }
}

impl Default for DistProcGroup {
  fn default() -> Self {
    DIST_PROC_GROUP_ONCE.call_once(|| {
      // TODO: use this to init MPI.
      //MPI::init()
      assert!(mpi_init_multithreaded().is_ok());
    });
    DistProcGroup{
      closed:   false,
    }
  }
}

impl Iterator for DistProcGroup {
  type Item = DistProc;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    // TODO: fork/join-over-SPMD impl.
    self.closed = true;
    let rank = MPIComm::world().rank() as usize;
    let nranks = MPIComm::world().num_ranks() as usize;
    Some(DistProc{
      rank,
      nranks,
    })
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
  rank:     usize,
  nranks:   usize,
}

impl DistProc {
  pub fn spawn<F>(self, f: F) -> Result<DistProcJoinHandle, ()> where F: FnOnce(DistProc) + Send + 'static {
    // TODO
    f(self);
    Ok(DistProcJoinHandle{})
  }

  pub fn rank(&self) -> usize {
    self.rank
  }

  pub fn num_ranks(&self) -> usize {
    self.nranks
  }

  pub fn barrier(&self) {
    mpi_barrier(&mut MPIComm::world()).unwrap();
  }

  pub fn allreduce_sum<T: MPIDataTypeExt + Copy>(&self, buf: &mut [T]) {
    let ptr = buf.as_mut_ptr();
    let len = buf.len();
    unsafe { mpi_allreduce(
        ptr, len,
        ptr, len,
        MPIReduceOp::Sum,
        &mut MPIComm::world(),
    ).unwrap() };
  }
}
