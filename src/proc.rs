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

use std::sync::{ONCE_INIT, Once};

static DIST_PROC_GROUP_ONCE: Once = ONCE_INIT;

pub struct DistProcGroup {
  closed:   bool,
}

impl Default for DistProcGroup {
  fn default() -> Self {
    DIST_PROC_GROUP_ONCE.call_once(|| {
      // TODO: use this to init MPI.
      //MPI::init()
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
    // TODO
    let rank = 0;
    //let rank = MPIComm::world().rank() as usize;
    Some(DistProc{rank})
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

pub struct DistProc {
  rank: usize,
}

impl DistProc {
  pub fn spawn<F>(self, f: F) -> Result<DistProcJoinHandle, ()> where F: FnOnce(usize) + 'static {
    // TODO
    f(self.rank);
    Ok(DistProcJoinHandle{})
  }
}
