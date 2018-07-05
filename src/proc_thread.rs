/*
Copyright 2018 Peter Jin

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

use std::sync::{Arc, Barrier};
use std::thread;

pub struct ThreadProcGroup {
  closed:   bool,
  nranks:   usize,
  rank_ctr: usize,
  barrier:  Arc<Barrier>,
}

impl ThreadProcGroup {
  pub fn new(num_ranks: usize) -> Self {
    ThreadProcGroup{
      closed:   false,
      nranks:   num_ranks,
      rank_ctr: 0,
      barrier:  Arc::new(Barrier::new(num_ranks)),
    }
  }
}

impl Iterator for ThreadProcGroup {
  type Item = ThreadProc;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    let rank = self.rank_ctr;
    self.rank_ctr += 1;
    if self.rank_ctr == self.nranks {
      self.closed = true;
    }
    Some(ThreadProc{
      rank:     rank,
      nranks:   self.nranks,
      barrier:  self.barrier.clone(),
    })
  }
}

pub struct ThreadProcJoinHandle {
}

impl Drop for ThreadProcJoinHandle {
  fn drop(&mut self) {
    // TODO
  }
}

impl ThreadProcJoinHandle {
  pub fn join(self) {
  }
}

#[derive(Clone)]
pub struct ThreadProc {
  rank:     usize,
  nranks:   usize,
  barrier:  Arc<Barrier>,
}

impl ThreadProc {
  pub fn spawn<F>(self, f: F) -> Result<ThreadProcJoinHandle, ()> where F: FnOnce(ThreadProc) + Send + 'static {
    thread::spawn(|| {
      f(self);
    });
    Ok(ThreadProcJoinHandle{})
  }

  pub fn rank(&self) -> usize {
    self.rank
  }

  pub fn num_ranks(&self) -> usize {
    self.nranks
  }

  pub fn barrier(&self) {
    self.barrier.wait();
  }

  pub fn allreduce_sum<T: Copy>(&self, _buf: &mut [T]) {
    // TODO
  }
}
