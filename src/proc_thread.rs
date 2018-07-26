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

use proc::*;

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
  type Item = ThreadProcSpawner;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    let rank = self.rank_ctr;
    self.rank_ctr += 1;
    if self.rank_ctr == self.nranks {
      self.closed = true;
    }
    let proc = ThreadProc{
      rank:     rank,
      nranks:   self.nranks,
      barrier:  self.barrier.clone(),
    };
    Some(ThreadProcSpawner{proc})
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

pub struct ThreadProcSpawner {
  proc: ThreadProc,
}

impl ThreadProcSpawner {
  pub fn spawn<F>(self, f: F) -> Result<ThreadProcJoinHandle, ()> where F: FnOnce(ThreadProc) + Send + 'static {
    thread::spawn(|| {
      f(self.proc);
    });
    Ok(ThreadProcJoinHandle{})
  }
}

#[derive(Clone)]
pub struct ThreadProc {
  rank:     usize,
  nranks:   usize,
  barrier:  Arc<Barrier>,
}

impl Proc<usize> for ThreadProc {
  fn rank(&self) -> usize {
    self.rank
  }

  fn sup_rank(&self) -> usize {
    self.nranks
  }

  fn wait_barrier(&self) -> bool {
    self.barrier.wait().is_leader()
  }

  /*pub fn allreduce_sum<T: Copy>(&self, _buf: &mut [T]) {
    // TODO
  }*/
}

/*pub trait ProcIO<Buf: ?Sized> {
  type Tx: ProcTxOnce<Buf>;
  type Rx: ProcRxOnce<Buf>;*/
pub trait ProcIO {
  type Tx;
  type Rx;

  fn message(&self, src: usize, dst: usize) -> (Self::Tx, Self::Rx);
  fn allreduce_sum(&self) -> (Self::Tx, Self::Rx);
  fn broadcast(&self, root: usize) -> (Self::Tx, Self::Rx);
}

pub trait ProcTxOnce<Buf: ?Sized> {
  fn send(self, buf: &Buf);
}

pub trait ProcRxOnce<Buf: ?Sized> {
  fn recv(self, buf: &mut Buf);
}

//impl<T: Copy> ProcIO<[T]> for ThreadProc {
impl ProcIO for ThreadProc {
  type Tx = ThreadProcTx;
  type Rx = ThreadProcRx;

  fn message(&self, src: usize, dst: usize) -> (Self::Tx, Self::Rx) {
    (ThreadProcTx{closed: self.rank != src, src, nranks: self.nranks},
     ThreadProcRx{closed: self.rank != dst, dst, nranks: self.nranks})
  }

  fn allreduce_sum(&self) -> (Self::Tx, Self::Rx) {
    (ThreadProcTx{closed: false, src: self.rank, nranks: self.nranks},
     ThreadProcRx{closed: false, dst: self.rank, nranks: self.nranks})
  }

  fn broadcast(&self, root: usize) -> (Self::Tx, Self::Rx) {
    (ThreadProcTx{closed: self.rank != root, src: root, nranks: self.nranks},
     ThreadProcRx{closed: self.rank == root, dst: self.rank, nranks: self.nranks})
  }
}

pub struct ThreadProcTx {
  closed:   bool,
  src:      usize,
  nranks:   usize,
}

pub struct ThreadProcRx {
  closed:   bool,
  dst:      usize,
  nranks:   usize,
}

impl Drop for ThreadProcTx {
  fn drop(&mut self) {
    assert!(self.closed);
  }
}

impl<T: Copy> ProcTxOnce<[T]> for ThreadProcTx {
  fn send(mut self, buf: &[T]) {
    assert!(!self.closed);
    // TODO
    self.closed = true;
  }
}

impl Drop for ThreadProcRx {
  fn drop(&mut self) {
    assert!(self.closed);
  }
}

impl<T: Copy> ProcRxOnce<[T]> for ThreadProcRx {
  fn recv(mut self, buf: &mut [T]) {
    assert!(!self.closed);
    // TODO
    self.closed = true;
  }
}
