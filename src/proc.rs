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

pub trait Proc<K>: Clone {
  fn rank(&self) -> K;
  fn sup_rank(&self) -> K;
  fn wait_barrier(&self) -> bool;
}

pub trait ProcSyncIO<T> {
  fn sync_allreduce_sum_inplace(&self, buf: &mut [T]);
}

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
