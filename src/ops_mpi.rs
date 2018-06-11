use ::*;
use context::*;
use ops::*;

use arithmetic::*;
use arrayidx::*;
//use cuda_blas::*;
//use cuda_dnn::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
//use rand::{Rng};

use std::cell::{RefMut};
//use std::marker::{PhantomData};
//use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::sync::{Arc};

pub struct MPIBroadcastOp;
pub struct MPIReduceOp;
pub struct MPIAllreduceOp;
pub struct MPIReduceScatterOp;
pub struct MPIAllgatherOp;

pub trait MPIBroadcastExt<V> {
  fn mpi_broadcast_src(rank: usize) -> Val<V>;
  fn mpi_broadcast(self, rank: usize) -> Val<V>;
}

pub trait MPIBroadcastOpExt<A> {
  fn build(rank: usize) -> Val<A>;
}

pub fn mpi_broadcast_src<V: 'static>(rank: usize) -> Val<V> where Val<V>: MPIBroadcastExt<V> {
  <Val<V> as MPIBroadcastExt<V>>::mpi_broadcast_src(rank)
}

pub trait MPIReduceExt<V> {
  fn mpi_reduce(self, rank: usize);
}

pub trait MPIAllreduceExt<V> {
  fn mpi_allreduce(self) -> Val<V>;
}

pub trait MPIReduceScatterExt<V> {
  fn mpi_reduce_scatter(self) -> Val<V>;
}

pub trait MPIAllgatherExt<V> {
  fn mpi_allgather(self) -> Val<V>;
}

#[cfg(feature = "gpu")]
pub mod gpu {

use ::*;
use context::*;
use ops::*;
use ops_mpi::*;

use arithmetic::*;
use arrayidx::*;
//use cuda_blas::*;
//use cuda_dnn::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
//use rand::{Rng};

use std::cell::{RefMut};
//use std::marker::{PhantomData};
//use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::sync::{Arc};

//impl<T: Copy> MPIBroadcastExt<GPUDeviceOuterBatchArray3d<T>> for Val<GPUDeviceOuterBatchArray3d<T>> {
impl<A> MPIBroadcastExt<A> for Val<A>
where A: GPUDeviceAsync + 'static,
{
  fn mpi_broadcast_src(rank: usize) -> Self {
    // TODO
    unimplemented!();
  }

  fn mpi_broadcast(self, rank: usize) -> Self {
    // TODO
    unimplemented!();
  }
}

impl<A> MPIBroadcastOpExt<A> for MPIBroadcastOp
where A: GPUDeviceAsync + 'static,
{
  fn build(rank: usize) -> Val<A> {
    let ext = OpExt{
      make_val: {
        Box::new(move |state: RefMut<Self>| {
          RWVal::from(Arc::new(move |txn: Txn| {
            // TODO
            unimplemented!();
          }))
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          if let Some(_) = output.write(txn) {
            // TODO
            unimplemented!();
          }
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, state: RefMut<_>, sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    // TODO
    unimplemented!();
  }
}

}
