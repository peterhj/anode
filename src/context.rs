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

use arrayidx::*;
#[cfg(feature = "gpu")] use cuda_coll::*;
#[cfg(feature = "gpu")] use gpudevicemem::{*, array::*, utils::*};
use memarray::*;
use parking_lot::{Mutex};
use rand::{thread_rng};
use rand::rngs::{ThreadRng};

use std::cell::{RefCell};
//use std::collections::{VecDeque};
//#[cfg(feature = "mpi")] use std::env;
//#[cfg(feature = "mpi")] use std::ffi::{CString};
use std::intrinsics::{type_name};
use std::ptr::{null_mut};
use std::rc::{Rc};
use std::sync::{Arc};

static NCCL_GROUP_MUTEX: Mutex<()> = Mutex::new(());

lazy_static! {
  static ref DEFAULT_CTX: Mutex<Option<DefaultCtx>> = Mutex::new(None);
}

thread_local! {
  static IMPLICIT:  RefCell<Vec<Rc<ExecutionCtx + 'static>>> = RefCell::new(vec![]);
  //static STREAM:    RefCell<VecDeque<Rc<ExecutionCtx + 'static>>> = RefCell::new(VecDeque::new());
}

pub fn default_ctx() -> impl ExecutionCtx {
  let mut ctx = DEFAULT_CTX.lock();
  if ctx.is_none() {
    *ctx = Some(DefaultCtx::default());
  }
  (*ctx.as_ref().unwrap()).clone()
}

pub fn implicit_ctx() -> Rc<ExecutionCtx + 'static> {
  IMPLICIT.with(|stack| {
    let mut stack = stack.borrow_mut();
    if stack.is_empty() {
      // If there is no context, create a `DefaultCtx`.
      stack.push(Rc::new(default_ctx()));
    }
    let ctx = stack.last().unwrap().clone();
    /*STREAM.with(|stream| {
      let mut stream = stream.borrow_mut();
      match stream.len() {
        0 => {}
        1 => {
          let prev_ctx = stream.pop_front().unwrap();
          // TODO: if `prev_ctx` is different from `ctx`, synchronize.
          /*if prev_ctx != ctx {
            prev_ctx.synchronize();
          }*/
        }
        _ => unreachable!(),
      }
      stream.push_back(ctx.clone());
    });*/
    ctx
  })
}

pub fn push_ctx<Ctx: ExecutionCtx + 'static>(ctx: Ctx) -> CtxGuard {
  IMPLICIT.with(|stack| {
    let mut stack = stack.borrow_mut();
    stack.push(Rc::new(ctx));
  });
  CtxGuard
}

pub trait ExecutionCtx {
  fn _debug_print(&self) {
    println!("DEBUG: ExecutionCtx: debug: {}", unsafe { type_name::<Self>() });
  }

  fn synchronize(&self) { unimplemented!(); }

  fn slow_rng(&self) -> ThreadRng {
    // FIXME
    thread_rng()
  }

  fn maybe_thread_pool(&self) -> Option<ThreadPoolCtx> { None }
  #[cfg(feature = "gpu")] fn maybe_gpu(&self) -> Option<GPUDeviceCtx> { None }
  #[cfg(feature = "gpu")] fn maybe_multi_gpu(&self) -> Option<MultiGPUDeviceCtx> { None }
  #[cfg(feature = "mpi")] fn maybe_mpi(&self) -> Option<MPIProcessCtx> { None }

  fn thread_pool(&self) -> ThreadPoolCtx {
    match self.maybe_thread_pool() {
      None => panic!("no thread pool ctx"),
      Some(ctx) => ctx,
    }
  }

  #[cfg(feature = "gpu")]
  fn gpu(&self) -> GPUDeviceCtx {
    match self.maybe_gpu() {
      None => panic!("no GPU device ctx"),
      Some(ctx) => ctx,
    }
  }

  #[cfg(feature = "gpu")]
  fn multi_gpu(&self) -> MultiGPUDeviceCtx {
    match self.maybe_multi_gpu() {
      None => panic!("no multi-GPU device ctx"),
      Some(ctx) => ctx,
    }
  }

  #[cfg(feature = "mpi")]
  fn mpi_rank(&self) -> MPIProcessCtx {
    // FIXME
    unimplemented!();
    /*match self.maybe_mpi_rank() {
      None => panic!("no MPI process ctx"),
      Some(ctx) => ctx,
    }*/
  }
}

pub struct CtxGuard;

impl !Send for CtxGuard {}
impl !Sync for CtxGuard {}

impl Drop for CtxGuard {
  fn drop(&mut self) {
    IMPLICIT.with(|stack| {
      let mut stack = stack.borrow_mut();
      let maybe_ctx = stack.pop();
      assert!(maybe_ctx.is_some());
    });
  }
}

#[cfg(not(feature = "gpu"))]
pub type DefaultCtx = NullCtx;

#[cfg(feature = "gpu")]
pub type DefaultCtx = MultiGPUDeviceCtx;

#[derive(Clone)]
pub struct NullCtx;

impl Default for NullCtx {
  fn default() -> Self {
    NullCtx
  }
}

impl ExecutionCtx for NullCtx {
}

pub struct ThreadPoolCtx {
}

impl ThreadPoolCtx {
  pub fn barrier(&self) {
    // TODO
  }
}

#[cfg(feature = "gpu")]
pub struct NcclState {
  comm_id:  NcclUniqueId,
  comm:     NcclComm,
  rank:     i32,
  max_rank: i32,
}

#[cfg(feature = "gpu")]
pub struct GPUDeviceCtx {
  // TODO: optional nccl stuff.
  pool:         GPUDeviceStreamPool,
  nccl_state:   Option<Arc<Mutex<NcclState>>>,
}

#[cfg(feature = "gpu")]
impl ExecutionCtx for GPUDeviceCtx {
  /*fn _debug_print(&self) {
    println!("DEBUG: GPUDeviceCtx");
  }*/

  fn maybe_gpu(&self) -> Option<GPUDeviceCtx> {
    Some(GPUDeviceCtx{
      pool:         self.pool.clone(),
      nccl_state:   self.nccl_state.clone(),
    })
  }

  fn maybe_multi_gpu(&self) -> Option<MultiGPUDeviceCtx> {
    Some(MultiGPUDeviceCtx{
      md_pools:     vec![self.pool.clone()],
      nccl_states:  vec![self.nccl_state.clone()],
    })
  }
}

#[cfg(feature = "gpu")]
impl Default for GPUDeviceCtx {
  fn default() -> Self {
    GPUDeviceCtx{
      pool:         GPUDeviceStreamPool::new(GPUDeviceId(0)),
      nccl_state:   None,
    }
  }
}

#[cfg(feature = "gpu")]
impl GPUDeviceCtx {
  pub fn pool(&self) -> GPUDeviceStreamPool {
    self.pool.clone()
  }

  pub fn conn(&mut self) -> GPUDeviceConn {
    self.pool.conn()
  }
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct MultiGPUDeviceCtx {
  md_pools:     Vec<GPUDeviceStreamPool>,
  nccl_states:  Vec<Option<Arc<Mutex<NcclState>>>>,
}

#[cfg(feature = "gpu")]
impl ExecutionCtx for MultiGPUDeviceCtx {
  /*fn _debug_print(&self) {
    println!("DEBUG: MultiGPUDeviceCtx");
  }*/

  fn maybe_gpu(&self) -> Option<GPUDeviceCtx> {
    Some(self.gpu(GPUDeviceId(0)))
  }

  fn maybe_multi_gpu(&self) -> Option<MultiGPUDeviceCtx> {
    Some(MultiGPUDeviceCtx{
      md_pools:     self.md_pools.clone(),
      nccl_states:  self.nccl_states.clone(),
    })
  }
}

#[cfg(feature = "gpu")]
impl Default for MultiGPUDeviceCtx {
  fn default() -> Self {
    let mut md_pools = vec![];
    let mut nccl_states  = vec![];
    let count = GPUDeviceId::count();
    for rank in 0 .. count {
      let pool = GPUDeviceStreamPool::new(GPUDeviceId(rank as _));
      md_pools.push(pool);
    }
    enable_gpu_peer_access(&mut md_pools);
    let comm_id = NcclUniqueId::create().unwrap();
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start() };
    for rank in 0 .. count {
      let nccl_state = {
        let conn = md_pools[rank].conn();
        let comm = NcclComm::init_rank(rank as _, count as _, comm_id.clone()).unwrap();
        NcclState{
          comm_id:  comm_id.clone(),
          comm:     comm,
          rank:     rank as _,
          max_rank: count as _,
        }
      };
      nccl_states.push(Some(Arc::new(Mutex::new(nccl_state))));
    }
    unsafe { NcclComm::group_end() };
    unsafe { NCCL_GROUP_MUTEX.raw_unlock() };
    MultiGPUDeviceCtx{
      md_pools:     md_pools,
      nccl_states:  nccl_states,
    }
  }
}

#[cfg(feature = "gpu")]
impl MultiGPUDeviceCtx {
  pub fn num_gpus(&self) -> usize {
    self.md_pools.len()
  }

  pub fn gpu(&self, device: GPUDeviceId) -> GPUDeviceCtx {
    assert!(device.rank() < self.md_pools.len(),
        "MultiGPUDeviceCtx: trying to activate an invalid device");
    GPUDeviceCtx{
      pool:         self.md_pools[device.rank()].clone(),
      nccl_state:   self.nccl_states[device.rank()].clone(),
    }
  }

  pub fn sync_broadcast_group<T>(&mut self, src: GPUDeviceArrayView1d<T>, mut dst: Vec<GPUDeviceArrayViewMut1d<T>>) where T: NcclDataType {
    let root_dev = src.device();
    {
      let conn = self.md_pools[root_dev.rank()].conn();
      dst[root_dev.rank()].copy(src, conn);
    }
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      conn.sync();
    }
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start() };
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      let mut stream = conn.cuda_stream();
      let mut nccl_state = self.nccl_states[rank].as_ref().unwrap().lock();
      // FIXME: size checks.
      assert_eq!(dst[rank].size(), dst[0].size());
      if dst[rank].size().is_packed(&dst[rank].stride()) {
        let res = unsafe { nccl_state.comm.broadcast(
            dst[rank].as_mut_dptr(),
            dst[rank].size(),
            root_dev.0,
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      } else {
        unimplemented!();
      }
    }
    unsafe { NcclComm::group_end() };
    unsafe { NCCL_GROUP_MUTEX.raw_unlock() };
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      conn.sync();
    }
  }

  pub fn sync_reduce_group<T>(&mut self, src: Vec<GPUDeviceArrayView1d<T>>, dst: GPUDeviceArrayViewMut1d<T>, op: NcclReduceOp) where T: NcclDataType {
    let root_dev = dst.device();
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      conn.sync();
    }
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start() };
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      let mut stream = conn.cuda_stream();
      let mut nccl_state = self.nccl_states[rank].as_ref().unwrap().lock();
      // FIXME: size checks.
      assert_eq!(src[rank].size(), src[0].size());
      assert_eq!(src[rank].size(), dst.size());
      if src[rank].size().is_packed(&src[rank].stride()) {
        let res = unsafe { nccl_state.comm.reduce(
            src[rank].as_dptr(),
            if rank == root_dev.rank() { dst.as_mut_dptr() } else { null_mut() },
            src[rank].size(),
            op,
            root_dev.0,
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      } else {
        unimplemented!();
      }
    }
    unsafe { NcclComm::group_end() };
    unsafe { NCCL_GROUP_MUTEX.raw_unlock() };
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      conn.sync();
    }
  }

  pub fn sync_allreduce_group<T>(&mut self, src: Vec<GPUDeviceArrayView1d<T>>, dst: Vec<GPUDeviceArrayViewMut1d<T>>, op: NcclReduceOp) where T: NcclDataType {
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      conn.sync();
    }
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start() };
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      let mut stream = conn.cuda_stream();
      let mut nccl_state = self.nccl_states[rank].as_ref().unwrap().lock();
      // FIXME: size checks.
      assert_eq!(src[rank].size(), src[0].size());
      assert_eq!(src[rank].size(), dst[rank].size());
      if src[rank].size().is_packed(&src[rank].stride()) {
        let res = unsafe { nccl_state.comm.all_reduce(
            src[rank].as_dptr(),
            dst[rank].as_mut_dptr(),
            src[rank].size(),
            op,
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      } else {
        unimplemented!();
      }
    }
    unsafe { NcclComm::group_end() };
    unsafe { NCCL_GROUP_MUTEX.raw_unlock() };
    for rank in 0 .. self.num_gpus() {
      let conn = self.md_pools[rank].conn();
      conn.sync();
    }
  }
}

#[cfg(feature = "gpu")]
pub struct LazyMultiGPUDeviceCtx {
  md_pools: RefCell<Vec<Option<GPUDeviceStreamPool>>>,
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct SharedMuxGPUDeviceCtxBuilder {
  // TODO: nccl and other comm stuff.
  num_devs: usize,
  //nccl_uid: NcclUniqueId,
}

#[cfg(feature = "gpu")]
impl SharedMuxGPUDeviceCtxBuilder {
  pub fn into_ctx(self, device: GPUDeviceId) -> GPUDeviceCtx {
    // TODO: create a new stream for this ctx.
    //let nccl_comm = NcclComm::init_rank(self.nccl_uid, self.num_devs, device_index).unwrap();
    /*GPUDeviceCtx{
      pool:         pool,
      nccl_comm:    Some(nccl_comm),
    }*/
    unimplemented!();
  }
}

#[cfg(feature = "mpi")]
pub struct MPIProcessCtx {
  //rank: i32,
  //size: i32,
}

#[cfg(feature = "mpi")]
impl Default for MPIProcessCtx {
  fn default() -> Self {
    //let rank = MPIComm::world().rank().unwrap();
    //let size = MPIComm::world().size().unwrap();
    MPIProcessCtx{
      //rank: rank,
      //size: size,
    }
  }
}

#[cfg(feature = "mpi")]
pub struct MPIProcessGroup {
}

#[cfg(feature = "mpi")]
impl MPIProcessGroup {
  // FIXME
  /*pub fn init() -> Vec<CString> {
    let args: Vec<_> = env::args_os().collect();
    let mut raw_argv = Vec::with_capacity(args.len());
    for arg in args.drain() {
      raw_argv.push(CString::new(arg).into_raw());
    }
    {
      let mut argc: i32 = raw_argv.len() as _;
      let mut raw_argv_copy = raw_argv.clone();
      let mut argv = raw_argv_copy.as_mut_slice().as_mut_ptr();
      let mut provided: i32 = -1;
      let status = unsafe { MPI_Init_thread(
          &mut argc as *mut _,
          &mut argv as *mut _,
          MPI_THREAD_SERIALIZED,
          &mut provided as *mut _,
      ) };
      assert_eq!(status, MPI_SUCCESS);
      assert!(MPI_THREAD_SERIALIZED <= provided);
    }
    let mut args = Vec::new();
    for raw_arg in raw_argv.drain() {
      args.push(unsafe { CString::from_raw(raw_arg) }.into_string().unwrap());
    }
    args
  }

  pub fn shutdown() {
    let status = unsafe { MPI_Barrier(MPI_COMM_WORLD) };
    assert_eq!(status, MPI_SUCCESS);
    let status = unsafe { MPI_Finalize() };
    assert_eq!(status, MPI_SUCCESS);
  }*/
}

pub struct CollectionCtx {
  ctxs: Vec<Rc<ExecutionCtx + 'static>>,
}

impl CollectionCtx {
  pub fn ctx(&self, ctx_index: usize) -> Rc<ExecutionCtx + 'static> {
    self.ctxs[ctx_index].clone()
  }
}
