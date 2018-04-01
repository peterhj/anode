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

#[cfg(feature = "gpu")] use cuda_coll::*;
#[cfg(feature = "gpu")] use gpudevicemem::*;
use parking_lot::{Mutex};

use std::cell::{RefCell};
use std::collections::{VecDeque};
use std::rc::{Rc};
use std::sync::{Arc};

static NCCL_GROUP_MUTEX: Mutex<()> = Mutex::new(());

thread_local! {
  static IMPLICIT:  RefCell<Vec<Rc<ExecutionCtx + 'static>>> = RefCell::new(vec![]);
  static STREAM:    RefCell<VecDeque<Rc<ExecutionCtx + 'static>>> = RefCell::new(VecDeque::new());
}

pub fn implicit_ctx() -> Rc<ExecutionCtx + 'static> {
  IMPLICIT.with(|stack| {
    let mut stack = stack.borrow_mut();
    if stack.is_empty() {
      // TODO: if there is no context, create a `DefaultCtx`.
      stack.push(Rc::new(DefaultCtx::default()));
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

pub trait ExecutionCtx {
  fn synchronize(&self) { unimplemented!(); }
  fn thread_pool(&self) -> Option<Rc<ThreadPoolCtx>> { None }
  #[cfg(feature = "gpu")] fn gpu(&self) -> Option<Rc<GPUDeviceCtx>> { None }
  #[cfg(feature = "gpu")] fn multi_gpu(&self) -> Option<Rc<MultiGPUDeviceCtx>> { None }
}

pub fn push_ctx(ctx: Rc<ExecutionCtx + 'static>) -> CtxGuard {
  IMPLICIT.with(|stack| {
    let mut stack = stack.borrow_mut();
    stack.push(ctx);
  });
  CtxGuard
}

pub struct CtxGuard;

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

//#[cfg(feature = "gpu")]
//pub type DefaultCtx = GPUDeviceCtx;
#[cfg(feature = "gpu")]
pub type DefaultCtx = MultiGPUDeviceCtx;

//#[derive(Clone)]
pub struct NullCtx;

impl Default for NullCtx {
  fn default() -> Self {
    NullCtx
  }
}

impl ExecutionCtx for NullCtx {
}

//#[derive(Clone)]
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
  fn gpu(&self) -> Option<Rc<GPUDeviceCtx>> {
    Some(Rc::new(GPUDeviceCtx{
      pool:         self.pool.clone(),
      nccl_state:   self.nccl_state.clone(),
    }))
  }

  fn multi_gpu(&self) -> Option<Rc<MultiGPUDeviceCtx>> {
    Some(Rc::new(MultiGPUDeviceCtx{
      md_pools:     vec![self.pool.clone()],
      nccl_states:  vec![self.nccl_state.clone()],
    }))
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

  pub fn conn(&self) -> GPUDeviceConn {
    self.pool.conn()
  }
}

#[cfg(feature = "gpu")]
pub struct MultiGPUDeviceCtx {
  md_pools:     Vec<GPUDeviceStreamPool>,
  nccl_states:  Vec<Option<Arc<Mutex<NcclState>>>>,
}

#[cfg(feature = "gpu")]
impl ExecutionCtx for MultiGPUDeviceCtx {
  fn gpu(&self) -> Option<Rc<GPUDeviceCtx>> {
    Some(self.gpu(GPUDeviceId(0)))
  }

  fn multi_gpu(&self) -> Option<Rc<MultiGPUDeviceCtx>> {
    Some(Rc::new(MultiGPUDeviceCtx{
      md_pools:     self.md_pools.clone(),
      nccl_states:  self.nccl_states.clone(),
    }))
  }
}

#[cfg(feature = "gpu")]
impl Default for MultiGPUDeviceCtx {
  fn default() -> Self {
    let mut md_pools = vec![];
    let mut nccl_states  = vec![];
    let count = GPUDeviceId::count();
    let comm_id = NcclUniqueId::create().unwrap();
    for rank in 0 .. count {
      let pool = GPUDeviceStreamPool::new(GPUDeviceId(rank as _));
      md_pools.push(pool);
    }
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

  pub fn gpu(&self, device: GPUDeviceId) -> Rc<GPUDeviceCtx> {
    assert!(device.rank() < self.md_pools.len(),
        "MultiGPUDeviceCtx: trying to activate an invalid device");
    Rc::new(GPUDeviceCtx{
      pool:         self.md_pools[device.rank()].clone(),
      nccl_state:   self.nccl_states[device.rank()].clone(),
    })
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
  pub fn into_ctx(self, device_index: usize) -> GPUDeviceCtx {
    // TODO: create a new stream for this ctx.
    //let nccl_comm = NcclComm::init_rank(self.nccl_uid, self.num_devs, device_index).unwrap();
    /*GPUDeviceCtx{
      pool:         pool,
      nccl_comm:    Some(nccl_comm),
    }*/
    unimplemented!();
  }
}

pub struct CollectionCtx {
  ctxs: Vec<Rc<ExecutionCtx + 'static>>,
}

impl CollectionCtx {
  pub fn ctx(&self, ctx_index: usize) -> Rc<ExecutionCtx + 'static> {
    self.ctxs[ctx_index].clone()
  }
}
