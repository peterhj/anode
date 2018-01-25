/*
Copyright 2017 the anode authors

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

#[cfg(feature = "gpu")] use devicemem_gpu::*;

use std::cell::{RefCell};
use std::collections::{VecDeque};
use std::rc::{Rc};

thread_local! {
  static IMPLICIT:  RefCell<Vec<Rc<ExecutionCtx + 'static>>> = RefCell::new(vec![]);
  static STREAM:    RefCell<VecDeque<Rc<ExecutionCtx + 'static>>> = RefCell::new(VecDeque::new());
}

pub fn implicit_ctx() -> Rc<ExecutionCtx + 'static> {
  IMPLICIT.with(|stack| {
    let mut stack = stack.borrow_mut();
    if stack.is_empty() {
      // TODO: if there is no context, create a `DefaultCtx`.
      stack.push(Rc::new(DefaultCtx::new()));
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
  //fn push(&self) -> CtxGuard { unimplemented!(); }
  fn synchronize(&self) { unimplemented!(); }
  fn thread_pool(&self) -> Option<ThreadPoolCtx> { None }
  #[cfg(feature = "gpu")] fn gpu_device(&self) -> Option<GPUDeviceCtx> { None }
  #[cfg(feature = "gpu")] fn multi_gpu_device(&self) -> Option<MultiGPUDeviceCtx> { None }
}

pub struct CtxGuard {
}

impl Drop for CtxGuard {
  fn drop(&mut self) {
    // TODO
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

//#[derive(Clone)]
pub struct ThreadPoolCtx {
}

impl ThreadPoolCtx {
  pub fn barrier(&self) {
    // TODO
  }
}

#[cfg(feature = "gpu")]
//#[derive(Clone)]
pub struct GPUDeviceCtx {
  // TODO: optional nccl stuff.
  pool: GPUDeviceStreamPool,
  //nccl_comm:    Option<NcclComm>,
}

#[cfg(feature = "gpu")]
impl ExecutionCtx for GPUDeviceCtx {
  fn gpu_device(&self) -> Option<GPUDeviceCtx> {
    Some(GPUDeviceCtx{pool: self.pool.clone()})
  }

  fn multi_gpu_device(&self) -> Option<MultiGPUDeviceCtx> {
    Some(MultiGPUDeviceCtx{
      md_pools: vec![self.pool.clone()],
    })
  }
}

#[cfg(feature = "gpu")]
impl GPUDeviceCtx {
  fn new() -> Self {
    GPUDeviceCtx{
      pool: GPUDeviceStreamPool::new(GPUDeviceId(0)),
    }
  }

  pub fn pool(&self) -> GPUDeviceStreamPool {
    self.pool.clone()
  }

  pub fn conn(&self) -> GPUDeviceConn {
    self.pool.conn()
  }
}

#[cfg(feature = "gpu")]
//#[derive(Clone)]
pub struct MultiGPUDeviceCtx {
  md_pools: Vec<GPUDeviceStreamPool>,
}

#[cfg(feature = "gpu")]
impl ExecutionCtx for MultiGPUDeviceCtx {
  fn gpu_device(&self) -> Option<GPUDeviceCtx> {
    Some(self.gpu_device(0))
  }

  fn multi_gpu_device(&self) -> Option<MultiGPUDeviceCtx> {
    Some(MultiGPUDeviceCtx{md_pools: self.md_pools.clone()})
  }
}

#[cfg(feature = "gpu")]
impl MultiGPUDeviceCtx {
  fn new() -> Self {
    // TODO: use all devices.
    MultiGPUDeviceCtx{
      md_pools: vec![GPUDeviceStreamPool::new(GPUDeviceId(0))],
    }
  }

  pub fn num_gpu_devices(&self) -> usize {
    self.md_pools.len()
  }

  pub fn gpu_device(&self, device_index: usize) -> GPUDeviceCtx {
    GPUDeviceCtx{pool: self.md_pools[device_index].clone()}
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
