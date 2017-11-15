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
use std::rc::{Rc};

thread_local! {
  static CTX_STACK: RefCell<Vec<Rc<ExecutionCtx + 'static>>> = RefCell::new(vec![]);
}

pub fn implicit_ctx() -> Rc<ExecutionCtx + 'static> {
  CTX_STACK.with(|stack| {
    let mut stack = stack.borrow_mut();
    if stack.is_empty() {
      // TODO: if there is no context, create a `DefaultCtx`.
      stack.push(Rc::new(DefaultCtx::new()));
    }
    stack.last().unwrap().clone()
  })
}

pub trait ExecutionCtx {
  fn push(&self) -> CtxGuard { unimplemented!(); }
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
#[cfg(feature = "gpu")]
pub type DefaultCtx = GPUDeviceCtx;
//#[cfg(feature = "gpu")]
//pub type DefaultCtx = MultiGPUDeviceCtx;

pub struct NullCtx;

pub struct ThreadPoolCtx {
}

impl ThreadPoolCtx {
  pub fn barrier(&self) {
    // TODO
  }
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct GPUDeviceCtx {
  pool: GPUDeviceStreamPool,
}

#[cfg(feature = "gpu")]
impl ExecutionCtx for GPUDeviceCtx {
  fn gpu_device(&self) -> Option<GPUDeviceCtx> {
    Some(self.clone())
  }
}

#[cfg(feature = "gpu")]
impl GPUDeviceCtx {
  pub fn new() -> Self {
    GPUDeviceCtx{
      pool: GPUDeviceStreamPool::new(GPUDeviceId(0)),
    }
  }

  pub fn pool(&self) -> &GPUDeviceStreamPool {
    &self.pool
  }
}

#[cfg(feature = "gpu")]
pub struct MultiGPUDeviceCtx {
  md_pools: Vec<GPUDeviceStreamPool>,
}

/*#[cfg(feature = "gpu")]
impl ExecutionCtx for Rc<MultiGPUDeviceCtx> {
  fn gpu_device(&self) -> Option<Rc<GPUDeviceCtx>> {
    // TODO
    unimplemented!();
  }
}*/

#[cfg(feature = "gpu")]
impl MultiGPUDeviceCtx {
  pub fn gpu_device(&self, device_index: usize) -> GPUDeviceCtx {
    // TODO
    unimplemented!();
  }
}
