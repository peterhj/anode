extern crate anode;
extern crate devicemem_gpu;

use anode::*;
use anode::context::*;
use anode::ops::*;
use anode::ops_gpu::*;
use devicemem_gpu::*;
use devicemem_gpu::array::*;

use std::rc::{Rc};
use std::thread::{sleep};
use std::time::{Duration};

#[test]
fn test_pool() {
  let pool = GPUDeviceStreamPool::new(GPUDeviceId(0));
}

#[test]
#[should_panic]
fn test_gpu_src_eval() {
  //let ctx = implicit_ctx().gpu_device().unwrap();
  //let conn = ctx.conn();
  //let x: Rc<AOp<V=Val<_>>> = src(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  let x: Rc<AOp<V=Val<_>>> = src(Rc::new(|pool: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, pool.conn())));
  x._eval(txn());
}

#[test]
fn test_gpu_zeros_eval() {
  //let ctx = implicit_ctx().multi_gpu_device().unwrap().gpu_device(0);
  //let conn = ctx.conn();
  //let x: Rc<AOp<V=Val<GPUDeviceArray1d<f32>>>> = zeros(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  //let x: Rc<AOp<V=Val<_>>> = zeros(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  let x: Rc<AOp<V=Val<_>>> = zeros(Rc::new(|pool: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, pool.conn())));
  x._eval(txn());
  println!("DEBUG: sleeping...");
  sleep(Duration::from_secs(10));
  println!("DEBUG: done sleeping");
}
