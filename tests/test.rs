extern crate anode;
//extern crate cuda;
extern crate devicemem_gpu;

use anode::*;
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
fn test_src_eval() {
  //let x: Rc<SrcOp<_, Val<GPUDeviceArray1d<f32>>>> = src(1);
  //let x: Rc<AOp<V=Val<GPUDeviceArray1d<f32>>>> = src(1024);
  let pool = GPUDeviceStreamPool::new(GPUDeviceId(0));
  let conn = pool.conn();
  //let x: Rc<AOp<V=Val<GPUDeviceArray1d<f32>>>> = src(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  let x: Rc<AOp<V=Val<_>>> = src(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  x.eval(txn());
}

#[test]
fn test_zeros_eval() {
  let pool = GPUDeviceStreamPool::new(GPUDeviceId(0));
  let conn = pool.conn();
  let x: Rc<AOp<V=Val<GPUDeviceArray1d<f32>>>> = zeros(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  x.eval(txn());
  println!("DEBUG: sleeping...");
  sleep(Duration::from_secs(10));
  println!("DEBUG: done sleeping");
}
