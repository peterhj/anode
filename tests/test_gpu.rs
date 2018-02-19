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
  //let x: Rc<AOp<V=Val<_>>> = src(Rc::new(|pool: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, pool.conn())));
  let x: Rc<AOp<V=_>> = src(Rc::new(|pool: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, pool.conn())));
  x._apply(txn());
}

#[test]
fn test_gpu_zeros_eval() {
  //let ctx = implicit_ctx().multi_gpu_device().unwrap().gpu_device(0);
  //let conn = ctx.conn();
  //let x: Rc<AOp<V=Val<GPUDeviceArray1d<f32>>>> = zeros(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  //let x: Rc<AOp<V=Val<_>>> = zeros(GPUDeviceArray1d::<f32>::zeros(1024, conn));
  //let x: Rc<AOp<V=Val<_>>> = zeros(Rc::new(|pool: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, pool.conn())));
  let x: Rc<AOp<V=_>> = zeros(Rc::new(|pool: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, pool.conn())));
  x.eval(txn());
  println!("DEBUG: sleeping...");
  sleep(Duration::from_secs(10));
  println!("DEBUG: done sleeping");
}

#[test]
fn test_gpu_mux() {
  let x = zeros(Rc::new(|stream: GPUDeviceStreamPool| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, stream.conn())
  }));
  /*let y = x.gpu_mux(GPUDeviceId(0));
  let y_node: Rc<ANode> = y.clone();
  let y_op: Rc<AOp<V=_>> = y.clone();*/
  //let (y_node, y_op) = x.gpu_mux(GPUDeviceId(0));
  let y_op: Rc<AOp<V=_>> = x.gpu_mux(GPUDeviceId(0));
  println!("DEBUG: test: eval...");
  y_op.eval(txn());
}
