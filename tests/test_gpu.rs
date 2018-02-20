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
fn test_stream() {
  println!();
  let stream = GPUDeviceStreamPool::new(GPUDeviceId(0));
}

#[test]
#[should_panic]
fn test_gpu_src_eval_fail() {
  println!();
  let x: Rc<AOp<_>> = src(Rc::new(|stream: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, stream.conn())));
  x.eval(txn());
}

#[test]
fn test_gpu_zeros_eval() {
  println!();
  let x: Rc<AOp<_>> = zeros(Rc::new(|stream: GPUDeviceStreamPool| GPUDeviceArray1d::<f32>::zeros(1024, stream.conn())));
  let t = txn();
  x.eval(t);
  x.eval(t);
  x.eval(t);
  println!("DEBUG: sleeping...");
  sleep(Duration::from_secs(5));
  println!("DEBUG: done sleeping");
}

#[test]
#[should_panic]
fn test_gpu_mux_fail() {
  println!();
  let x = zeros(Rc::new(|stream: GPUDeviceStreamPool| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, stream.conn())
  }));
  let y: Rc<AOp<_>> = x.gpu_mux(GPUDeviceId(1));
  println!("DEBUG: test: eval...");
  y.eval(txn());
}

#[test]
fn test_gpu_mux() {
  println!();
  let x = zeros(Rc::new(|stream: GPUDeviceStreamPool| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, stream.conn())
  }));
  let y: Rc<AOp<_>> = x.gpu_mux(GPUDeviceId(0));
  println!("DEBUG: test: eval...");
  y.eval(txn());
}
