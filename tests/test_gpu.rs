extern crate anode;
extern crate gpudevicemem;
extern crate memarray;

use anode::*;
use anode::context::*;
use anode::ops::*;
use anode::ops_gpu::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;

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

  let x: Rc<AOp<_>> = zeros(Rc::new(|stream: GPUDeviceStreamPool| {
    let mut h_arr = MemArray1d::<f32>::zeros(1024);
    {
      let mut h_arr = h_arr.as_view_mut();
      let mut x = h_arr.flat_slice_mut().unwrap();
      for k in 0 .. 1024 {
        x[k] = k as f32;
      }
    }
    let mut arr = GPUDeviceArray1d::<f32>::zeros(1024, stream.conn());
    arr.as_view_mut().copy_mem(h_arr.as_view(), stream.conn());

    let mut h_arr2 = MemArray1d::<f32>::zeros(1024);
    arr.as_view().dump_mem(h_arr2.as_view_mut(), stream.conn());
    {
      let mut h_arr2 = h_arr2.as_view();
      let x = h_arr2.flat_slice().unwrap();
      for k in 0 .. 1024 {
        assert_eq!(x[k], k as f32);
      }
    }

    arr
  }));

  let t = txn();
  x.eval(t);
  x.eval(t);
  x.eval(t);

  {
    let ctx = implicit_ctx().gpu().unwrap();
    let stream = ctx.pool();
    let x = x.value();
    let v = x.get(t);
    let mut h_arr = MemArray1d::<f32>::zeros(1024);
    v.as_view().dump_mem(h_arr.as_view_mut(), stream.conn());
    {
      let mut h_arr = h_arr.as_view();
      let x = h_arr.flat_slice().unwrap();
      for k in 0 .. 1024 {
        assert_eq!(x[k], 0.0);
      }
    }
  }

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
