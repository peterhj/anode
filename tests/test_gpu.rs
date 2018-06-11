extern crate anode;
extern crate gpudevicemem;
extern crate memarray;
extern crate rand;

use anode::*;
use anode::context::*;
use anode::ops::*;
use anode::ops_gpu::*;
use anode::utils::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use rand::*;

use std::rc::{Rc};
use std::thread::{sleep};
use std::time::{Duration};

/*#[test]
fn test_stream() {
  println!();
  let stream = GPUDeviceStreamPool::new(GPUDeviceId(0));
}*/

#[test]
#[should_panic]
fn test_gpu_src_eval_fail() {
  println!();
  let x: Val<_> = src(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  x.eval(txn());
}

#[test]
fn test_gpu_random_src_eval() {
  println!();
  let x: Val<_> = random_bits(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<u32>::zeros(1024, conn)
  }));
  let t = txn();
  x.eval(t);
  x.reset();
  x.eval(t);
  let t = txn();
  x.eval(t);
  x.release();
  x.eval(t);
  let t = txn();
  x.eval(t);
  x.reset();
  x.eval(t);
}

#[test]
fn test_gpu_zeros_eval() {
  println!();

  let x: Val<_> = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    let mut h_arr = MemArray1d::<f32>::zeros(1024);
    {
      let mut h_arr = h_arr.as_view_mut();
      let mut x = h_arr.flat_slice_mut().unwrap();
      for k in 0 .. 1024 {
        x[k] = k as f32;
      }
    }
    let mut arr = GPUDeviceArray1d::<f32>::zeros(1024, conn.clone());
    arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());

    let mut h_arr2 = MemArray1d::<f32>::zeros(1024);
    arr.as_view().sync_dump_mem(h_arr2.as_view_mut(), conn.clone());
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
    let ctx = implicit_ctx().gpu();
    let mut stream = ctx.pool();
    //let x = x.value();
    let v = x.get(t);
    let mut h_arr = MemArray1d::<f32>::zeros(1024);
    v.as_view().sync_dump_mem(h_arr.as_view_mut(), stream.conn());
    {
      let mut h_arr = h_arr.as_view();
      let x = h_arr.flat_slice().unwrap();
      for k in 0 .. 1024 {
        assert_eq!(x[k], 0.0);
      }
    }
  }

  /*println!("DEBUG: sleeping...");
  sleep(Duration::from_secs(2));
  println!("DEBUG: done sleeping");*/
}

#[test]
fn test_gpu_zeros_fill_uniform() {
  println!();
  let x: Val<_> = touch(GPUDeviceArray1d::<f32>::uniform_fill(1024, -1.0, 1.0, &mut thread_rng()));
  let t = txn();
  x.eval(t);
  {
    let ctx = implicit_ctx().gpu();
    let mut stream = ctx.pool();
    let y = x.get(t);
    let mut z = MemArray1d::<f32>::zeros(1024);
    println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
    y.flat_view().unwrap().sync_dump_mem(z.as_view_mut(), stream.conn());
    println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
  }
}

#[test]
fn test_gpu_io_deserialize() {
  println!();
  let x: Val<_> = src(GPUDeviceOuterBatchArray3d::<f32>::zeros_fill(([32, 32, 3], 64)));
  let src = MemArray4d::<f32>::zeros([32, 32, 3, 64]);
  let t = txn();
  x.deserialize(t, &mut ArrayIO::new(src));
  x.eval(t);
}

#[test]
fn test_gpu_io_serialize() {
  println!();
  let x = touch(GPUDeviceOuterBatchArray3d::<f32>::uniform_fill(([32, 32, 3], 64), -1.0, 1.0, &mut thread_rng()));
  let dst = MemArray4d::<f32>::zeros([32, 32, 3, 64]);
  println!("DEBUG: {:?}", &dst.flat_view().unwrap().as_slice()[.. 10]);
  let t = txn();
  x.eval(t);
  let mut dst = ArrayIO::new(dst);
  x.serialize(t, &mut dst);
  let dst = dst.take();
  println!("DEBUG: {:?}", &dst.flat_view().unwrap().as_slice()[.. 10]);
}

#[test]
#[should_panic]
fn test_gpu_mux_fail() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  //let y: Rc<AOp<_>> = x.gpu_mux(GPUDeviceId(1));
  let y: Val<_> = x.gpu_mux(GPUDeviceId(1));
  println!("DEBUG: test: eval...");
  y.eval(txn());
}

#[test]
fn test_gpu_mux() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  //let y: Rc<AOp<_>> = x.gpu_mux(GPUDeviceId(0));
  let y: Val<_> = x.gpu_mux(GPUDeviceId(0));
  println!("DEBUG: test: eval...");
  y.eval(txn());
}

#[test]
fn test_gpu_mux_2() {
  println!();
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceOuterBatchArray1d::<f32>::zeros(1024, 1, conn)
  }));
  let x2 = x1.positive_clip();
  let y = x2.gpu_mux(GPUDeviceId(0));
  println!("DEBUG: test: eval...");
  let t = txn();
  y.eval(t);
}

#[test]
fn test_gpu_switch() {
  println!();
  let flag = TCell::new(false);
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let x2 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let y = switch(flag.clone(), x1, x2);
  let t = txn();
  flag.propose(t, |_| false);
  y.eval(t);
  let t = txn();
  flag.propose(t, |_| true);
  y.eval(t);
  // TODO
}

#[test]
fn test_gpu_adj() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  }));
  let mut sink = Sink::from(x.clone());
  let dx = x.adjoint(&mut sink);
}

#[test]
fn test_gpu_switch_adj() {
  println!();
  let flag = TCell::default();
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  }));
  let x2 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  }));
  let y = switch(flag.clone(), x1.clone(), x2.clone());
  println!("DEBUG: building adjoint");
  let mut sink = Sink::from(y.clone());
  let dy = y.adjoint(&mut sink).unwrap();
  let dx1 = x1.adjoint(&mut sink).unwrap();
  let dx2 = x2.adjoint(&mut sink).unwrap();
  println!("DEBUG: ON path (expect 2 allocs)");
  let t = txn();
  flag.propose(t, |_| true);
  dx1.eval(t);
  dx2.eval(t);
  println!("DEBUG: OFF path (expect 1 alloc)");
  let t2 = txn();
  flag.propose(t2, |_| false);
  dx1.eval(t2);
  dx2.eval(t2);
}

#[test]
#[should_panic]
fn test_gpu_adj_fail() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let mut sink = Sink::from(x.clone());
  let dx = x.adjoint(&mut sink);
}
