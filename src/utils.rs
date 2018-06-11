use ::{Txn};

use arrayidx::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;

use rand::*;
use rand::distributions::{Distribution, Uniform, Normal};
use std::rc::{Rc};

pub trait ZerosFill<Shape> {
  type RValue;

  fn zeros_fill(shape: Shape) -> Self::RValue;
}

impl ZerosFill<usize> for MemArray1d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_fill(shape: usize) -> Self::RValue {
    Rc::new(move || {
      MemArray1d::<f32>::zeros(shape)
    })
  }
}

impl ZerosFill<[usize; 2]> for MemArray2d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_fill(shape: [usize; 2]) -> Self::RValue {
    Rc::new(move || {
      MemArray2d::<f32>::zeros(shape)
    })
  }
}

impl ZerosFill<[usize; 4]> for MemArray4d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_fill(shape: [usize; 4]) -> Self::RValue {
    Rc::new(move || {
      MemArray4d::<f32>::zeros(shape)
    })
  }
}

impl ZerosFill<usize> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_fill(shape: usize) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceArray1d::<f32>::zeros(shape, conn.clone())
    })
  }
}

impl ZerosFill<[usize; 2]> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_fill(shape: [usize; 2]) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceArray2d::<f32>::zeros(shape, conn.clone())
    })
  }
}

impl ZerosFill<[usize; 4]> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_fill(shape: [usize; 4]) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceArray4d::<f32>::zeros(shape, conn.clone())
    })
  }
}

impl ZerosFill<usize> for GPUDeviceOuterBatchScalar<u32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_fill(shape: usize) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchScalar::<u32>::zeros((), shape, conn.clone())
    })
  }
}

impl ZerosFill<(usize, usize)> for GPUDeviceOuterBatchArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_fill(shape: (usize, usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray1d::<f32>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

impl ZerosFill<([usize; 3], usize)> for GPUDeviceOuterBatchArray3d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_fill(shape: ([usize; 3], usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray3d::<f32>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

pub trait UniformFill<Shape, T, R: Rng> {
  type RValue;

  fn uniform_fill(shape: Shape, lo: T, hi: T, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> UniformFill<usize, f32, R> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_fill(shape: usize, lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray1d::<f32>::zeros(shape);
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceArray1d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> UniformFill<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_fill(shape: [usize; 2], lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray2d::<f32>::zeros(shape);
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> UniformFill<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_fill(shape: [usize; 4], lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape);
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> UniformFill<(usize, usize), f32, R> for GPUDeviceOuterBatchArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_fill(shape: (usize, usize), lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray2d::<f32>::zeros(shape.0.index_append(shape.1));
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceOuterBatchArray1d::<f32>::zeros(shape.0, shape.1, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> UniformFill<([usize; 3], usize), f32, R> for GPUDeviceOuterBatchArray3d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_fill(shape: ([usize; 3], usize), lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape.0.index_append(shape.1));
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceOuterBatchArray3d::<f32>::zeros(shape.0, shape.1, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

pub trait NormalFill<Shape, T, R: Rng> {
  type RValue;

  fn normal_fill(shape: Shape, mean: T, std: T, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> NormalFill<usize, f32, R> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_fill(shape: usize, mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray1d::<f32>::zeros(shape);
      {
        let dist = Normal::new(mean as f64, std as f64);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray1d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> NormalFill<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_fill(shape: [usize; 2], mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray2d::<f32>::zeros(shape);
      {
        let dist = Normal::new(mean as f64, std as f64);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> NormalFill<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_fill(shape: [usize; 4], mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape);
      {
        let dist = Normal::new(mean as f64, std as f64);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}
