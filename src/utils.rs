use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;

use rand::*;
use rand::distributions::{Distribution, Uniform, Normal};
use std::rc::{Rc};

pub trait ZerosFill<Idx> {
  type RValue;

  fn zeros_fill(size: Idx) -> Self::RValue;
}

impl ZerosFill<usize> for MemArray1d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_fill(size: usize) -> Self::RValue {
    Rc::new(move || {
      MemArray1d::<f32>::zeros(size)
    })
  }
}

impl ZerosFill<[usize; 2]> for MemArray2d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_fill(size: [usize; 2]) -> Self::RValue {
    Rc::new(move || {
      MemArray2d::<f32>::zeros(size)
    })
  }
}

impl ZerosFill<[usize; 4]> for MemArray4d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_fill(size: [usize; 4]) -> Self::RValue {
    Rc::new(move || {
      MemArray4d::<f32>::zeros(size)
    })
  }
}

impl ZerosFill<usize> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn zeros_fill(size: usize) -> Self::RValue {
    Rc::new(move |conn: GPUDeviceConn| {
      GPUDeviceArray1d::<f32>::zeros(size, conn.clone())
    })
  }
}

impl ZerosFill<[usize; 2]> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn zeros_fill(size: [usize; 2]) -> Self::RValue {
    Rc::new(move |conn: GPUDeviceConn| {
      GPUDeviceArray2d::<f32>::zeros(size, conn.clone())
    })
  }
}

impl ZerosFill<[usize; 4]> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn zeros_fill(size: [usize; 4]) -> Self::RValue {
    Rc::new(move |conn: GPUDeviceConn| {
      GPUDeviceArray4d::<f32>::zeros(size, conn.clone())
    })
  }
}

pub trait UniformFill<Idx, T, R: Rng> {
  type RValue;

  fn uniform_fill(size: Idx, lo: T, hi: T, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> UniformFill<usize, f32, R> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn uniform_fill(size: usize, lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray1d::<f32>::zeros(size);
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let mut xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceArray1d::<f32>::zeros(size, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> UniformFill<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn uniform_fill(size: [usize; 2], lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray2d::<f32>::zeros(size);
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let mut xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(size, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> UniformFill<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn uniform_fill(size: [usize; 4], lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(size);
      {
        let dist = Uniform::new_inclusive(lo, hi);
        let mut v = h_arr.as_view_mut();
        let mut xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng());
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(size, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

pub trait NormalFill<Idx, T, R: Rng> {
  type RValue;

  fn normal_fill(size: Idx, mean: T, std: T, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> NormalFill<usize, f32, R> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn normal_fill(size: usize, mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray1d::<f32>::zeros(size);
      {
        let dist = Normal::new(mean as f64, std as f64);
        let mut v = h_arr.as_view_mut();
        let mut xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray1d::<f32>::zeros(size, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> NormalFill<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn normal_fill(size: [usize; 2], mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray2d::<f32>::zeros(size);
      {
        let dist = Normal::new(mean as f64, std as f64);
        let mut v = h_arr.as_view_mut();
        let mut xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(size, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

impl<R: Rng> NormalFill<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(GPUDeviceConn) -> Self>;

  fn normal_fill(size: [usize; 4], mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
    let seed = seed_rng.next_u64();
    Rc::new(move |conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(size);
      {
        let dist = Normal::new(mean as f64, std as f64);
        let mut v = h_arr.as_view_mut();
        let mut xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(size, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}
