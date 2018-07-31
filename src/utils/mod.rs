/*
Copyright 2017-2018 Peter Jin

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

use ::{Txn};

use arrayidx::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal};

use std::f32;
use std::rc::{Rc};

pub mod nn;

pub fn _arg_max(xs: &[f32]) -> usize {
  let mut top_x = f32::NEG_INFINITY;
  let mut top_k = 0;
  for (k, &x) in xs.iter().enumerate() {
    if x > top_x {
      top_x = x;
      top_k = k;
    }
  }
  top_k
}

pub struct PiecewiseSeries<T> {
  init_val: T,
  pieces:   Vec<(usize, T)>,
}

impl<T> PiecewiseSeries<T> {
  pub fn new(init_val: T, pieces: Vec<(usize, T)>) -> Self {
    PiecewiseSeries{init_val, pieces}
  }
}

impl<T: Clone> PiecewiseSeries<T> {
  pub fn at(&self, t: usize) -> T {
    for (p, &(start, ref val)) in self.pieces.iter().enumerate() {
      if t < start {
        match p {
          0 => return self.init_val.clone(),
          _ => return self.pieces[p - 1].1.clone(),
        }
      }
    }
    self.pieces[self.pieces.len() - 1].1.clone()
  }
}

pub trait ZerosInit<Shape> {
  type RValue;

  fn zeros_init(shape: Shape) -> Self::RValue;
}

impl ZerosInit<usize> for MemArray1d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_init(shape: usize) -> Self::RValue {
    Rc::new(move || {
      MemArray1d::<f32>::zeros(shape)
    })
  }
}

impl ZerosInit<[usize; 2]> for MemArray2d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_init(shape: [usize; 2]) -> Self::RValue {
    Rc::new(move || {
      MemArray2d::<f32>::zeros(shape)
    })
  }
}

impl ZerosInit<[usize; 4]> for MemArray4d<f32> {
  type RValue = Rc<Fn() -> Self>;

  fn zeros_init(shape: [usize; 4]) -> Self::RValue {
    Rc::new(move || {
      MemArray4d::<f32>::zeros(shape)
    })
  }
}

impl ZerosInit<()> for GPUDeviceScalar<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(_shape: ()) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceScalar::<f32>::zeros((), conn.clone())
    })
  }
}

impl ZerosInit<usize> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: usize) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceArray1d::<f32>::zeros(shape, conn.clone())
    })
  }
}

impl ZerosInit<[usize; 2]> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: [usize; 2]) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceArray2d::<f32>::zeros(shape, conn.clone())
    })
  }
}

impl ZerosInit<[usize; 4]> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: [usize; 4]) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceArray4d::<f32>::zeros(shape, conn.clone())
    })
  }
}

impl ZerosInit<usize> for GPUDeviceOuterBatchScalar<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: usize) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchScalar::<f32>::zeros((), shape, conn.clone())
    })
  }
}

impl ZerosInit<usize> for GPUDeviceOuterBatchScalar<u32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: usize) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchScalar::<u32>::zeros((), shape, conn.clone())
    })
  }
}

impl ZerosInit<(usize, usize)> for GPUDeviceOuterBatchArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: (usize, usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray1d::<f32>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

impl ZerosInit<([usize; 2], usize)> for GPUDeviceOuterBatchArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: ([usize; 2], usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray2d::<f32>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

impl ZerosInit<([usize; 2], usize)> for GPUDeviceOuterBatchArray2d<u32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: ([usize; 2], usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray2d::<u32>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

impl ZerosInit<([usize; 3], usize)> for GPUDeviceOuterBatchArray3d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: ([usize; 3], usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray3d::<f32>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

impl ZerosInit<([usize; 3], usize)> for GPUDeviceOuterBatchArray3d<u8> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: ([usize; 3], usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray3d::<u8>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

impl<T: ZeroBits + 'static> ZerosInit<([usize; 4], usize)> for GPUDeviceOuterBatchArray4d<T> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn zeros_init(shape: ([usize; 4], usize)) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      GPUDeviceOuterBatchArray4d::<T>::zeros(shape.0, shape.1, conn.clone())
    })
  }
}

pub trait UniformInit<Shape, T, R: Rng> {
  type RValue;

  fn uniform_init(shape: Shape, lo: T, hi: T, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> UniformInit<usize, f32, R> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_init(shape: usize, lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
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

impl<R: Rng> UniformInit<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_init(shape: [usize; 2], lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
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

impl<R: Rng> UniformInit<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_init(shape: [usize; 4], lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
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

impl<R: Rng> UniformInit<(usize, usize), f32, R> for GPUDeviceOuterBatchArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_init(shape: (usize, usize), lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
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

impl<R: Rng> UniformInit<([usize; 3], usize), f32, R> for GPUDeviceOuterBatchArray3d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn uniform_init(shape: ([usize; 3], usize), lo: f32, hi: f32, seed_rng: &mut R) -> Self::RValue {
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

pub trait NormalInit<Shape, T, R: Rng> {
  type RValue;

  fn normal_init(shape: Shape, mean: T, std: T, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> NormalInit<usize, f32, R> for GPUDeviceArray1d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_init(shape: usize, mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
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

impl<R: Rng> NormalInit<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_init(shape: [usize; 2], mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
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

impl<R: Rng> NormalInit<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_init(shape: [usize; 4], mean: f32, std: f32, seed_rng: &mut R) -> Self::RValue {
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
