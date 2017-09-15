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

use super::*;
use ops::*;

use arithmetic::*;
use cuda_blas::new::*;
use devicemem_cuda::v2::*;

use std::marker::{PhantomData};
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::ptr::{Shared};
use std::sync::{Arc};

#[inline]
pub fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

#[repr(C)]
struct KernelConfig {
  pub block_sz:     u32,
  pub max_blocks:   u32,
}

pub type Index0d = ();
pub type Index1d = usize;
pub type Index2d = [usize; 2];
pub type Index3d = [usize; 3];
pub type Index4d = [usize; 4];
pub type Index5d = [usize; 5];
pub struct UnimplIndex;

pub type Range0d = ();
pub type Range1d = Range<usize>;
pub type Range2d = [Range<usize>; 2];
pub type Range3d = [Range<usize>; 3];
pub type Range4d = [Range<usize>; 4];
pub type Range5d = [Range<usize>; 5];

pub type RangeFrom0d = ();
pub type RangeFrom1d = RangeFrom<usize>;
pub type RangeFrom2d = [RangeFrom<usize>; 2];
pub type RangeFrom3d = [RangeFrom<usize>; 3];
pub type RangeFrom4d = [RangeFrom<usize>; 4];
pub type RangeFrom5d = [RangeFrom<usize>; 5];

pub type RangeTo0d = ();
pub type RangeTo1d = RangeTo<usize>;
pub type RangeTo2d = [RangeTo<usize>; 2];
pub type RangeTo3d = [RangeTo<usize>; 3];
pub type RangeTo4d = [RangeTo<usize>; 4];
pub type RangeTo5d = [RangeTo<usize>; 5];

pub type RangeFull0d = ();
pub type RangeFull1d = RangeFull;
pub type RangeFull2d = [RangeFull; 2];
pub type RangeFull3d = [RangeFull; 3];
pub type RangeFull4d = [RangeFull; 4];
pub type RangeFull5d = [RangeFull; 5];

/*#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Stride<Idx>(Idx);

impl<Idx> Stride<Idx> where Idx: Copy {
  pub fn unwrap(&self) -> Idx {
    self.0
  }
}

impl<Idx> Stride<Idx> where Idx: ArrayShape + Copy {
  pub fn append_packed(&self, dim: Idx) -> Stride<Idx::Above> {
    Stride(self.0.append(self.0.outside() * dim.outside()))
  }
}*/

pub trait ArrayShape {
  type Range;
  //type RangeFull;
  type Above: Sized;

  fn zero() -> Self where Self: Sized;

  fn add(&self, shift: &Self) -> Self where Self: Sized;
  fn sub(&self, shift: &Self) -> Self where Self: Sized;

  fn to_packed_stride(&self) -> Self where Self: Sized;
  fn is_packed(&self, stride: &Self) -> bool where Self: Sized;

  fn prepend(&self, new_inside: usize) -> Self::Above;
  fn append(&self, new_outside: usize) -> Self::Above;

  fn packed_stride_append(&self, dim: &Self) -> Self::Above where Self: Sized {
    self.append(self.outside() * dim.outside())
  }

  fn flat_len(&self) -> usize;
  fn flat_offset(&self, stride: &Self) -> isize;

  fn inside(&self) -> usize;
  fn outside(&self) -> usize;
}

pub trait ArrayRange<Idx> {
  fn start(&self, offset: &Idx) -> Idx;
  fn end(&self, limit: &Idx) -> Idx;
}

impl ArrayShape for Index0d {
  type Range = Range0d;
  type Above = Index1d;

  fn zero() -> Self {
    ()
  }

  fn add(&self, shift: &Self) -> Self {
    ()
  }

  fn sub(&self, shift: &Self) -> Self {
    ()
  }

  fn to_packed_stride(&self) -> Self {
    ()
  }

  fn is_packed(&self, stride: &Self) -> bool {
    true
  }

  fn prepend(&self, major: usize) -> Index1d {
    major
  }

  fn append(&self, minor: usize) -> Index1d {
    minor
  }

  fn flat_len(&self) -> usize {
    1
  }

  fn flat_offset(&self, stride: &Self) -> isize {
    0
  }

  fn inside(&self) -> usize {
    1
  }

  fn outside(&self) -> usize {
    1
  }
}

impl ArrayShape for Index1d {
  type Range = Range1d;
  type Above = Index2d;

  fn zero() -> Self {
    1
  }

  fn add(&self, shift: &Self) -> Self {
    *self + *shift
  }

  fn sub(&self, shift: &Self) -> Self {
    *self - *shift
  }

  fn to_packed_stride(&self) -> Self {
    1
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index2d {
    [major, *self]
  }

  fn append(&self, minor: usize) -> Index2d {
    [*self, minor]
  }

  fn flat_len(&self) -> usize {
    *self
  }

  fn flat_offset(&self, stride: &Self) -> isize {
    (*self * *stride) as _
  }

  fn inside(&self) -> usize {
    *self
  }

  fn outside(&self) -> usize {
    *self
  }
}

impl ArrayShape for Index2d {
  type Range = Range2d;
  type Above = Index3d;

  fn zero() -> Self {
    [0, 0]
  }

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index3d {
    [major, self[0], self[1]]
  }

  fn append(&self, minor: usize) -> Index3d {
    [self[0], self[1], minor]
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1]
  }

  fn flat_offset(&self, stride: &Self) -> isize {
    ( self[0] * stride[0] +
      self[1] * stride[1] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[1]
  }
}

impl ArrayShape for Index3d {
  type Range = Range3d;
  type Above = Index4d;

  fn zero() -> Self {
    [0, 0, 0]
  }

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1],
      self[2] + shift[2], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1],
      self[2] - shift[2], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s[2] = s[1] * self[1];
    s
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index4d {
    [major, self[0], self[1], self[2]]
  }

  fn append(&self, minor: usize) -> Index4d {
    [self[0], self[1], self[2], minor]
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1] * self[2]
  }

  fn flat_offset(&self, stride: &Self) -> isize {
    ( self[0] * stride[0] +
      self[1] * stride[1] +
      self[2] * stride[2] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[2]
  }
}

impl ArrayShape for Index4d {
  type Range = Range4d;
  type Above = Index5d;

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1],
      self[2] + shift[2],
      self[3] + shift[3], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1],
      self[2] - shift[2],
      self[3] - shift[3], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0, 0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s[2] = s[1] * self[1];
    s[3] = s[2] * self[2];
    s
  }

  fn zero() -> Self {
    [0, 0, 0, 0]
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index5d {
    [major, self[0], self[1], self[2], self[3]]
  }

  fn append(&self, minor: usize) -> Index5d {
    [self[0], self[1], self[2], self[3], minor]
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1] * self[2] * self[3]
  }

  fn flat_offset(&self, stride: &Self) -> isize {
    ( self[0] * stride[0] +
      self[1] * stride[1] +
      self[2] * stride[2] +
      self[3] * stride[3] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[3]
  }
}

impl ArrayShape for Index5d {
  type Range = Range5d;
  type Above = UnimplIndex;

  fn zero() -> Self {
    [0, 0, 0, 0, 0]
  }

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1],
      self[2] + shift[2],
      self[3] + shift[3],
      self[4] + shift[4], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1],
      self[2] - shift[2],
      self[3] - shift[3],
      self[4] - shift[4], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0, 0, 0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s[2] = s[1] * self[1];
    s[3] = s[2] * self[2];
    s[4] = s[3] * self[3];
    s
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> UnimplIndex {
    unimplemented!();
  }

  fn append(&self, minor: usize) -> UnimplIndex {
    unimplemented!();
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1] * self[2] * self[3] * self[4]
  }

  fn flat_offset(&self, stride: &Self) -> isize {
    ( self[0] * stride[0] +
      self[1] * stride[1] +
      self[2] * stride[2] +
      self[3] * stride[3] +
      self[4] * stride[4] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[4]
  }
}

pub struct DeviceArray<Idx, T> where T: Copy {
  dim:      Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<DeviceMem<T>>,
}

pub trait Array {
  type Idx;

  fn dim(&self) -> Self::Idx;
}

pub struct BatchWrap<T>(pub T);

pub trait BatchArray: Array {
  fn batch_size(&self) -> usize;
  fn set_batch_size(&mut self, new_batch_sz: usize);
}

pub trait DeviceArrayZeros: Array {
  fn zeros(dim: Self::Idx, conn: &DeviceConn) -> Self where Self: Sized;
  //fn zeros_with_offset_stride(dim: Self::Idx, offset: Self::Idx, stride: Self::Idx, conn: &DeviceConn) -> Self where Self: Sized;
}

pub trait DeviceBatchArrayZeros: BatchArray {
  fn zeros(dim: Self::Idx, batch_sz: usize, conn: &DeviceConn) -> Self where Self: Sized;
  //fn zeros_with_offset_stride(dim: Self::Idx, offset: Self::Idx, stride: Self::Idx, batch_sz: usize, conn: &DeviceConn) -> Self where Self: Sized;
}

pub trait AsView {
  type ViewTy;

  fn as_view(&self) -> Self::ViewTy;
}

/*pub trait AsViewMut: AsView {
  type ViewMutTy;

  fn as_view_mut(&self) -> Self::ViewMutTy;
}*/

pub trait FlatView {
  type FlatViewTy;

  fn flat_view(&self) -> Option<Self::FlatViewTy>;
}

/*pub trait FlatViewMut: FlatView {
  type FlatViewMutTy;

  fn flat_view_mut(&self) -> Option<Self::FlatViewMutTy>;
}*/

pub trait View<Idx> {
  fn view(self, idx: Idx) -> Self where Self: Sized;
}

pub type DeviceScalar<T>  = DeviceArray<Index0d, T>;
pub type DeviceArray1d<T> = DeviceArray<Index1d, T>;
pub type DeviceArray2d<T> = DeviceArray<Index2d, T>;
pub type DeviceArray3d<T> = DeviceArray<Index3d, T>;
pub type DeviceArray4d<T> = DeviceArray<Index4d, T>;
pub type DeviceArray5d<T> = DeviceArray<Index5d, T>;

impl<Idx, T> DeviceArrayZeros for DeviceArray<Idx, T> where Idx: Copy, T: Copy {
  fn zeros(dim: Idx, conn: &DeviceConn) -> Self {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> Array for DeviceArray<Idx, T> where Idx: Copy, T: Copy {
  type Idx = Idx;

  fn dim(&self) -> Idx {
    self.dim
  }
}

impl<Idx, T> AsView for DeviceArray<Idx, T> where Idx: Copy, T: Copy {
  type ViewTy = DeviceArrayView<Idx, T>;

  fn as_view(&self) -> DeviceArrayView<Idx, T> {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> FlatView for DeviceArray<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  type FlatViewTy = DeviceArrayView1d<T>;

  fn flat_view(&self) -> Option<DeviceArrayView1d<T>> {
    if !self.dim.is_packed(&self.stride) {
      None
    } else {
      let flat_dim = self.dim.flat_len();
      Some(DeviceArrayView{
        dim:    flat_dim,
        offset: 0,
        stride: flat_dim.to_packed_stride(),
        mem:    self.mem.clone(),
      })
    }
  }
}

pub struct DeviceInnerBatchArray<Idx, T> where T: Copy {
  dim:          Idx,
  offset:       Idx,
  stride:       Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:          Arc<DeviceMem<T>>,
}

pub type DeviceInnerBatchScalar<T>  = DeviceInnerBatchArray<Index0d, T>;
pub type DeviceInnerBatchArray1d<T> = DeviceInnerBatchArray<Index1d, T>;
pub type DeviceInnerBatchArray2d<T> = DeviceInnerBatchArray<Index2d, T>;
pub type DeviceInnerBatchArray3d<T> = DeviceInnerBatchArray<Index3d, T>;
pub type DeviceInnerBatchArray4d<T> = DeviceInnerBatchArray<Index4d, T>;

impl<Idx, T> Array for DeviceInnerBatchArray<Idx, T> where Idx: Copy, T: Copy {
  type Idx = Idx;

  fn dim(&self) -> Idx {
    self.dim
  }
}

impl<Idx, T> BatchArray for DeviceInnerBatchArray<Idx, T> where Idx: Copy, T: Copy {
  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for DeviceInnerBatchArray<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  type ViewTy = DeviceArrayView<Idx::Above, T>;

  fn as_view(&self) -> DeviceArrayView<Idx::Above, T> {
    let view_dim = self.dim.prepend(self.batch_sz);
    // TODO
    unimplemented!();
  }
}

pub struct DeviceOuterBatchArray<Idx, T> where T: Copy {
  dim:          Idx,
  offset:       Idx,
  stride:       Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:          Arc<DeviceMem<T>>,
}

pub type DeviceOuterBatchScalar<T>  = DeviceOuterBatchArray<Index0d, T>;
pub type DeviceOuterBatchArray1d<T> = DeviceOuterBatchArray<Index1d, T>;
pub type DeviceOuterBatchArray2d<T> = DeviceOuterBatchArray<Index2d, T>;
pub type DeviceOuterBatchArray3d<T> = DeviceOuterBatchArray<Index3d, T>;
pub type DeviceOuterBatchArray4d<T> = DeviceOuterBatchArray<Index4d, T>;

impl<Idx, T> DeviceBatchArrayZeros for DeviceOuterBatchArray<Idx, T> where Idx: Copy, T: Copy {
  fn zeros(dim: Idx, batch_sz: usize, conn: &DeviceConn) -> Self {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> Array for DeviceOuterBatchArray<Idx, T> where Idx: Copy, T: Copy {
  type Idx = Idx;

  fn dim(&self) -> Idx {
    self.dim
  }
}

impl<Idx, T> BatchArray for DeviceOuterBatchArray<Idx, T> where Idx: Copy, T: Copy {
  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for DeviceOuterBatchArray<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  type ViewTy = DeviceArrayView<Idx::Above, T>;

  fn as_view(&self) -> DeviceArrayView<Idx::Above, T> {
    let view_dim = self.dim.append(self.batch_sz);
    let view_offset = self.offset.append(0);
    let view_stride = self.stride.packed_stride_append(&self.dim);
    DeviceArrayView{
      dim:      view_dim,
      offset:   view_offset,
      stride:   view_stride,
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> FlatView for DeviceOuterBatchArray<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  type FlatViewTy = DeviceArrayView1d<T>;

  fn flat_view(&self) -> Option<DeviceArrayView1d<T>> {
    if !self.dim.is_packed(&self.stride) {
      None
    } else {
      let flat_dim = self.dim.flat_len() * self.batch_sz;
      Some(DeviceArrayView{
        dim:    flat_dim,
        offset: 0,
        stride: flat_dim.to_packed_stride(),
        mem:    self.mem.clone(),
      })
    }
  }
}

#[derive(Clone)]
pub struct DeviceArrayView<Idx, T> where T: Copy {
  dim:      Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<DeviceMem<T>>,
}

pub type DeviceScalarView<T>  = DeviceArrayView<Index0d, T>;
pub type DeviceArrayView1d<T> = DeviceArrayView<Index1d, T>;
pub type DeviceArrayView2d<T> = DeviceArrayView<Index2d, T>;
pub type DeviceArrayView3d<T> = DeviceArrayView<Index3d, T>;
pub type DeviceArrayView4d<T> = DeviceArrayView<Index4d, T>;
pub type DeviceArrayView5d<T> = DeviceArrayView<Index5d, T>;

/*pub struct DeviceArrayViewMut<Idx, T> where T: Copy {
  dim:      Idx,
  stride:   Idx,
  mem:      Arc<DeviceMem<T>>,
}

pub type DeviceScalarViewMut<T>  = DeviceArrayViewMut<Index0d, T>;
pub type DeviceArrayViewMut1d<T> = DeviceArrayViewMut<Index1d, T>;
pub type DeviceArrayViewMut2d<T> = DeviceArrayViewMut<Index2d, T>;
pub type DeviceArrayViewMut3d<T> = DeviceArrayViewMut<Index3d, T>;
pub type DeviceArrayViewMut4d<T> = DeviceArrayViewMut<Index4d, T>;*/

impl<Idx, T> DeviceArrayView<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  pub unsafe fn as_dptr(&self) -> *const T {
    self.mem.as_dptr().offset(self.offset.flat_offset(&self.stride))
  }

  pub unsafe fn as_mut_dptr(&self) -> *mut T {
    self.mem.as_mut_dptr().offset(self.offset.flat_offset(&self.stride))
  }

  pub fn stride(&self) -> Idx {
    self.stride
  }
}

impl<Idx, T> Array for DeviceArrayView<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  type Idx = Idx;

  fn dim(&self) -> Idx {
    self.dim
  }
}

impl<Idx, T> DeviceArrayView<Idx, T> where Idx: ArrayShape + Copy, T: Copy {
  pub fn copy(&mut self, other: &DeviceArrayView<Idx, T>, conn: &DeviceConn) {
    // TODO
    unimplemented!();
  }

  pub fn add(&mut self, other: &DeviceArrayView<Idx, T>, conn: &DeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<Idx, Range, T> View<Range> for DeviceArrayView<Idx, T> where Idx: ArrayShape + Copy, Range: ArrayRange<Idx> + Copy, T: Copy {
  fn view(self, range: Range) -> Self {
    // TODO: bounds check.
    let view_dim = range.end(&self.dim).sub(&range.start(&Idx::zero()));
    let view_offset = self.offset.add(&range.start(&Idx::zero()));
    let view_stride = self.stride;
    DeviceArrayView{
      dim:      view_dim,
      offset:   view_offset,
      stride:   view_stride,
      mem:      self.mem.clone(),
    }
  }
}

pub trait DeviceMemIoReader<'a> {
  fn read_dev_mem(&mut self, src: &'a Any) -> Option<()>;
}

pub trait DeviceMemIoWriter<'a> {
  fn write_dev_mem(&mut self, mode: WriteMode, dst: &'a mut Any) -> Option<()>;
}

impl SumJoinOp {
  pub fn build_device_op<T, A, V>(inputs_: Vec<Rc<AOp<V=V>>>)
      -> Rc<JoinOp<Self, V, V>>
  where T: Copy + PseudoField,
        A: DeviceArrayZeros + FlatView<FlatViewTy=DeviceArrayView1d<T>>,
        V: RWVal<T=A> + 'static,
  {
    let make = {
      let inputs_ = inputs_.clone();
      Rc::new(move || {
        let x0 = inputs_[0].value();
        <V as RWVal>::from(Rc::new(move |txn: Txn| {
          let x0_dim = x0.get(txn).dim();
          let pool = DeviceStreamPool::implicit();
          let conn = pool.conn();
          A::zeros(x0_dim, &conn)
        }))
      })
    };
    let ext = OpExt{
      make,
      func: {
        let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };
            let pool = DeviceStreamPool::implicit();
            let conn = pool.conn();
            let mut y = match output.get_mut(txn, token).flat_view() {
              None => panic!(),
              Some(y) => y,
            };
            let x0 = match inputs[0].get(txn).flat_view() {
              None => panic!(),
              Some(x) => x,
            };
            if overwrite {
              y.copy(&x0, &conn);
            } else {
              y.add(&x0, &conn);
            }
            for i in 1 .. inputs.len() {
              let x = match inputs[i].get(txn).flat_view() {
                None => panic!(),
                Some(x) => x,
              };
              y.add(&x, &conn);
            }
          }
        })
      },
      tangent:  None,
      adjoint:  None,
      adjoint2: None,
    };
    let output = (ext.make)();
    Rc::new(JoinOp::new(SumJoinOp, ext, inputs_, output))
  }

  pub fn build_device_batch_op<T, A, V>(inputs_: Vec<Rc<AOp<V=V>>>)
      -> Rc<JoinOp<Self, V, V>>
  where T: Copy + PseudoField,
        A: DeviceBatchArrayZeros + FlatView<FlatViewTy=DeviceArrayView1d<T>>,
        V: RWVal<T=A> + 'static,
  {
    let make = {
      let inputs_ = inputs_.clone();
      Rc::new(move || {
        let x0 = inputs_[0].value();
        <V as RWVal>::from(Rc::new(move |txn: Txn| {
          let x0_dim = x0.get(txn).dim();
          let x0_batch_sz = x0.get(txn).batch_size();
          let pool = DeviceStreamPool::implicit();
          let conn = pool.conn();
          A::zeros(x0_dim, x0_batch_sz, &conn)
        }))
      })
    };
    let ext = OpExt{
      make,
      func: {
        let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };
            let pool = DeviceStreamPool::implicit();
            let conn = pool.conn();
            let mut y = match output.get_mut(txn, token).flat_view() {
              None => panic!(),
              Some(y) => y,
            };
            let batch_sz0 = inputs[0].get(txn).batch_size();
            let x0 = match inputs[0].get(txn).flat_view() {
              None => panic!(),
              Some(x) => x,
            };
            output.get_mut(txn, token).set_batch_size(batch_sz0);
            if overwrite {
              y.copy(&x0, &conn);
            } else {
              y.add(&x0, &conn);
            }
            for i in 1 .. inputs.len() {
              let batch_sz = inputs[i].get(txn).batch_size();
              assert_eq!(batch_sz, batch_sz0);
              let x = match inputs[i].get(txn).flat_view() {
                None => panic!(),
                Some(x) => x,
              };
              y.add(&x, &conn);
            }
          }
        })
      },
      tangent:  None,
      adjoint:  None,
      adjoint2: None,
    };
    let output = (ext.make)();
    Rc::new(JoinOp::new(SumJoinOp, ext, inputs_, output))
  }
}

impl<T, V> SumJoinOpExt<DeviceArray1d<T>, V> for SumJoinOp
where T: Copy + PseudoField,
      V: RWVal<T=DeviceArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<Self, V, V>> {
    Self::build_device_op::<T, DeviceArray1d<T>, V>(xs_)
  }
}

impl<T, V> SumJoinOpExt<DeviceOuterBatchArray1d<T>, V> for SumJoinOp
where T: Copy + PseudoField,
      V: RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<Self, V, V>> {
    Self::build_device_batch_op::<T, DeviceOuterBatchArray1d<T>, V>(xs_)
  }
}

impl<A, V> SumExt<A, V> for Rc<AOp<V=V>>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V=V>>) -> Rc<JoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}

impl<A, V, This> SumExt<A, V> for Rc<This>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
      This: AOp<V=V> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V=V>>) -> Rc<JoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}

impl LinearMapOp {
  pub fn build_device_op<T, V1, V2, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>
  where T: Copy + PseudoField,
        V1: RWVal<T=DeviceArray1d<T>> + 'static,
        V2: RWVal<T=DeviceArray2d<T>> + 'static,
        W:  RWVal<T=DeviceArray1d<T>> + 'static,
        CublasHandle: CublasExt<T>,
  {
    let make = {
      let map_ = map_.clone();
      Rc::new(move || {
        let map = map_.value();
        <W as RWVal>::from(Rc::new(move |txn| {
          let a_dim = map.get(txn).dim();
          let pool = DeviceStreamPool::implicit();
          let conn = pool.conn();
          DeviceArray1d::zeros(a_dim[0], &conn)
        }))
      })
    };
    let ext = OpExt{
      make,
      func: {
        let input = input_.value();
        let map = map_.value();
        Rc::new(move |txn, output: W| {
          if let Some((mode, token)) = output.write(txn) {
            let alpha = T::one();
            let beta = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => T::one(),
              (_, true) => T::zero(),
              _ => unreachable!(),
            };
            let pool = DeviceStreamPool::implicit();
            let conn = pool.conn();
            assert_eq!(input.get(txn).dim(), map.get(txn).dim()[1]);
            assert_eq!(output.get_mut(txn, token).dim(), map.get(txn).dim()[0]);
            assert_eq!(1, map.get(txn).as_view().stride()[0]);
            let res = unsafe { conn.cublas().gemv(
                CublasTranspose::N,
                sz2int(map.get(txn).as_view().dim()[0]),
                sz2int(map.get(txn).as_view().dim()[1]),
                &alpha,
                map.get(txn).as_view().as_dptr(),
                sz2int(map.get(txn).as_view().stride()[1]),
                input.get(txn).as_view().as_dptr(),
                sz2int(input.get(txn).as_view().stride()),
                &beta,
                output.get_mut(txn, token).as_view().as_mut_dptr(),
                sz2int(output.get_mut(txn, token).as_view().stride()),
            ) };
            if res.is_err() {
              panic!("LinearMapOp: cublas gemv error: {:?}", res);
            }
          }
        })
      },
      tangent: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Rc::new(move || {
          let input_ = input_.clone();
          let map_ = map_.clone();
          let tng_input_ = input_.tangent().1;
          let tng_map_ = map_.tangent().1;
          let y_ = map_.mult(tng_input_).add(tng_map_.mult(input_));
          (y_.clone(), y_)
        })
      }),
      adjoint: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Rc::new(move |y_: Rc<AOp<V=W>>, sink: &mut Sink| {
          //let make = make.clone();
          let input_ = input_.clone();
          let map_ = map_.clone();
          if let Some(adj_y_) = sink.get_adj::<W>(y_.var()) {
            // TODO
            unimplemented!();
            //let adj_a_ = adj_y_.mult_right_transpose(input_);
            //let adj_x_ = map_.mult_left_transpose(adj_y_);
            //sink.put_adj::<V2, _>(map_.var(), adj_a_);
            //sink.put_adj::<V1, _>(input_.var(), adj_x_);
          }
        })
      }),
      adjoint2: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Rc::new(move |y_: Rc<AOp<V=W>>, sink: &Sink, sink2: &mut Sink| {
          //let make = make.clone();
          let input_ = input_.clone();
          let map_ = map_.clone();
          if let Some(adj_y_) = sink2.get_adj::<W>(y_.var()) {
            // TODO
            unimplemented!();
            //let adj_a_ = adj_y_.mult_right_transpose(input_.square());
            //let adj_x_ = map_.square().mult_left_transpose(adj_y_);
            //sink2.put_adj::<V2, _>(map_.var(), adj_a_);
            //sink2.put_adj::<V1, _>(input_.var(), adj_x_);
          }
        })
      }),
    };
    let output = (ext.make)();
    Rc::new(F2Op::new(LinearMapOp, ext, input_, map_, output))
  }

  pub fn build_device_obatch_op<T, V1, V2, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>
  where T: Copy,
        V1: RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=DeviceArray2d<T>> + 'static,
        W:  RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }

  pub fn build_device_batch_affine_op<T, V1, V2, V3, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>, bias_: Rc<AOp<V=V3>>)
      -> Rc<F3Op<Self, V1, V2, V3, W>>
  where T: Copy,
        V1: RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=DeviceArray2d<T>> + 'static,
        V3: RWVal<T=DeviceArray1d<T>> + 'static,
        W:  RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }
}

impl<T, V1, V2, W> MultOpExt<DeviceArray1d<T>, V1, DeviceArray2d<T>, V2, DeviceArray1d<T>, W> for Rc<AOp<V=V2>>
where T: Copy + PseudoField,
      V1: RWVal<T=DeviceArray1d<T>> + 'static,
      V2: RWVal<T=DeviceArray2d<T>> + 'static,
      W:  RWVal<T=DeviceArray1d<T>> + 'static,
      CublasHandle: CublasExt<T>,
{
  fn mult(self, x: Rc<AOp<V=V1>>) -> Rc<F2Op<LinearMapOp, V1, V2, W>> {
    LinearMapOp::build_device_op(x, self)
  }
}

/*impl<T, V1, V2, W> MultOpExt<DeviceOuterBatchArray1d<T>, V1, DeviceOuterBatchArray1d<T>, W> for Rc<AOp<V=V2>>
where T: Copy,
      V1: RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=DeviceArray2d<T>> + 'static,
      W:  RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
{
  fn mult(self, x: Rc<AOp<V=V1>>) -> Rc<AOp<V=W>> {
    LinearMapOp::build_device_obatch_op(x, self)
  }
}*/

impl<T, V1, V2, V3, W> MultAddOpExt<DeviceOuterBatchArray1d<T>, V1, DeviceArray2d<T>, V2, DeviceArray1d<T>, V3, DeviceOuterBatchArray1d<T>, W> for Rc<AOp<V=V2>>
where T: Copy,
      V1: RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=DeviceArray2d<T>> + 'static,
      V3: RWVal<T=DeviceArray1d<T>> + 'static,
      W:  RWVal<T=DeviceOuterBatchArray1d<T>> + 'static,
{
  fn mult_add(self, x: Rc<AOp<V=V1>>, shift: Rc<AOp<V=V3>>) -> Rc<F3Op<LinearMapOp, V1, V2, V3, W>> {
    LinearMapOp::build_device_batch_affine_op(x, self, shift)
  }
}
