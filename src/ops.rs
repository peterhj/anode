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

//use std::marker::{PhantomData};
use std::ops::{DerefMut};
use std::rc::{Rc};

pub trait MemIoReader<'a> {
  fn read_mem(&mut self, src: &'a Any) -> Option<()>;
}

pub trait MemIoWriter<'a> {
  fn write_mem(&mut self, mode: WriteMode, dst: &mut &'a mut Any) -> Option<()>;
}

impl<'a, Mem, T> MemIoReader<'a> for FlatReader<'a, Mem> where Mem: DerefMut<Target=[T]>, T: Copy + 'static {
  fn read_mem(&mut self, src: &'a Any) -> Option<()> {
    if let Some(_) = src.downcast_ref::<()>() {
      Some(())
    } else if let Some(ref src) = src.downcast_ref::<Vec<T>>() {
      let src_len = src.len();
      self.inner[self.offset .. self.offset + src_len].copy_from_slice(src);
      self.offset += src_len;
      Some(())
    } else {
      None
    }
  }
}

impl<'a, Mem, T> MemIoWriter<'a> for FlatWriter<'a, Mem> where Mem: DerefMut<Target=[T]>, T: Copy + 'static {
  fn write_mem(&mut self, mode: WriteMode, dst: &mut &'a mut Any) -> Option<()> {
    if let Some(_) = dst.downcast_ref::<()>() {
      Some(())
    } else if let Some(ref mut dst) = (*dst).downcast_mut::<Vec<T>>() {
      let dst_len = dst.len();
      dst.copy_from_slice(&self.inner[self.offset .. self.offset + dst_len]);
      self.offset += dst_len;
      Some(())
    } else {
      None
    }
  }
}

pub struct StopOp;
pub struct CopyOp;
pub struct CastOp;

pub struct PassOp;
pub struct FlatViewOp;
pub struct ReshapeViewOp;
pub struct AutoMap<AutoMapF> { f: AutoMapF, }
pub struct TransposeOp;
pub struct SumJoinOp;
pub struct SumJoinAccumulateOp;
pub struct FlatSumJoinOp;
pub struct BatchSumJoinOp;
pub struct FlatMapOp<FlatMapF> { f: FlatMapF, }
pub struct FlatLinearMapOp;
pub struct LinearMapOp;
pub struct Conv1dLinearMapOp;
pub struct Conv2dLinearMapOp;
pub struct Conv3dLinearMapOp;
pub struct Resample2dOp<ResampleF> { f: ResampleF, }

pub struct SoftmaxNLLFusedOp;
pub struct SoftmaxCrossEntropyFusedOp;
pub struct SoftmaxEntropyFusedOp;

pub struct SquareFlatMapF;
pub struct ModulusFlatMapF;
pub struct PositiveClipFlatMapF;
pub struct UnitStepFlatMapF;
pub struct LogPositiveClipFlatMapF;
pub struct PositiveReciprocalFlatMapF;
pub struct NormalCdfFlatMapF;
pub struct TanhFlatMapF;
pub struct RCosh2FlatMapF;

pub struct Conv2dShape {
  pub axes:     [usize; 2],
  pub kernel:   [usize; 2],
  pub stride:   [usize; 2],
  pub pad:      [usize; 2],
}

pub struct MaxPoolResample2dF;
pub struct AvgPoolResample2dF;
pub struct BilinearResample2dF;
pub struct BicubicResample2dF;

pub struct Pool2dShape {
  pub axes:     [usize; 2],
  pub window:   [usize; 2],
  pub stride:   [usize; 2],
  pub pad:      [usize; 2],
}

pub trait SumJoinOpExt<A, V> where V: AVal {
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<SumJoinOp, V, V>>;
}

pub trait SumExt<X, V> where V: AVal {
  fn sum(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<SumJoinOp, V, V>>;
  fn add(self, x_: Rc<AOp<V=V>>) -> Rc<JoinOp<SumJoinOp, V, V>>;
}

pub trait FlatMultOpExt<X, V1, A, V2, Y, W>
where V1: AVal,
      V2: AVal,
      W:  AVal,
{
  fn flat_mult(self, x: Rc<AOp<V=V1>>) -> Rc<F2Op<FlatLinearMapOp, V1, V2, W>>;
}

pub trait MultOpExt<X, V1, A, V2, Y, W>
where V1: AVal,
      V2: AVal,
      W:  AVal,
{
  fn mult(self, x: Rc<AOp<V=V1>>) -> Rc<F2Op<LinearMapOp, V1, V2, W>>;
}

pub trait MultAddOpExt<X, V1, A, V2, B, V3, Y, W>
where V1: AVal,
      V2: AVal,
      V3: AVal,
      W:  AVal,
{
  fn mult_add(self, x: Rc<AOp<V=V1>>, shift: Rc<AOp<V=V3>>) -> Rc<F3Op<LinearMapOp, V1, V2, V3, W>>;
}
