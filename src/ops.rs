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

use ::*;

use std::marker::{PhantomData};
use std::ops::{Add, Mul};
use std::rc::{Rc};

/*pub trait MemIoReader<'a> {
  fn read_mem(&mut self, src: &'a Any) -> Option<()>;
}

pub trait MemIoWriter<'a> {
  fn write_mem(&mut self, cap: WriteCap, dst: &mut &'a mut Any) -> Option<()>;
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
  fn write_mem(&mut self, cap: WriteCap, dst: &mut &'a mut Any) -> Option<()> {
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
}*/

pub struct PassFun;
pub struct FreezeFun;
//pub struct CopyFun;

pub struct CastFun;
pub struct DequantizeFun<T, Scale> { pub base: T, pub range: T, _marker: PhantomData<Scale> }

pub struct SrcOp;
pub struct ZerosSrcOp;
pub struct OnesSrcOp;
// TODO: distribution parameters?
pub struct RandomBitsSrcOp;
pub struct UniformSrcOp;
pub struct NormalSrcOp;

pub struct FlatViewOp;
pub struct ReshapeViewOp;
pub struct MapFun<MapF> { pub f: MapF, }
pub struct TransposeOp;
pub struct SumJoinOp;
pub struct SumJoinAccumulateOp;
pub struct FlatSumJoinOp;
pub struct BatchSumJoinOp;
pub struct FlatMapFun<FlatMapF> { pub f: FlatMapF }
pub struct FlatMapInplaceFun<FlatMapF> { pub f: FlatMapF }
pub struct FlatLinearMapFun;
pub struct InnerProductFun;
pub struct LinearMapOp;
pub struct LeftTransposeLinearMapOp;
pub struct RightTransposeLinearMapOp;
pub struct Conv1dLinearMapOp;
pub struct Conv2dLinearMapOp;
pub struct Conv3dLinearMapOp;
pub struct TransposeConv1dLinearMapOp;
pub struct TransposeConv2dLinearMapOp;
pub struct TransposeConv3dLinearMapOp;
pub struct Resample2dOp<ResampleF> { pub f: ResampleF }
pub struct ReduceFun<ReduceF> { pub f: ReduceF, /*pub axes: _*/ }
pub struct BatchNormFun { /*pub axes: _*/ }

pub struct SoftmaxNLLFusedOp;
pub struct SoftmaxCrossEntropyFusedOp;
pub struct SoftmaxEntropyFusedOp;

#[derive(Clone)] pub struct ModulusFlatMapF;
#[derive(Clone)] pub struct SquareFlatMapF;
#[derive(Clone)] pub struct PositiveClipFlatMapF;
#[derive(Clone)] pub struct UnitStepFlatMapF;
#[derive(Clone)] pub struct LogPositiveClipFlatMapF;
#[derive(Clone)] pub struct PositiveReciprocalFlatMapF;
#[derive(Clone)] pub struct NormalCdfFlatMapF;
#[derive(Clone)] pub struct TanhFlatMapF;
#[derive(Clone)] pub struct RCosh2FlatMapF;

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

#[derive(Clone)] pub struct MeanReduceF;
#[derive(Clone)] pub struct VarianceReduceF;

pub trait PassExt<V> {
  fn pass(&self) -> Rc<F1Op<PassFun, V, V>>;
}

pub trait FreezeExt<V> {
  fn freeze(&self) -> Rc<F1Op<FreezeFun, V, V>>;
}

pub struct LinearScale;

pub trait DequantizeExt<V, W, T, Scale=LinearScale> {
  fn dequantize(&self, base: T, range: T) -> Rc<F1Op<DequantizeFun<T, Scale>, V, W>>;
}

pub trait SrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub fn src<V, Init>(init: Init) -> Val<V> where SrcOp: SrcOpExt<V, Init> {
  <SrcOp as SrcOpExt<V, Init>>::build(init)
}

pub trait RandomBitsSrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub fn random_bits<V, Init>(init: Init) -> Val<V> where RandomBitsSrcOp: RandomBitsSrcOpExt<V, Init> {
  <RandomBitsSrcOp as RandomBitsSrcOpExt<V, Init>>::build(init)
}

pub trait ZerosSrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub fn zeros<V, Init>(init: Init) -> Val<V> where ZerosSrcOp: ZerosSrcOpExt<V, Init> {
  <ZerosSrcOp as ZerosSrcOpExt<V, Init>>::build(init)
}

pub trait OnesSrcOpMaybeExt<V> {
  fn maybe_build() -> Option<Val<V>>;
}

impl<V> OnesSrcOpMaybeExt<V> for OnesSrcOp {
  default fn maybe_build() -> Option<Val<V>> {
    None
  }
}

pub trait ConstantOpsExt<T, V> {
  fn set_constant(self, c: T) -> Val<V>;
  fn add_constant(self, c: T) -> Val<V>;
  fn mult_constant(self, c: T) -> Val<V>;
}

pub trait SumJoinOpMaybeExt<V> {
  fn maybe_build(xs_: Vec<Val<V>>) -> Option<Val<V>>;
}

impl<V> SumJoinOpMaybeExt<V> for SumJoinOp {
  default fn maybe_build(xs_: Vec<Val<V>>) -> Option<Val<V>> {
    None
  }
}

impl<V> SumJoinOpMaybeExt<V> for SumJoinOp where SumJoinOp: SumJoinOpExt<V> {
  fn maybe_build(xs_: Vec<Val<V>>) -> Option<Val<V>> {
    Some(<SumJoinOp as SumJoinOpExt<V>>::build(xs_))
  }
}

pub trait SumJoinOpExt<V> {
  fn build(xs_: Vec<Val<V>>) -> Val<V>;
}

pub trait SumExt<V> {
  fn sum(xs_: Vec<Val<V>>) -> Val<V>;
}

impl<V> Add<Val<V>> for Val<V> where Self: SumExt<V> {
  type Output = Val<V>;

  fn add(self, x_: Val<V>) -> Val<V> {
    <Val<V> as SumExt<V>>::sum(vec![self, x_])
  }
}

pub trait FlatLinearExt<A, X, Y> {
  fn flat_mult(self, x_: Val<X>) -> Val<Y>;
}

impl<V> Mul<Val<V>> for Val<V> where Self: FlatLinearExt<V, V, V> {
  type Output = Val<V>;

  fn mul(self, x_: Val<V>) -> Val<V> {
    self.flat_mult(x_)
  }
}

pub trait FlatAffineExt<A, X, Y, B> {
  fn flat_mult(self, x_: Val<X>, b_: Val<B>) -> Val<Y>;
}

pub trait BroadcastAddExt<A, X, Y> {
  // TODO: axes.
  fn broadcast_add(self, axes: (), x_: Val<X>) -> Val<Y>;
}

pub trait BroadcastLinearExt<A, X, Y> {
  // TODO: axes.
  fn broadcast_mult(self, axes: (), x_: Val<X>) -> Val<Y>;
}

pub trait BroadcastAffineExt<A, X, Y, B> {
  // TODO: axes.
  fn broadcast_mult_add(self, axes: (), x_: Val<X>, b_: Val<B>) -> Val<Y>;
}

pub trait LinearExt<A, X, Y> {
  fn mult(self, x: Val<X>) -> Val<Y>;
}

pub trait AffineExt<A, X, Y, B> {
  fn mult_add(self, x: Val<X>, b: Val<B>) -> Val<Y>;
}

pub trait LeftTransposeLinearExt<A, Y, X> {
  fn left_transpose_mult(self, y: Val<Y>) -> Val<X>;
}

pub trait RightTransposeLinearExt<Y, X, A> {
  fn right_transpose_mult(self, x: Val<X>) -> Val<A>;
}

pub trait ConvLinearExt<A, X, Y> {
  fn conv(self, x: Val<X>) -> Val<Y>;
}

pub trait ConvAffineExt<A, X, Y, B> {
  fn conv_add(self, x: Val<X>, b: Val<B>) -> Val<Y>;
}

pub trait LeftTransposeConvLinearExt<A, Y, X> {
  fn left_transpose_conv(self, y: Val<Y>) -> Val<X>;
}

pub trait RightTransposeConvLinearExt<Y, X, A> {
  fn right_transpose_conv(self, x: Val<X>) -> Val<A>;
}
