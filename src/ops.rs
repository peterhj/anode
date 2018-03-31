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
use std::ops::{DerefMut};
use std::rc::{Rc};

pub trait MemIoReader<'a> {
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
}

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

pub trait SumExt<X, V> {
  fn sum(xs_: Vec<Rc<AOp<V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>>;
  fn add(self, x_: Rc<AOp<V>>) -> Rc<FJoinOp<SumJoinOp, V, V>>;
}

pub trait FlatMultOpExt<X, V1, A, V2, Y, W>
{
  fn flat_mult(self, x: Rc<AOp<V1>>) -> Rc<F2Op<FlatLinearMapFun, V1, V2, W>>;
}

pub trait MultOpExt<X, V1, A, V2, Y, W>
{
  fn mult(self, x: Rc<AOp<V1>>) -> Rc<F2Op<LinearMapOp, V1, V2, W>>;
}

pub trait MultAddOpExt<X, V1, A, V2, B, V3, Y, W>
{
  fn mult_add(self, x: Rc<AOp<V1>>, shift: Rc<AOp<V3>>) -> Rc<F3Op<LinearMapOp, V1, V2, V3, W>>;
}

pub trait LinearExt<A, X, Y> {
  //fn mult(&self, x: Rc<AOp<X>>) -> Rc<F2Op<LinearMapOp, A, X, Y>>;
  fn mult(&self, x: Val<X>) -> Val<Y>;
}

pub trait LeftTransposeLinearExt<A, Y, X> {
  //fn mult_left_transpose(&self, y: Rc<AOp<Y>>) -> Rc<F2Op<LeftTransposeLinearMapOp, A, Y, X>>;
  fn mult_left_transpose(&self, y: Val<Y>) -> Val<X>;
}

pub trait RightTransposeLinearExt<Y, X, A> {
  //fn mult_right_transpose(&self, x: Rc<AOp<X>>) -> Rc<F2Op<RightTransposeLinearMapOp, Y, X, A>>;
  fn mult_right_transpose(&self, x: Val<X>) -> Val<A>;
}

/*pub trait ConvLinearExt<A, X, Y> {
  fn conv(&self, x: Rc<AOp<X>>) -> Rc<F2Op<ConvLinearMapOp, A, X, Y>>;
}

pub trait LeftTransposeConvLinearExt<A, Y, X> {
  fn conv_left_transpose(&self, y: Rc<AOp<Y>>) -> Rc<F2Op<LeftTransposeConvLinearMapOp, A, Y, X>>;
}

pub trait RightTransposeConvLinearExt<Y, X, A> {
  fn conv_right_transpose(&self, x: Rc<AOp<X>>) -> Rc<F2Op<RightTransposeConvLinearMapOp, Y, X, A>>;
}*/
