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

//use std::marker::{PhantomData};
use std::ops::{Add, Mul};
use std::rc::{Rc};

pub struct PassOp;
pub struct FixOp;
pub struct DuplicateOp;

pub struct SerializeOp;
pub struct DeserializeOp;

pub struct CastOp;
pub struct DequantizeOp<T> { pub lo: T, pub hi: T }

pub struct SwitchOp;

pub struct SrcOp;
pub struct TouchSrcOp;
pub struct ZerosSrcOp;
pub struct OnesSrcOp;
// TODO: distribution parameters?
pub struct RandomBitsSrcOp;
pub struct UniformSrcOp;
pub struct NormalSrcOp;

pub struct FlattenOp;
pub struct ReshapeOp;
pub struct MapOp<MapF> { pub f: MapF, }
pub struct TransposeOp;
pub struct SumJoinOp;
pub struct SumJoinAccumulateOp;
pub struct FlatSumOp;
pub struct ReduceSumOp;
pub struct BatchSumOp;
pub struct FlatMapOp<FlatMapF> { pub f: FlatMapF }
pub struct FlatMapInplaceOp<FlatMapF> { pub f: FlatMapF }
pub struct FlatJoinOp<FlatJoin> { pub f: FlatJoin }
pub struct FlatLinearOp;
pub struct BatchMean2dOp;
pub struct BatchVariance2dOp;
pub struct BatchNormalize2dOp;
pub struct OnlineAverageOp;
pub struct LinearOp;
pub struct AffineOp;
pub struct LeftTransposeLinearOp;
pub struct RightTransposeLinearOp;
pub struct OuterLinearOp;
pub struct Conv1dLinearOp;
pub struct Conv1dAffineOp;
pub struct Conv2dLinearOp { pub conv_shape: Conv2dShape }
pub struct Conv2dAffineOp { pub conv_shape: Conv2dShape }
pub struct Conv3dLinearOp;
pub struct Conv3dAffineOp;
pub struct LeftTransposeConv1dLinearOp;
pub struct LeftTransposeConv2dLinearOp;
pub struct LeftTransposeConv3dLinearOp;
pub struct OuterConv1dLinearOp;
pub struct OuterConv2dLinearOp;
pub struct OuterConv3dLinearOp;
pub struct Resample2dOp<ResampleF> { pub f: ResampleF }
pub struct ReduceOp<ReduceF> { pub f: ReduceF, /*pub axes: _*/ }

pub struct SoftmaxNLLFusedOp;
pub struct SoftmaxCrossEntropyFusedOp;
pub struct SoftmaxEntropyFusedOp;

pub struct SpacePadOp;

#[derive(Clone)] pub struct IdentityFlatMapF;
#[derive(Clone)] pub struct ModulusFlatMapF;
#[derive(Clone)] pub struct SquareFlatMapF;
#[derive(Clone)] pub struct PositiveClipFlatMapF;
#[derive(Clone)] pub struct UnitStepFlatMapF;
#[derive(Clone)] pub struct LogPositiveClipFlatMapF;
#[derive(Clone)] pub struct PositiveReciprocalFlatMapF;
#[derive(Clone)] pub struct NormalCdfFlatMapF;
#[derive(Clone)] pub struct TanhFlatMapF;
#[derive(Clone)] pub struct RCosh2FlatMapF;

#[derive(Clone)] pub struct SumReduce;
#[derive(Clone)] pub struct ProductReduce;

#[derive(Clone)] pub struct Map2FlatJoin<Map1, Map2, Reduce> { pub map1: Map1, pub map2: Map2, pub reduce: Reduce }

#[derive(Clone, Copy)]
pub struct Conv2dShape {
  pub src_space_axes:   [isize; 2],
  pub src_feature_axis: isize,
  pub src_batch_axis:   isize,
  pub dst_space_axes:   [isize; 2],
  pub dst_feature_axis: isize,
  pub dst_batch_axis:   isize,
  pub ker_space_axes:   [isize; 2],
  pub ker_output_axis:  isize,
  pub ker_size:         [usize; 2],
  pub dilation:         [usize; 2],
  pub stride:           [usize; 2],
  pub zero_pad:         [usize; 2],
}

impl Conv2dShape {
  pub fn default_nchw() -> Self {
    Self::default_space_major()
  }

  pub fn default_space_major() -> Self {
    Conv2dShape{
      src_space_axes:   [0, 1],
      src_feature_axis: 2,
      src_batch_axis:   3,
      dst_space_axes:   [0, 1],
      dst_feature_axis: 2,
      dst_batch_axis:   3,
      ker_space_axes:   [0, 1],
      ker_output_axis:  3,
      ker_size:         [0, 0],
      dilation:         [1, 1],
      stride:           [1, 1],
      zero_pad:         [0, 0],
    }
  }

  pub fn default_nhwc() -> Self {
    Self::default_feature_major()
  }

  pub fn default_feature_major() -> Self {
    Conv2dShape{
      src_space_axes:   [1, 2],
      src_feature_axis: 0,
      src_batch_axis:   3,
      dst_space_axes:   [1, 2],
      dst_feature_axis: 0,
      dst_batch_axis:   3,
      ker_space_axes:   [0, 1],
      ker_output_axis:  3,
      ker_size:         [0, 0],
      dilation:         [1, 1],
      stride:           [1, 1],
      zero_pad:         [0, 0],
    }
  }

  pub fn default_chwn() -> Self {
    Self::default_batch_major()
  }

  pub fn default_batch_major() -> Self {
    Conv2dShape{
      src_space_axes:   [1, 2],
      src_feature_axis: 3,
      src_batch_axis:   0,
      dst_space_axes:   [1, 2],
      dst_feature_axis: 3,
      dst_batch_axis:   0,
      ker_space_axes:   [0, 1],
      ker_output_axis:  3,
      ker_size:         [0, 0],
      dilation:         [1, 1],
      stride:           [1, 1],
      zero_pad:         [0, 0],
    }
  }

  pub fn calculate_output_size(&self, w_size: [usize; 4], x_size: [usize; 3]) -> [usize; 3] {
    assert!(self.ker_size[0] >= 1);
    assert!(self.ker_size[1] >= 1);
    assert!(self.dilation[0] >= 1);
    assert!(self.dilation[1] >= 1);
    assert!(self.stride[0] >= 1);
    assert!(self.stride[1] >= 1);
    let src_w = x_size[self.src_space_axes[0] as usize];
    let src_h = x_size[self.src_space_axes[1] as usize];
    let dst_w = 1 + (src_w + 2 * self.zero_pad[0] - (((self.ker_size[0] - 1) * self.dilation[0]) + 1)) / self.stride[0];
    let dst_h = 1 + (src_h + 2 * self.zero_pad[1] - (((self.ker_size[1] - 1) * self.dilation[1]) + 1)) / self.stride[1];
    let dst_c = w_size[self.ker_output_axis as usize];
    let mut dst_size = [0, 0, 0];
    dst_size[self.dst_space_axes[0] as usize] = dst_w;
    dst_size[self.dst_space_axes[1] as usize] = dst_h;
    dst_size[self.dst_feature_axis as usize] = dst_c;
    dst_size
  }
}

#[derive(Clone, Copy)]
pub struct Pool2dShape {
  pub src_space_axes:   [isize; 2],
  pub src_feature_axis: isize,
  pub src_batch_axis:   isize,
  pub dst_space_axes:   [isize; 2],
  pub dst_feature_axis: isize,
  pub dst_batch_axis:   isize,
  pub filter:           [usize; 2],
  pub dilation:         [usize; 2],
  pub stride:           [usize; 2],
  pub zero_pad:         [usize; 2],
}

pub struct MaxPoolResample2dF;
pub struct AvgPoolResample2dF;
pub struct BilinearResample2dF;
pub struct BicubicResample2dF;

#[derive(Clone)] pub struct MeanReduceF;
#[derive(Clone)] pub struct VarianceReduceF;

pub trait PassExt<V> {
  fn pass(&self) -> Val<V>;
}

pub trait FixExt<V> {
  fn fix(&self) -> Val<V>;
}

pub trait FixOpExt<V> {
  fn build(x_: Val<V>) -> Val<V>;
}

pub trait SerializeExt<V> {
  fn serialize(&self) -> Val<V>;
}

pub trait DeserializeExt<V> {
  fn deserialize(&self, src: Val<V>) -> Node;
}

pub struct LinearScale;

pub trait DequantizeExt<V, W, T, /*Scale=LinearScale*/> {
  //fn dequantize(&self, base: T, range: T) -> Rc<F1Op<DequantizeFun<T, Scale>, V, W>>;
  fn dequantize(&self, lo: T, hi: T) -> Val<W>;
}

pub trait SwitchOpExt<V> {
  fn build(flag: TCell<bool>, off_: Val<V>, on_: Val<V>) -> Val<V>;
}

pub fn switch<V>(flag: TCell<bool>, off_: Val<V>, on_: Val<V>) -> Val<V> where SwitchOp: SwitchOpExt<V> {
  <SwitchOp as SwitchOpExt<V>>::build(flag, off_, on_)
}

pub trait SrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub fn src<V, Init>(init: Init) -> Val<V> where SrcOp: SrcOpExt<V, Init> {
  <SrcOp as SrcOpExt<V, Init>>::build(init)
}

pub trait TouchSrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub fn touch<V, Init>(init: Init) -> Val<V> where TouchSrcOp: TouchSrcOpExt<V, Init> {
  <TouchSrcOp as TouchSrcOpExt<V, Init>>::build(init)
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

pub trait ZerosSrcOpLikeExt<V> {
  fn build_like(x_: Val<V>) -> Val<V>;
}

pub fn zeros<V, Init>(init: Init) -> Val<V> where ZerosSrcOp: ZerosSrcOpExt<V, Init> {
  <ZerosSrcOp as ZerosSrcOpExt<V, Init>>::build(init)
}

pub fn zeros_like<V>(x_: Val<V>) -> Val<V> where ZerosSrcOp: ZerosSrcOpLikeExt<V> {
  <ZerosSrcOp as ZerosSrcOpLikeExt<V>>::build_like(x_)
}

pub trait OnesSrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub trait OnesSrcOpLikeExt<V> {
  fn build_like(x_: Val<V>) -> Val<V>;
}

pub trait OnesSrcOpMaybeExt<V> {
  fn maybe_build_like(x_: Val<V>) -> Option<Val<V>>;
}

impl<V> OnesSrcOpMaybeExt<V> for OnesSrcOp {
  default fn maybe_build_like(_: Val<V>) -> Option<Val<V>> {
    None
  }
}

pub fn ones<V, Init>(init: Init) -> Val<V> where OnesSrcOp: OnesSrcOpExt<V, Init> {
  <OnesSrcOp as OnesSrcOpExt<V, Init>>::build(init)
}

pub fn ones_like<V>(x_: Val<V>) -> Val<V> where OnesSrcOp: OnesSrcOpLikeExt<V> {
  <OnesSrcOp as OnesSrcOpLikeExt<V>>::build_like(x_)
}

pub trait FlattenExt<V, W> {
  fn flatten(self) -> Val<W>;
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

pub trait ReduceSumExt<V, W> {
  fn reduce_sum(self, axis: isize) -> Val<W>;
}

impl<V> Add<Val<V>> for Val<V> where Self: SumExt<V> {
  type Output = Val<V>;

  fn add(self, x_: Val<V>) -> Val<V> {
    <Val<V> as SumExt<V>>::sum(vec![self, x_])
  }
}

pub trait BatchMean2dOpExt<X, M> {
  fn build(axes: [isize; 2], x_: Val<X>) -> Val<M>;
}

pub trait BatchVariance2dOpExt<X, M> {
  fn build(axes: [isize; 2], x_: Val<X>) -> Val<M>;
}

pub trait BatchNormalize2dOpExt<X, M> {
  fn build(axes: [isize; 2], x_: Val<X>, mean_: Val<M>, var_: Val<M>) -> Val<X>;
}

pub trait BatchNormalizeExt<T, X, M> where T: Copy {
  fn batch_normalize_2d(self, axes: [isize; 2], online: TCell<bool>, epsilon: TCell<T>) -> (Val<X>, Val<M>, Val<M>, Val<M>, Val<M>);
}

impl<T, X, M> BatchNormalizeExt<T, X, M> for Val<X>
where T: Copy,
      X: 'static,
      M: 'static,
      BatchMean2dOp: BatchMean2dOpExt<X, M>,
      BatchVariance2dOp: BatchVariance2dOpExt<X, M>,
      BatchNormalize2dOp: BatchNormalize2dOpExt<X, M>,
      OnlineAverageOp: OnlineAverageOpExt<T, M>,
      ZerosSrcOp: ZerosSrcOpLikeExt<M> + ZerosSrcOpLikeExt<X>,
{
  fn batch_normalize_2d(self, axes: [isize; 2], online: TCell<bool>, epsilon: TCell<T>) -> (Val<X>, Val<M>, Val<M>, Val<M>, Val<M>) {
    let mean_ = <BatchMean2dOp as BatchMean2dOpExt<X, M>>::build(axes, self.clone());
    let var_ = <BatchVariance2dOp as BatchVariance2dOpExt<X, M>>::build(axes, self.clone());
    let avg_mean_ = zeros_like(mean_.clone()).online_average(epsilon.clone(), mean_.clone());
    let avg_var_ = zeros_like(var_.clone()).online_average(epsilon.clone(), var_.clone());
    let online_y_ = <BatchNormalize2dOp as BatchNormalize2dOpExt<X, M>>::build(axes, self.clone(), mean_.clone(), var_.clone());
    let avg_y_ = <BatchNormalize2dOp as BatchNormalize2dOpExt<X, M>>::build(axes, self.clone(), avg_mean_.clone(), avg_var_.clone());
    let y_ = switch(online, avg_y_, online_y_);
    (y_, mean_, var_, avg_mean_, avg_var_)
  }
}

pub trait OnlineAverageOpExt<T, V> where T: Copy {
  fn build(rate: TCell<T>, x_: Val<V>, y_: Val<V>) -> Val<V>;
}

pub trait OnlineAverageExt<T, V> where T: Copy {
  fn online_average(self, rate: TCell<T>, x_: Val<V>) -> Val<V>;
}

impl<T, V> OnlineAverageExt<T, V> for Val<V>
where T: Copy,
      OnlineAverageOp: OnlineAverageOpExt<T, V>,
{
  fn online_average(self, rate: TCell<T>, x_: Val<V>) -> Val<V> {
    <OnlineAverageOp as OnlineAverageOpExt<T, V>>::build(rate, x_, self)
  }
}

pub trait PositiveClipFlatMapExt<V> {
  fn positive_clip(self) -> Val<V>;

  fn rect(self) -> Val<V> where Self: Sized {
    self.positive_clip()
  }

  fn relu(self) -> Val<V> where Self: Sized {
    self.positive_clip()
  }
}

pub trait TanhFlatMapExt<V> {
  fn tanh(self) -> Val<V>;
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

pub trait OuterLinearExt<Y, X, A> {
  fn outer_mult(self, x: Val<X>) -> Val<A>;
}

pub trait ConvLinearExt<A, X, Y> {
  type ConvShape;

  fn conv(self, conv_shape: Self::ConvShape, x: Val<X>) -> Val<Y>;
}

pub trait ConvAffineExt<A, X, Y, B>: ConvLinearExt<A, X, Y> {
  fn conv_add(self, conv_shape: Self::ConvShape, x: Val<X>, b: Val<B>) -> Val<Y>;
}

pub trait LeftTransposeConvLinearExt<A, Y, X> {
  fn left_transpose_conv(self, y: Val<Y>) -> Val<X>;
}

pub trait OuterConvLinearExt<Y, X, A> {
  fn outer_conv(self, x: Val<X>) -> Val<A>;
}

impl<A: 'static> FixOpExt<A> for FixOp {
  fn build(x_: Val<A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<Self>| {
          x_._op()._value()._clone()
        })
      },
      apply: {
        Box::new(move |_: Txn, _state: RefMut<_>, _output: OVal<A>| {
          // The output should be a simple clone of the input,
          // so don't want to actually touch it.
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<Self>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(FixOp, ext, x_)))
  }
}

impl<A> SwitchOpExt<A> for SwitchOp
where A: 'static,
      ZerosSrcOp: ZerosSrcOpLikeExt<A>,
{
  fn build(flag: TCell<bool>, off_: Val<A>, on_: Val<A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |_state: RefMut<Self>| {
          unreachable!();
        })
      },
      apply: {
        Box::new(move |_: Txn, _state: RefMut<_>, _output: OVal<A>| {
          // The output should be a simple clone of one of the inputs,
          // so don't want to actually touch it.
        })
      },
      build: Some({
        let flag = flag.clone();
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: Some({
        let flag = flag.clone();
        let off_ = off_.clone();
        let on_ = on_.clone();
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<Self>, sink: &mut Sink| {
          if let Some(this_adj) = this.adjoint(sink) {
            let off_adj = switch(flag.clone(), this_adj.clone(), zeros_like(this_adj.clone()));
            off_.put_adjoint(off_adj, sink);
            let on_adj = switch(flag.clone(), zeros_like(this_adj.clone()), this_adj.clone());
            on_.put_adjoint(on_adj, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSwitchOp::new(SwitchOp, ext, flag, off_, on_)))
  }
}
