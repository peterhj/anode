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

use std::intrinsics::{type_name};
use std::marker::{PhantomData};
use std::ops::{Add, Mul};
use std::rc::{Rc};

//pub struct DefaultOpVariant;
//pub struct GPUOpVariant;

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
pub struct ReshapeLikeOp;
pub struct MapOp<MapF> { pub f: MapF, }
pub struct TransposeOp;
pub struct SumJoinOp;
pub struct SumJoinAccumulateOp;
//pub struct SumJoinOp<Variant> { _mrk: PhantomData<Variant> }
pub struct FlatSumOp;
pub struct ReduceSumOp;
pub struct BatchSumOp;
pub struct BatchBroadcastOp;
pub struct BatchBroadcastLikeOp;
pub struct FlatMapOp<FlatMapF> { pub f: FlatMapF }
pub struct FlatMapInplaceOp<FlatMapF> { pub f: FlatMapF }
pub struct FlatJoinOp<FlatJoin> { pub f: FlatJoin }
pub struct FlatLinearOp;
pub struct BroadcastLinearOp;
pub struct BroadcastAffineOp;
pub struct LinearReduceSumOp;
pub struct BatchMean2dOp;
pub struct BatchMean2dBwdOp;
pub struct BatchVariance2dOp;
pub struct BatchVariance2dBwdOp;
pub struct BatchVariance2dBwdMeanOp;
pub struct BatchNormalize2dOp;
pub struct BatchNormalize2dBwdOp;
pub struct BatchNormalize2dBwdMeanOp;
pub struct BatchNormalize2dBwdVarianceOp;
pub struct OnlineAverageOp;
pub struct SoftmaxOp;
pub struct SoftmaxCategoricalNLLOp;
pub struct SoftmaxCategoricalNLLBwdOp;
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
pub struct Conv2dReduceBwdOp { pub conv_shape: Conv2dShape }
pub struct LeftTransposeConv1dLinearOp;
pub struct LeftTransposeConv2dLinearOp { pub conv_shape: Conv2dShape }
pub struct LeftTransposeConv3dLinearOp;
pub struct OuterConv1dLinearOp;
pub struct OuterConv2dLinearOp { pub conv_shape: Conv2dShape }
pub struct OuterConv3dLinearOp;
pub struct Pool2dOp<Pool> { pub pool_shape: Pool2dShape, pub pool: Pool }
pub struct Pool2dBwdOp<Pool> { pub pool_shape: Pool2dShape, pub pool: Pool }
pub struct TransposePool2dOp<Pool> { pub pool_shape: Pool2dShape, pub pool: Pool }
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
  pub src_size:         [usize; 3],
  //pub src_size:         [usize; 2],
  //pub src_features:     usize,
  pub dst_space_axes:   [isize; 2],
  pub dst_feature_axis: isize,
  pub dst_batch_axis:   isize,
  pub ker_space_axes:   [isize; 2],
  pub ker_output_axis:  isize,
  pub ker_size:         [usize; 2],
  pub features:         usize,
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
      src_size:         [0, 0, 0],
      dst_space_axes:   [0, 1],
      dst_feature_axis: 2,
      dst_batch_axis:   3,
      ker_space_axes:   [0, 1],
      ker_output_axis:  3,
      ker_size:         [0, 0],
      features:         0,
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
      src_size:         [0, 0, 0],
      dst_space_axes:   [1, 2],
      dst_feature_axis: 0,
      dst_batch_axis:   3,
      ker_space_axes:   [0, 1],
      ker_output_axis:  3,
      ker_size:         [0, 0],
      features:         0,
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
      src_size:         [0, 0, 0],
      dst_space_axes:   [1, 2],
      dst_feature_axis: 3,
      dst_batch_axis:   0,
      ker_space_axes:   [0, 1],
      ker_output_axis:  3,
      ker_size:         [0, 0],
      features:         0,
      dilation:         [1, 1],
      stride:           [1, 1],
      zero_pad:         [0, 0],
    }
  }

  pub fn calculate_output_size(&self, w_size: [usize; 4], x_size: [usize; 3]) -> [usize; 3] {
    // TODO: this assumes NCHW layout.
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
  pub src_size:         [usize; 2],
  pub src_features:     usize,
  pub dst_space_axes:   [isize; 2],
  pub dst_feature_axis: isize,
  pub dst_batch_axis:   isize,
  pub ker_size:         [usize; 2],
  pub stride:           [usize; 2],
  pub zero_pad:         [usize; 2],
}

impl Pool2dShape {
  pub fn default_nchw() -> Self {
    Self::default_space_major()
  }

  pub fn default_space_major() -> Self {
    Pool2dShape{
      src_space_axes:   [0, 1],
      src_feature_axis: 2,
      src_batch_axis:   3,
      src_size:         [0, 0],
      src_features:     0,
      dst_space_axes:   [0, 1],
      dst_feature_axis: 2,
      dst_batch_axis:   3,
      ker_size:         [0, 0],
      stride:           [1, 1],
      zero_pad:         [0, 0],
    }
  }

  pub fn calculate_output_size(&self, x_size: [usize; 3]) -> [usize; 3] {
    // TODO: this assumes NCHW layout.
    assert!(self.ker_size[0] >= 1);
    assert!(self.ker_size[1] >= 1);
    assert!(self.stride[0] >= 1);
    assert!(self.stride[1] >= 1);
    let src_w = x_size[self.src_space_axes[0] as usize];
    let src_h = x_size[self.src_space_axes[1] as usize];
    // FIXME: this is a copy-paste of the conv calculation,
    // but pooling might be different; double check this.
    let dst_w = 1 + (src_w + 2 * self.zero_pad[0] - self.ker_size[0]) / self.stride[0];
    let dst_h = 1 + (src_h + 2 * self.zero_pad[1] - self.ker_size[1]) / self.stride[1];
    let dst_c = self.src_features;
    let mut dst_size = [0, 0, 0];
    dst_size[self.dst_space_axes[0] as usize] = dst_w;
    dst_size[self.dst_space_axes[1] as usize] = dst_h;
    dst_size[self.dst_feature_axis as usize] = dst_c;
    dst_size
  }
}

pub struct AveragePool;
pub struct MaxPool;

pub struct MaxPoolResample2dF;
pub struct AvgPoolResample2dF;
pub struct BilinearResample2dF;
pub struct BicubicResample2dF;

#[derive(Clone)] pub struct MeanReduceF;
#[derive(Clone)] pub struct VarianceReduceF;

pub trait PassOpExt<V> {
  fn build(x_: Val<V>) -> Val<V>;
}

pub trait PassExt<V> {
  fn pass(self) -> Val<V>;
}

impl<V> PassExt<V> for Val<V> where PassOp: PassOpExt<V> {
  fn pass(self) -> Val<V> {
    <PassOp as PassOpExt<V>>::build(self)
  }
}

pub trait FixOpExt<V> {
  fn build(x_: Val<V>) -> Val<V>;
}

pub trait FixExt<V> {
  fn fix(self) -> Val<V>;
}

impl<V> FixExt<V> for Val<V> where FixOp: FixOpExt<V> {
  fn fix(self) -> Val<V> {
    <FixOp as FixOpExt<V>>::build(self)
  }
}

pub trait SerializeExt<V> {
  fn serialize(&self) -> Val<V>;
}

pub trait DeserializeExt<V> {
  fn deserialize(&self, src: Val<V>) -> Node;
}

pub trait DequantizeOpExt<T, V, W> {
  fn build(lo: T, hi: T, x_: Val<V>) -> Val<W>;
}

pub trait DequantizeExt<T, V, W> {
  fn dequantize(self, lo: T, hi: T) -> Val<W>;
}

impl<T, V, W> DequantizeExt<T, V, W> for Val<V> where DequantizeOp<T>: DequantizeOpExt<T, V, W> {
  fn dequantize(self, lo: T, hi: T) -> Val<W> {
    <DequantizeOp<T> as DequantizeOpExt<T, V, W>>::build(lo, hi, self)
  }
}

pub trait SwitchOpExt<V> {
  //fn build(flag: TCell<bool>, off_: Val<V>, on_: Val<V>) -> Val<V>;
  fn build(flag: Val<bool>, off_: Val<V>, on_: Val<V>) -> Val<V>;
}

//pub fn switch<V>(flag: TCell<bool>, off_: Val<V>, on_: Val<V>) -> Val<V> where SwitchOp: SwitchOpExt<V> {
pub fn switch<V>(flag: Val<bool>, off_: Val<V>, on_: Val<V>) -> Val<V> where SwitchOp: SwitchOpExt<V> {
  <SwitchOp as SwitchOpExt<V>>::build(flag, off_, on_)
}

pub trait SrcOpExt<V, Init> {
  fn build(init: Init) -> Val<V>;
}

pub trait SrcOpCloneExt<V: Clone + 'static> {
  fn build_clone(init_value: V) -> Val<V>;
}

pub fn src<V, Init>(init: Init) -> Val<V> where SrcOp: SrcOpExt<V, Init> {
  <SrcOp as SrcOpExt<V, Init>>::build(init)
}

pub fn src_init<V>(init_value: V) -> Val<V> where V: Clone + 'static, SrcOp: SrcOpCloneExt<V> {
  <SrcOp as SrcOpCloneExt<V>>::build_clone(init_value)
}

impl<T: Clone + 'static> SrcOpCloneExt<T> for SrcOp {
  fn build_clone(init_value: T) -> Val<T> {
    let ext = OpExt{
      make_val: {
        Box::new(move |_: RefMut<_>| {
          let init_value = init_value.clone();
          RWVal::from(Arc::new(move |_txn: Txn| {
            init_value.clone()
          }))
        })
      },
      apply: {
        Box::new(move |txn: Txn, _: RefMut<_>, output: OVal<_>| {
          output.write_v2(txn, |_, _| {
            panic!("WARNING: SrcOpExt: should never write");
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<_>, _: RefMut<_>, _sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(SrcOp, ext)))
  }
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
  fn maybe_build_inplace(xs_: Vec<Val<V>>) -> Option<(Val<V>, Vec<Val<V>>)>;
}

impl<V> SumJoinOpMaybeExt<V> for SumJoinOp {
  default fn maybe_build(xs_: Vec<Val<V>>) -> Option<Val<V>> {
    //println!("DEBUG: SumJoinOpMaybeExt: maybe build: none");
    None
  }

  default fn maybe_build_inplace(xs_: Vec<Val<V>>) -> Option<(Val<V>, Vec<Val<V>>)> {
    //println!("DEBUG: SumJoinOpMaybeExt: maybe build inplace: none");
    None
  }
}

impl<V> SumJoinOpMaybeExt<V> for SumJoinOp where SumJoinOp: SumJoinOpExt<V> {
  fn maybe_build(xs_: Vec<Val<V>>) -> Option<Val<V>> {
    //println!("DEBUG: SumJoinOpMaybeExt: maybe build: SOME");
    Some(<SumJoinOp as SumJoinOpExt<V>>::build(xs_))
  }

  fn maybe_build_inplace(xs_: Vec<Val<V>>) -> Option<(Val<V>, Vec<Val<V>>)> {
    //println!("DEBUG: SumJoinOpMaybeExt: maybe build inplace: SOME");
    Some(<SumJoinOp as SumJoinOpExt<V>>::build_inplace(xs_))
  }
}

pub trait SumJoinOpExt<V> {
  fn build(xs_: Vec<Val<V>>) -> Val<V>;
  fn build_inplace(xs_: Vec<Val<V>>) -> (Val<V>, Vec<Val<V>>);
}

pub fn sum<V>(xs_: Vec<Val<V>>) -> Val<V> where Val<V>: SumExt<V> {
  <Val<V> as SumExt<V>>::sum(xs_)
}

pub trait SumExt<V> {
  fn sum(xs_: Vec<Val<V>>) -> Val<V>;
}

impl<V> SumExt<V> for Val<V> where SumJoinOp: SumJoinOpExt<V> {
  fn sum(xs_: Vec<Val<V>>) -> Val<V> {
    <SumJoinOp as SumJoinOpExt<V>>::build(xs_)
  }
}

impl<V> Add<Val<V>> for Val<V> where Val<V>: SumExt<V> {
  type Output = Val<V>;

  fn add(self, x_: Val<V>) -> Val<V> {
    <Val<V> as SumExt<V>>::sum(vec![self, x_])
  }
}

pub fn sum_inplace_unstable<V>(xs_: Vec<Val<V>>) -> Val<V> where Val<V>: SumInplaceExt<V> {
  <Val<V> as SumInplaceExt<V>>::sum_inplace_unstable(xs_)
}

pub trait SumInplaceExt<V> {
  fn sum_inplace_unstable(xs_: Vec<Val<V>>) -> Val<V>;
}

impl<V> SumInplaceExt<V> for Val<V> where SumJoinOp: SumJoinOpExt<V> {
  fn sum_inplace_unstable(xs_: Vec<Val<V>>) -> Val<V> {
    <SumJoinOp as SumJoinOpExt<V>>::build_inplace(xs_).0
  }
}

pub trait ReduceSumExt<V, W> {
  fn reduce_sum(self, axis: isize) -> Val<W>;
}

pub trait BatchSumOpExt<V, W> {
  fn build(x_: Val<V>) -> Val<W>;
}

pub trait BatchSumExt<V, W> {
  fn batch_sum(self) -> Val<W>;
}

impl<V, W> BatchSumExt<V, W> for Val<V> where BatchSumOp: BatchSumOpExt<V, W> {
  fn batch_sum(self) -> Val<W> {
    <BatchSumOp as BatchSumOpExt<V, W>>::build(self)
  }
}

pub trait BatchBroadcastOpExt<V, W> {
  fn build(x_: Val<V>, target: usize) -> Val<W>;
}

pub trait BatchBroadcastExt<V, W> {
  fn batch_broadcast(self, target: usize) -> Val<W>;
}

impl<V, W> BatchBroadcastExt<V, W> for Val<V> where BatchBroadcastOp: BatchBroadcastOpExt<V, W> {
  fn batch_broadcast(self, target: usize) -> Val<W> {
    <BatchBroadcastOp as BatchBroadcastOpExt<V, W>>::build(self, target)
  }
}

pub trait BatchBroadcastLikeOpExt<V, W> {
  fn build(x_: Val<V>, target_: Val<W>) -> Val<W>;
}

pub trait BatchBroadcastLikeExt<V, W> {
  fn batch_broadcast_like(self, target: Val<W>) -> Val<W>;
}

impl<V, W> BatchBroadcastLikeExt<V, W> for Val<V> where BatchBroadcastLikeOp: BatchBroadcastLikeOpExt<V, W> {
  fn batch_broadcast_like(self, target: Val<W>) -> Val<W> {
    <BatchBroadcastLikeOp as BatchBroadcastLikeOpExt<V, W>>::build(self, target)
  }
}

pub trait BatchMean2dOpExt<T, X, M> {
  fn build(axes: [isize; 2], x_: Val<X>) -> Val<M>;
}

pub trait BatchVariance2dOpExt<T, X, M> {
  fn build(axes: [isize; 2], epsilon: T, x_: Val<X>, mean_: Val<M>) -> Val<M>;
}

pub trait BatchNormalize2dOpExt<T, X, M> {
  fn build(axes: [isize; 2], x_: Val<X>, mean_: Val<M>, var_: Val<M>) -> Val<X>;
}

pub trait BatchNormalizeExt<T, X, M> where T: Copy {
  //fn batch_normalize_2d(self, axes: [isize; 2], online: TCell<bool>, avg_rate: TCell<T>, epsilon: T) -> (Val<X>, Val<M>, Val<M>, Val<M>, Val<M>);
  fn batch_normalize_2d(self, axes: [isize; 2], online: Val<bool>, avg_rate: Val<T>, epsilon: T) -> (Val<X>, Val<M>, Val<M>, Val<M>, Val<M>);
}

impl<T, X, M> BatchNormalizeExt<T, X, M> for Val<X>
where T: Copy + 'static,
      X: 'static,
      M: 'static,
      BatchMean2dOp: BatchMean2dOpExt<T, X, M>,
      BatchVariance2dOp: BatchVariance2dOpExt<T, X, M>,
      BatchNormalize2dOp: BatchNormalize2dOpExt<T, X, M>,
      OnlineAverageOp: OnlineAverageOpExt<T, M>,
      ZerosSrcOp: ZerosSrcOpLikeExt<M> + ZerosSrcOpLikeExt<X>,
{
  //fn batch_normalize_2d(self, axes: [isize; 2], online: TCell<bool>, avg_rate: TCell<T>, epsilon: T) -> (Val<X>, Val<M>, Val<M>, Val<M>, Val<M>) {
  fn batch_normalize_2d(self, axes: [isize; 2], online: Val<bool>, avg_rate: Val<T>, epsilon: T) -> (Val<X>, Val<M>, Val<M>, Val<M>, Val<M>) {
    let mean_ = <BatchMean2dOp as BatchMean2dOpExt<T, X, M>>::build(axes, self.clone());
    let var_ = <BatchVariance2dOp as BatchVariance2dOpExt<T, X, M>>::build(axes, epsilon, self.clone(), mean_.clone());
    let avg_mean_ = zeros_like(mean_.clone()).online_average(avg_rate.clone(), mean_.clone());
    let avg_var_ = zeros_like(var_.clone()).online_average(avg_rate.clone(), var_.clone());
    let online_y_ = <BatchNormalize2dOp as BatchNormalize2dOpExt<T, X, M>>::build(axes, self.clone(), mean_.clone(), var_.clone());
    let avg_y_ = <BatchNormalize2dOp as BatchNormalize2dOpExt<T, X, M>>::build(axes, self.clone(), avg_mean_.clone(), avg_var_.clone()).fix();
    let y_ = switch(online, avg_y_, online_y_);
    (y_, mean_, var_, avg_mean_, avg_var_)
  }
}

pub trait OnlineAverageOpExt<T, V> where T: Copy {
  fn build(avg_rate: Val<T>, x_: Val<V>, y_: Val<V>) -> Val<V>;
}

pub trait OnlineAverageExt<T, V> where T: Copy {
  fn online_average(self, avg_rate: Val<T>, x_: Val<V>) -> Val<V>;
}

impl<T, V> OnlineAverageExt<T, V> for Val<V>
where T: Copy,
      OnlineAverageOp: OnlineAverageOpExt<T, V>,
{
  fn online_average(self, avg_rate: Val<T>, x_: Val<V>) -> Val<V> {
    <OnlineAverageOp as OnlineAverageOpExt<T, V>>::build(avg_rate, x_, self)
  }
}

pub trait SoftmaxOpExt<X> {
  fn build(x_: Val<X>) -> Val<X>;
}

pub trait SoftmaxExt<X> {
  fn softmax(self) -> Val<X>;
}

impl<X> SoftmaxExt<X> for Val<X> where SoftmaxOp: SoftmaxOpExt<X> {
  fn softmax(self) -> Val<X> {
    <SoftmaxOp as SoftmaxOpExt<X>>::build(self)
  }
}

pub trait SoftmaxCategoricalNLLOpExt<T, X, K, L> {
  fn build(x_: Val<X>, softmax_: Val<X>, category_data_: Val<K>) -> Val<L>;
}

pub trait SoftmaxCategoricalNLLExt<T, X, K, L> where T: Copy {
  fn softmax_categorical_nll(self, category_data_: Val<K>) -> (Val<L>, Val<X>);
}

impl<T, X, K, L> SoftmaxCategoricalNLLExt<T, X, K, L> for Val<X>
where T: Copy,
      X: 'static,
      SoftmaxOp: SoftmaxOpExt<X>,
      SoftmaxCategoricalNLLOp: SoftmaxCategoricalNLLOpExt<T, X, K, L>,
{
  fn softmax_categorical_nll(self, category_data_: Val<K>) -> (Val<L>, Val<X>) {
    let softmax_ = <SoftmaxOp as SoftmaxOpExt<X>>::build(self.clone());
    let nll_ = <SoftmaxCategoricalNLLOp as SoftmaxCategoricalNLLOpExt<T, X, K, L>>::build(self.clone(), softmax_.clone().fix(), category_data_);
    (nll_, softmax_)
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

pub trait Broadcast1dAddExt<A, X, Y> {
  fn broadcast_1d_add(self, axis: isize, x_: Val<X>) -> Val<Y>;
}

pub trait Broadcast1dLinearExt<A, X, Y> {
  fn broadcast_1d_mult(self, axis: isize, x_: Val<X>) -> Val<Y>;
}

pub trait Broadcast1dAffineExt<A, X, Y, B> {
  fn broadcast_1d_mult_add(self, axis: isize, x_: Val<X>, b_: Val<B>) -> Val<Y>;
}

pub trait Reduce1dSumExt<X, Y> {
  fn reduce_1d_sum(self, axis: isize, x_: Val<X>) -> Val<Y>;
}

pub trait MultReduce1dSumExt<X, Y> {
  fn mult_reduce_1d_sum(self, axis: isize, x1_: Val<X>, x2_: Val<X>) -> Val<Y>;
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

pub trait ConvReduceBwdExt<X, B> {
  type ConvShape;

  fn conv_reduce_bwd(self, conv_shape: Self::ConvShape) -> Val<B>;
}

pub trait LeftTransposeConvLinearExt<A, X, Y> {
  type ConvShape;

  fn left_transpose_conv(self, conv_shape: Self::ConvShape, y: Val<Y>) -> Val<X>;
}

pub trait OuterConvLinearExt<A, X, Y> {
  type ConvShape;

  fn outer_conv(self, conv_shape: Self::ConvShape, x: Val<X>) -> Val<A>;
}

pub trait PoolExt<X> {
  type PoolShape;

  fn average_pool(self, pool_shape: Self::PoolShape) -> Val<X>;
  fn max_pool(self, pool_shape: Self::PoolShape) -> Val<X>;
}

pub trait PoolBwdExt<X>: PoolExt<X> {
  fn average_pool_bwd(self, pool_shape: Self::PoolShape, y: Val<X>, x: Val<X>) -> Val<X>;
  fn max_pool_bwd(self, pool_shape: Self::PoolShape, y: Val<X>, x: Val<X>) -> Val<X>;
}

pub trait SomePoolBwdExt<Pool, X>: PoolBwdExt<X> {
  fn pool_bwd(self, pool_shape: Self::PoolShape, y: Val<X>, x: Val<X>) -> Val<X>;
}

impl<X> SomePoolBwdExt<AveragePool, X> for Val<X> where Val<X>: PoolBwdExt<X> {
  fn pool_bwd(self, pool_shape: Self::PoolShape, y: Val<X>, x: Val<X>) -> Val<X> {
    self.average_pool_bwd(pool_shape, y, x)
  }
}

impl<X> SomePoolBwdExt<MaxPool, X> for Val<X> where Val<X>: PoolBwdExt<X> {
  fn pool_bwd(self, pool_shape: Self::PoolShape, y: Val<X>, x: Val<X>) -> Val<X> {
    self.max_pool_bwd(pool_shape, y, x)
  }
}

pub trait TransposePoolExt<X>: PoolExt<X> {
  fn transpose_pool(self, pool_shape: Self::PoolShape) -> Val<X>;
}

pub struct WriteSection;

pub trait WriteSectionImpl<A: 'static>: Clone {
  fn copy(&mut self, dst: &mut A, src: &A) {
    unimplemented!("WriteSectionImpl: impl type '{}' missing copy for data type '{}'", unsafe { type_name::<Self>() }, unsafe { type_name::<A>() });
  }

  fn add(&mut self, dst: &mut A, src: &A) {
    //unimplemented!("WriteSectionImpl: missing add");
    unimplemented!("WriteSectionImpl: impl type '{}' missing add for data type '{}'", unsafe { type_name::<Self>() }, unsafe { type_name::<A>() });
  }
}

pub trait WriteSectionExt<A: 'static> {
  type Section: WriteSectionImpl<A>;

  fn maybe() -> Option<Self::Section>;
}

impl<A: 'static> WriteSectionImpl<A> for () {
}

impl<A: 'static> WriteSectionExt<A> for WriteSection {
  default type Section = ();

  default fn maybe() -> Option<Self::Section> {
    None
  }
}

pub fn pass_apply<F, A: 'static>(x_: Val<A>) -> Box<Fn(Txn, RefMut<F>, OVal<A>) -> bool> {
  let section = match <WriteSection as WriteSectionExt<A>>::maybe() {
    None => unimplemented!("pass_apply: missing WriteSection impl for data type '{}'", unsafe { type_name::<A>() }),
    Some(section) => section,
  };
  Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<A>| {
    if output._valref().is_some() && x_._valref() != output._valref() {
      //if let Some((cap, token)) = output.write(txn) {
      output.write_v2(txn, |cap, token| {
        let mut section = section.clone();
        let x = x_.get(txn);
        let mut y = output.get_mut(txn, token);
        match cap {
          WriteCap::Assign => {
            section.copy(&mut *y, &*x);
          }
          WriteCap::Accumulate => {
            section.add(&mut *y, &*x);
          }
        }
      })
    } else {
      false
    }
  })
}

impl<A: 'static> PassOpExt<A> for PassOp {
  fn build(x_: Val<A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          x_._make_value()
        })
      },
      apply: {
        pass_apply::<_, A>(x_.clone())
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(this_adj) = this.adjoint(sink) {
            x_.put_adjoint(this_adj, sink);
          }
        })
      }),
      inplace: None,
    };
    let x_value = x_._static_value();
    Val::with_value(Rc::new(F1Op::new(PassOp, ext, x_)), x_value)
  }
}

impl<A: 'static> FixOpExt<A> for FixOp {
  fn build(x_: Val<A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          x_._make_value()
        })
      },
      apply: {
        pass_apply::<_, A>(x_.clone())
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: None,
      /*adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<Self>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),*/
      inplace: None,
    };
    let x_value = x_._static_value();
    Val::with_value(Rc::new(F1Op::new(FixOp, ext, x_)), x_value)
  }
}

//pub fn switch_apply<F, A: 'static>(flag: TCell<bool>, off_: Val<A>, on_: Val<A>) -> Box<Fn(Txn, RefMut<F>, OVal<A>) -> bool> {
pub fn switch_apply<F, A: 'static>(flag: Val<bool>, off_: Val<A>, on_: Val<A>) -> Box<Fn(Txn, RefMut<F>, OVal<A>) -> bool> {
  let section = match <WriteSection as WriteSectionExt<A>>::maybe() {
    None => unimplemented!("switch_apply: missing WriteSection impl for data type '{}'", unsafe { type_name::<A>() }),
    Some(section) => section,
  };
  Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<A>| {
    let x_ = match *flag.get(txn) {
      false => &off_,
      true  => &on_,
    };
    if !output._valref().is_none() && x_._valref() != output._valref() {
      //if let Some((cap, token)) = output.write(txn) {
      output.write_v2(txn, |cap, token| {
        let mut section = section.clone();
        let x = x_.get(txn);
        let mut y = output.get_mut(txn, token);
        match cap {
          WriteCap::Assign => {
            section.copy(&mut *y, &*x);
          }
          WriteCap::Accumulate => {
            section.add(&mut *y, &*x);
          }
        }
      })
    } else {
      false
    }
  })
}

impl<A> SwitchOpExt<A> for SwitchOp
where A: 'static,
      ZerosSrcOp: ZerosSrcOpLikeExt<A>,
{
  //fn build(flag: TCell<bool>, off_: Val<A>, on_: Val<A>) -> Val<A> {
  fn build(flag: Val<bool>, off_: Val<A>, on_: Val<A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          unreachable!();
        })
      },
      apply: {
        /*Box::new(move |_: Txn, _state: RefMut<_>, _output: OVal<A>| {
          // The output should be a simple clone of one of the inputs,
          // so don't want to actually touch it.
        })*/
        switch_apply::<_, A>(flag.clone(), off_.clone(), on_.clone())
      },
      build: Some({
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
    //Val::from(Rc::new(FSwitchOp::new(SwitchOp, ext, flag, off_, on_)))
    Val::with_value(Rc::new(FSwitchOp::new(SwitchOp, ext, flag, off_, on_)), None)
  }
}
