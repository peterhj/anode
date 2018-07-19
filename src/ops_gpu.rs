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
use context::*;
use ffi::routines_gpu::*;
use log::*;
use ops::*;
use utils::{ZerosInit};

//use arithmetic::*;
//use arrayidx::*;
//use cuda_blas::*;
use cuda_dnn::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use gpudevicemem::array::linalg::*;
use gpudevicemem::array::tensor::*;
use gpudevicemem::array::tensor::conv::*;
use gpudevicemem::array::tensor::pool::*;
use gpudevicemem::array::tensor::softmax::*;
use gpudevicemem::ffi::routines_gpu::*;
use memarray::*;
use num_traits::identities::*;
use rand::prelude::{Rng};

use std::cell::{RefMut};
use std::iter::{FromIterator};
use std::marker::{PhantomData};
//use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::ops::{Add, Mul};
use std::sync::{Arc};

#[inline]
fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

#[inline]
fn sz2uint(sz: usize) -> u32 {
  assert!(sz <= u32::max_value() as _);
  sz as _
}

pub struct GPUPlacement {
  dev:  GPUDeviceId,
}

impl Placement for GPUPlacement {
  fn _place(&self) -> Rc<dyn PlaceGuard> {
    let ctx = implicit_ctx().multi_gpu().gpu(self.dev);
    let g = push_ctx(ctx);
    Rc::new(CtxPlaceGuard{ctxg: g})
  }
}

#[derive(Clone, Default)]
pub struct GPUWriteSection {
  section:  GPULazyAsyncSection,
}

impl WriteSectionExt<GPUDeviceScalar<f32>> for WriteSection {
  type Section = GPUWriteSection;

  fn maybe() -> Option<Self::Section> {
    Some(GPUWriteSection::default())
  }
}

impl WriteSectionImpl<GPUDeviceScalar<f32>> for GPUWriteSection {
  fn copy(&mut self, dst: &mut GPUDeviceScalar<f32>, src: &GPUDeviceScalar<f32>) {
    //println!("DEBUG: GPU write section: in copy...");
    let ctx = implicit_ctx().gpu();
    let mut pool = ctx.pool();
    let conn = pool.conn();
    let mut guard = self.section.enter(conn.clone());
    guard._wait(src.async_state());
    guard._wait(dst.async_state());
    dst.as_view_mut().copy(src.as_view(), conn);
  }

  fn add(&mut self, dst: &mut GPUDeviceScalar<f32>, src: &GPUDeviceScalar<f32>) {
    //println!("DEBUG: GPU write section: in add...");
    let ctx = implicit_ctx().gpu();
    let mut pool = ctx.pool();
    let conn = pool.conn();
    let mut guard = self.section.enter(conn.clone());
    guard._wait(src.async_state());
    guard._wait(dst.async_state());
    dst.as_view_mut().add(src.as_view(), conn);
  }
}

impl WriteSectionExt<GPUDeviceArray1d<f32>> for WriteSection {
  type Section = GPUWriteSection;

  fn maybe() -> Option<Self::Section> {
    Some(GPUWriteSection::default())
  }
}

impl WriteSectionImpl<GPUDeviceArray1d<f32>> for GPUWriteSection {
  fn copy(&mut self, dst: &mut GPUDeviceArray1d<f32>, src: &GPUDeviceArray1d<f32>) {
    let ctx = implicit_ctx().gpu();
    let mut pool = ctx.pool();
    let conn = pool.conn();
    let mut guard = self.section.enter(conn.clone());
    guard._wait(src.async_state());
    guard._wait(dst.async_state());
    dst.as_view_mut().copy(src.as_view(), conn);
  }

  fn add(&mut self, dst: &mut GPUDeviceArray1d<f32>, src: &GPUDeviceArray1d<f32>) {
    let ctx = implicit_ctx().gpu();
    let mut pool = ctx.pool();
    let conn = pool.conn();
    let mut guard = self.section.enter(conn.clone());
    guard._wait(src.async_state());
    guard._wait(dst.async_state());
    dst.as_view_mut().add(src.as_view(), conn);
  }
}

impl WriteSectionExt<GPUDeviceOuterBatchScalar<f32>> for WriteSection {
  type Section = GPUWriteSection;

  fn maybe() -> Option<Self::Section> {
    Some(GPUWriteSection::default())
  }
}

impl WriteSectionImpl<GPUDeviceOuterBatchScalar<f32>> for GPUWriteSection {
  fn copy(&mut self, dst: &mut GPUDeviceOuterBatchScalar<f32>, src: &GPUDeviceOuterBatchScalar<f32>) {
    // TODO
    unimplemented!();
  }

  fn add(&mut self, dst: &mut GPUDeviceOuterBatchScalar<f32>, src: &GPUDeviceOuterBatchScalar<f32>) {
    // TODO
    unimplemented!();
  }
}

impl WriteSectionExt<GPUDeviceOuterBatchArray1d<f32>> for WriteSection {
  type Section = GPUWriteSection;

  fn maybe() -> Option<Self::Section> {
    Some(GPUWriteSection::default())
  }
}

impl WriteSectionImpl<GPUDeviceOuterBatchArray1d<f32>> for GPUWriteSection {
  fn copy(&mut self, dst: &mut GPUDeviceOuterBatchArray1d<f32>, src: &GPUDeviceOuterBatchArray1d<f32>) {
    // TODO
    unimplemented!();
  }

  fn add(&mut self, dst: &mut GPUDeviceOuterBatchArray1d<f32>, src: &GPUDeviceOuterBatchArray1d<f32>) {
    // TODO
    unimplemented!();
  }
}

impl WriteSectionExt<GPUDeviceOuterBatchArray3d<f32>> for WriteSection {
  type Section = GPUWriteSection;

  fn maybe() -> Option<Self::Section> {
    Some(GPUWriteSection::default())
  }
}

impl WriteSectionImpl<GPUDeviceOuterBatchArray3d<f32>> for GPUWriteSection {
  fn copy(&mut self, dst: &mut GPUDeviceOuterBatchArray3d<f32>, src: &GPUDeviceOuterBatchArray3d<f32>) {
    let ctx = implicit_ctx().gpu();
    let mut pool = ctx.pool();
    let conn = pool.conn();
    let mut guard = self.section.enter(conn.clone());
    guard._wait(src.async_state());
    guard._wait(dst.async_state());
    dst.as_view_mut().copy(src.as_view(), conn);
  }

  fn add(&mut self, dst: &mut GPUDeviceOuterBatchArray3d<f32>, src: &GPUDeviceOuterBatchArray3d<f32>) {
    let ctx = implicit_ctx().gpu();
    let mut pool = ctx.pool();
    let conn = pool.conn();
    let mut guard = self.section.enter(conn.clone());
    guard._wait(src.async_state());
    guard._wait(dst.async_state());
    dst.as_view_mut().add(src.as_view(), conn);
  }
}

pub struct GPUOp;
pub struct MultiGPUBroadcastOp;
pub struct MultiGPUBatchSplitOp;
pub struct MultiGPUSpaceSplit1dOp;
pub struct MultiGPUSpaceSplit2dOp;

pub trait MultiGPUBroadcastExt<X> {
  fn multi_gpu_broadcast(self) -> Ring<Val<X>>;
}

pub trait MultiGPUBatchSplitExt<X> {
  fn multi_gpu_batch_split(self) -> Ring<Val<X>>;
}

pub trait MultiGPUSpaceSplit1dExt<X> {
  fn multi_gpu_space_split_1d(self, axis: isize) -> Ring<Val<X>>;
}

pub trait MultiGPUSpaceSplit2dExt<X> {
  fn multi_gpu_space_split_2d(self, axes: [isize; 2]) -> Torus<Val<X>>;
}

pub struct GPUMuxOp<A> {
  pub dev:  GPUDeviceId,
  pub val:  Val<A>,
}

impl<A> GPUMuxOp<A> where A: 'static {
  pub fn build_ext() -> OpExt<GPUMuxOp<A>, A> {
    let ext = OpExt{
      make_val: {
        Box::new(move |state: RefMut<Self>| {
          println!("DEBUG: GPUMuxOp: ext: make_val");
          let ctx = implicit_ctx().multi_gpu().gpu(state.dev);
          let guard = push_ctx(ctx);
          state.val._make_value()
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<Self>, _output: OVal<A>| {
          println!("DEBUG: GPUMuxOp: ext: apply");
          let ctx = implicit_ctx().multi_gpu().gpu(state.dev);
          let guard = push_ctx(ctx);
          state.val._apply(txn)
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |pass: Pass, state: RefMut<Self>, feedfwd: &mut FeedFwd| {
          let guard = push_wrapper(GPUMuxWrap{dev: state.dev});
          state.val._push_tangent(pass, feedfwd)
        })
      }),
      adjoint: Some({
        Box::new(move |pass: Pass, _this: Val<A>, state: RefMut<Self>, sink: &mut Sink| {
          let guard = push_wrapper(GPUMuxWrap{dev: state.dev});
          state.val._pop_adjoint(pass, sink);
        })
      }),
      inplace: None,
    };
    ext
  }
}

impl VectorizeOpExt<GPUDeviceArray1d<f32>> for VectorizeOp {
  fn build(src: NodeVec) -> Val<GPUDeviceArray1d<f32>> {
    let ext = OpExt{
      make_val: {
        let src = src.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let src = src.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let src = src.clone();
            let count = src.serialize_vec(txn, &mut ());
            println!("DEBUG: VectorizeOp: len: {} count: {}", src.len(), count);
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = GPUDeviceArray1d::zeros(count, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        //let section = GPULazyAsyncSection::default();
        let src = src.clone();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let mut x = output.get_mut(txn, token);
            match cap {
              WriteCap::Assign => {
                let count = src.serialize_vec(txn, &mut *x);
                assert_eq!(count, x.as_view().flat_size());
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    let mut op = FSrcOp::new(VectorizeOp, ext);
    op._extend_deps(&src);
    Val::from(Rc::new(op))
  }
}

impl DevectorizeOpExt<GPUDeviceArray1d<f32>> for DevectorizeOp {
  fn build(x_: Val<GPUDeviceArray1d<f32>>, dst: NodeVec) -> Val<()> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          RWVal::from(Arc::new(move |txn: Txn| {
            ()
          }))
        })
      },
      apply: {
        //let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let dst = dst.clone();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<()>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let x = x_.get(txn);
            match cap {
              WriteCap::Assign => {
                let count = dst.deserialize_vec(txn, &*x);
                assert_eq!(count, x.as_view().flat_size());
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::new(Rc::new(F1Op::new(DevectorizeOp, ext, x_)))
  }
}

impl DequantizeOpExt<f32, GPUDeviceOuterBatchArray3d<u8>, GPUDeviceOuterBatchArray3d<f32>> for DequantizeOp<f32> {
  fn build(lo: f32, hi: f32, x_: Val<GPUDeviceOuterBatchArray3d<u8>>) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    DequantizeOp::<f32>::build_device_u8_to_f32_obatch_3d_op(lo, hi, x_)
  }
}

impl DequantizeOpExt<f32, GPUDeviceOuterBatchArray4d<u8>, GPUDeviceOuterBatchArray4d<f32>> for DequantizeOp<f32> {
  fn build(lo: f32, hi: f32, x_: Val<GPUDeviceOuterBatchArray4d<u8>>) -> Val<GPUDeviceOuterBatchArray4d<f32>> {
    DequantizeOp::<f32>::build_device_u8_to_f32_obatch_4d_op(lo, hi, x_)
  }
}

impl DequantizeOp<f32> {
  pub fn build_device_u8_to_f32_obatch_4d_op(lo: f32, hi: f32, x_: Val<GPUDeviceOuterBatchArray4d<u8>>) -> Val<GPUDeviceOuterBatchArray4d<f32>> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = GPUDeviceOuterBatchArray4d::zeros(x.size(), x.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            assert_eq!(y.size(), x.size());
            y.set_batch_size(x.batch_size());
            match cap {
              WriteCap::Assign => {
                if x.is_packed() && y.is_packed() {
                  let x = x.flat_view().unwrap();
                  let mut y = y.flat_view_mut().unwrap();
                  assert_eq!(x.size(), y.size());
                  let mut stream = conn.cuda_stream();
                  unsafe { anode_gpu_dequantize_u8_packed_f32(
                      sz2uint(x.size()),
                      lo,
                      hi,
                      x.as_dptr(),
                      y.as_mut_dptr(),
                      conn.cuda_kernel_config() as *const _,
                      stream.as_mut_ptr(),
                  ) };
                } else {
                  unimplemented!();
                }
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
            double_check_scalar::<Self, _>(|| {
              //println!("DEBUG: DequantizeOp: double checking: len: {} {}", y.flat_size(), y.flat_view().unwrap().size());
              y.flat_view().unwrap().sync_vector_norm(conn)
            });
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(DequantizeOp{lo, hi}, ext, x_)))
  }

  pub fn build_device_u8_to_f32_obatch_3d_op(lo: f32, hi: f32, x_: Val<GPUDeviceOuterBatchArray3d<u8>>) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = GPUDeviceOuterBatchArray3d::zeros(x.size(), x.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            assert_eq!(y.size(), x.size());
            y.set_batch_size(x.batch_size());
            match cap {
              WriteCap::Assign => {
                if x.is_packed() && y.is_packed() {
                  let x = x.flat_view().unwrap();
                  let mut y = y.flat_view_mut().unwrap();
                  let mut stream = conn.cuda_stream();
                  unsafe { anode_gpu_dequantize_u8_packed_f32(
                      sz2uint(x.size()),
                      lo,
                      hi,
                      x.as_dptr(),
                      y.as_mut_dptr(),
                      conn.cuda_kernel_config() as *const _,
                      stream.as_mut_ptr(),
                  ) };
                } else {
                  unimplemented!();
                }
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
            double_check_scalar::<Self, _>(|| {
              //println!("DEBUG: DequantizeOp: double checking: len: {} {}", y.flat_size(), y.flat_view().unwrap().size());
              y.flat_view().unwrap().sync_vector_norm(conn)
            });
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(DequantizeOp{lo, hi}, ext, x_)))
  }
}

impl<A> SrcOpExt<A, Rc<Fn(Txn, GPUDeviceConn) -> A>> for SrcOp
where A: GPUDeviceAsync + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: SrcOpExt: init gpu...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: SrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //if let Some(_) = output.write(txn) {
          output.write(txn, |_, _| {
            panic!("WARNING: SrcOpExt: should never write");
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(SrcOp, ext)))
  }
}

impl<A> TouchSrcOpExt<A, Rc<Fn(Txn, GPUDeviceConn) -> A>> for TouchSrcOp
where A: GPUDeviceAsync + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> A>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: TouchSrcOpExt: init gpu...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: TouchSrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //if let Some((_, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            // No-op, do nothing.
            let _ = output.get_mut(txn, token);
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(TouchSrcOp, ext)))
  }
}

impl<T, A, F> RandomBitsSrcOpExt<A, Rc<F>> for RandomBitsSrcOp
//where A: GPUDeviceAsync + AsViewMut + 'static,
where T: Copy,
      A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
          + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
          + GPUDeviceAsync
          + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> A) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<A> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: RandomBitsSrcOpExt: init gpu...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: RandomBitsSrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let rng_seed = LazyConst::default();
        let rng_offset = TCell::new(0_u64);
        let rng = LazyCurandGenerator::default_shared_local();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            println!("DEBUG: RandomBitsSrcOpExt: apply: writing...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                let mut flat_y = y.flat_view_mut().unwrap();
                let n_elems = flat_y.size();
                rng_offset.rollback(txn);
                let prev_offset = rng_offset.propose(txn, |x| x + n_elems as u64);
                let status = rng.borrow_mut().set_seed(rng_seed.set_once(|| implicit_ctx().slow_rng().gen()));
                assert!(status.is_ok());
                println!("DEBUG: RandomBitsSrcOpExt: apply:   set offset: {}", prev_offset);
                let status = rng.borrow_mut().set_offset(prev_offset);
                assert!(status.is_ok());
                flat_y.fill_random(&mut *rng.borrow_mut(), conn);
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(SrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceScalar<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(_: Val<GPUDeviceScalar<T>>) -> Val<GPUDeviceScalar<T>> {
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceScalar<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let y = GPUDeviceScalar::zeros((), conn);
          y
        })
    )
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceScalar<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceScalar<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>) -> Val<GPUDeviceScalar<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceScalar>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceScalar>: make_val: allocating...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceScalar<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceScalar>: apply: writing...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceScalar<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceArray1d<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceArray1d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let y = GPUDeviceArray1d::zeros(x_.get(txn).size(), conn);
          y
        })
    )
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray1d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: make_val: allocating...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: apply: writing...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceArray2d<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray2d<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceArray2d<T>>) -> Val<GPUDeviceArray2d<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceArray2d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let y = GPUDeviceArray2d::zeros(x_.get(txn).size(), conn);
          y
        })
    )
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceArray2d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray2d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray2d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray2d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray2d<T>>) -> Val<GPUDeviceArray2d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray2d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                //println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray2d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceArray4d<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray4d<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceArray4d<T>>) -> Val<GPUDeviceArray4d<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceArray4d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let y = GPUDeviceArray4d::zeros(x_.get(txn).size(), conn);
          y
        })
    )
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceArray4d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray4d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray4d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray4d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray4d<T>>) -> Val<GPUDeviceArray4d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray4d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                //println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray4d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceOuterBatchScalar<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceOuterBatchScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchScalar<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceOuterBatchScalar<T>>) -> Val<GPUDeviceOuterBatchScalar<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceOuterBatchScalar<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let x = x_.get(txn);
          let y = GPUDeviceOuterBatchScalar::zeros((), x.max_batch_size(), conn);
          y
        })
    )
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchScalar<T>, usize> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      GPUDeviceOuterBatchScalar<T>: ZerosInit<usize, RValue=Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchScalar<T>>>,
{
  fn build(batch_sz: usize) -> Val<GPUDeviceOuterBatchScalar<T>> {
    zeros(<GPUDeviceOuterBatchScalar<T> as ZerosInit<_>>::zeros_init(batch_sz))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceOuterBatchScalar<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchScalar<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceOuterBatchScalar<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceOuterBatchScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchScalar<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchScalar<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchScalar<T>>) -> Val<GPUDeviceOuterBatchScalar<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchScalar<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                //println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceOuterBatchScalar<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceOuterBatchArray1d<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceOuterBatchArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray1d<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceOuterBatchArray1d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let x = x_.get(txn);
          let y = GPUDeviceOuterBatchArray1d::zeros(x.size(), x.max_batch_size(), conn);
          y
        })
    )
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchArray1d<T>, (usize, usize)> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      GPUDeviceOuterBatchArray1d<T>: ZerosInit<(usize, usize), RValue=Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray1d<T>>>,
{
  fn build(shape: (usize, usize)) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    zeros(<GPUDeviceOuterBatchArray1d<T> as ZerosInit<_>>::zeros_init(shape))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceOuterBatchArray1d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray1d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceOuterBatchArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray1d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray1d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                //println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceOuterBatchArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceOuterBatchArray3d<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceOuterBatchArray3d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let x = x_.get(txn);
          let y = GPUDeviceOuterBatchArray3d::zeros(x.size(), x.max_batch_size(), conn);
          y
        })
    )
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchArray3d<T>, ([usize; 3], usize)> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      GPUDeviceOuterBatchArray3d<T>: ZerosInit<([usize; 3], usize), RValue=Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>>,
{
  fn build(shape: ([usize; 3], usize)) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    zeros(<GPUDeviceOuterBatchArray3d<T> as ZerosInit<_>>::zeros_init(shape))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                //println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceOuterBatchArray3d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> ZerosSrcOpLikeExt<GPUDeviceOuterBatchArray4d<T>> for ZerosSrcOp
where ZerosSrcOp: ZerosSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>>,
      T: ZeroBits + Copy + 'static,
{
  fn build_like(x_: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    let x_ = x_.clone();
    <ZerosSrcOp as ZerosSrcOpExt<GPUDeviceOuterBatchArray4d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          // TODO: async section.
          let x = x_.get(txn);
          let y = GPUDeviceOuterBatchArray4d::zeros(x.size(), x.max_batch_size(), conn);
          y
        })
    )
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchArray4d<T>, ([usize; 4], usize)> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      GPUDeviceOuterBatchArray4d<T>: ZerosInit<([usize; 4], usize), RValue=Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>>,
{
  fn build(shape: ([usize; 4], usize)) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    zeros(<GPUDeviceOuterBatchArray4d<T> as ZerosInit<_>>::zeros_init(shape))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray4d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                //println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {
                let _ = output.get_mut(txn, token);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceOuterBatchArray4d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T> OnesSrcOpMaybeExt<GPUDeviceScalar<T>> for OnesSrcOp
where T: ZeroBits + Zero + One + Copy + 'static,
      OnesSrcOp: OnesSrcOpLikeExt<GPUDeviceScalar<T>>,
{
  fn maybe_build_like(x_: Val<GPUDeviceScalar<T>>) -> Option<Val<GPUDeviceScalar<T>>> {
    Some(<Self as OnesSrcOpLikeExt<GPUDeviceScalar<T>>>::build_like(x_))
  }
}

impl<T> OnesSrcOpLikeExt<GPUDeviceScalar<T>> for OnesSrcOp
where T: ZeroBits + Zero + One + Copy + 'static,
      OnesSrcOp: OnesSrcOpExt<GPUDeviceScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>>,
{
  fn build_like(_: Val<GPUDeviceScalar<T>>) -> Val<GPUDeviceScalar<T>> {
    <OnesSrcOp as OnesSrcOpExt<GPUDeviceScalar<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let y = GPUDeviceScalar::zeros((), conn);
          y
        })
    )
  }
}

impl<T, F> OnesSrcOpExt<GPUDeviceScalar<T>, Rc<F>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceScalar<T>> {
    <Self as OnesSrcOpExt<GPUDeviceScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>>>::build(init_val)
  }
}

impl<T> OnesSrcOpExt<GPUDeviceScalar<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceScalar<T>>) -> Val<GPUDeviceScalar<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceScalar>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceScalar>: make_val: allocating...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceScalar<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceScalar>: apply: writing...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                //println!("DEBUG: OnesSrcOp: apply: assign");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_constant(T::one(), conn);
              }
              WriteCap::Accumulate => {
                //println!("DEBUG: OnesSrcOp: apply: accumulate");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().add_constant_inplace(T::one(), conn);
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceScalar<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(OnesSrcOp, ext)))
  }
}

impl<T, F> OnesSrcOpExt<GPUDeviceArray1d<T>, Rc<F>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray1d<T>> {
    <Self as OnesSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>>>::build(init_val)
  }
}

impl<T> OnesSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceArray1d>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceArray1d>: make_val: allocating...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceArray1d>: apply: writing...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_constant(T::one(), conn);
              }
              WriteCap::Accumulate => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().add_constant_inplace(T::one(), conn);
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(OnesSrcOp, ext)))
  }
}

impl<T> OnesSrcOpMaybeExt<GPUDeviceOuterBatchArray3d<T>> for OnesSrcOp
where T: ZeroBits + Zero + One + Copy + 'static,
      OnesSrcOp: OnesSrcOpLikeExt<GPUDeviceOuterBatchArray3d<T>>,
{
  fn maybe_build_like(x_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Option<Val<GPUDeviceOuterBatchArray3d<T>>> {
    Some(<Self as OnesSrcOpLikeExt<GPUDeviceOuterBatchArray3d<T>>>::build_like(x_))
  }
}

impl<T> OnesSrcOpLikeExt<GPUDeviceOuterBatchArray3d<T>> for OnesSrcOp
where T: ZeroBits + Zero + One + Copy + 'static,
      OnesSrcOp: OnesSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>>,
{
  fn build_like(x_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    <OnesSrcOp as OnesSrcOpExt<GPUDeviceOuterBatchArray3d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let x = x_.get(txn);
          let y = GPUDeviceOuterBatchArray3d::zeros(x.size(), x.max_batch_size(), conn);
          y
        })
    )
  }
}

impl<T, F> OnesSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<F>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    <Self as OnesSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>>>::build(init_val)
  }
}

impl<T> OnesSrcOpExt<GPUDeviceOuterBatchArray3d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceOuterBatchArray3d>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceOuterBatchArray3d>: make_val: allocating...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceOuterBatchArray3d>: apply: writing...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.flat_view_mut().unwrap().set_constant(T::one(), conn);
              }
              WriteCap::Accumulate => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.flat_view_mut().unwrap().add_constant_inplace(T::one(), conn);
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceOuterBatchArray3d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(OnesSrcOp, ext)))
  }
}

impl<T> OnesSrcOpMaybeExt<GPUDeviceOuterBatchArray4d<T>> for OnesSrcOp
where T: ZeroBits + Zero + One + Copy + 'static,
      OnesSrcOp: OnesSrcOpLikeExt<GPUDeviceOuterBatchArray4d<T>>,
{
  fn maybe_build_like(x_: Val<GPUDeviceOuterBatchArray4d<T>>) -> Option<Val<GPUDeviceOuterBatchArray4d<T>>> {
    Some(<Self as OnesSrcOpLikeExt<GPUDeviceOuterBatchArray4d<T>>>::build_like(x_))
  }
}

impl<T> OnesSrcOpLikeExt<GPUDeviceOuterBatchArray4d<T>> for OnesSrcOp
where T: ZeroBits + Zero + One + Copy + 'static,
      OnesSrcOp: OnesSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>>,
{
  fn build_like(x_: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    <OnesSrcOp as OnesSrcOpExt<GPUDeviceOuterBatchArray4d<T>, _>>::build(
        Rc::new(move |txn, conn| {
          let x = x_.get(txn);
          let y = GPUDeviceOuterBatchArray4d::zeros(x.size(), x.max_batch_size(), conn);
          y
        })
    )
  }
}

impl<T, F> OnesSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<F>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
      F: (Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    <Self as OnesSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>>>::build(init_val)
  }
}

impl<T> OnesSrcOpExt<GPUDeviceOuterBatchArray4d<T>, Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>> for OnesSrcOp
where T: Zero + One + Copy + 'static,
{
  fn build(init_val: Rc<Fn(Txn, GPUDeviceConn) -> GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceOuterBatchArray4d>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceOuterBatchArray4d>: make_val: allocating...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(txn, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray4d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: OnesSrcOpExt<|| GPUDeviceOuterBatchArray4d>: apply: writing...");
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.flat_view_mut().unwrap().set_constant(T::one(), conn);
              }
              WriteCap::Accumulate => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.flat_view_mut().unwrap().add_constant_inplace(T::one(), conn);
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceOuterBatchArray4d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = this.adjoint(sink) {
            // Do nothing.
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(OnesSrcOp, ext)))
  }
}

impl<T: ZeroBits + Copy + 'static> FlattenExt<GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray1d<T>> for Val<GPUDeviceOuterBatchArray3d<T>> {
  fn flatten(self) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    ReshapeOp::build_device_obatch_3d_to_1d_op::<T>(self)
  }
}

impl ReshapeOp {
  pub fn build_device_obatch_3d_to_1d_op<T>(x_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>>
  where T: ZeroBits + Copy + 'static,
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = GPUDeviceOuterBatchArray1d::zeros(x.flat_size(), x.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            assert_eq!(y.size(), x.flat_size());
            assert_eq!(y.max_batch_size(), x.max_batch_size());
            match cap {
              WriteCap::Assign => {
                if x.is_packed() && y.is_packed() {
                  y.flat_view_mut().unwrap().copy(x.flat_view().unwrap(), conn);
                } else {
                  // TODO
                  unimplemented!();
                }
              }
              WriteCap::Accumulate => {
                if x.is_packed() && y.is_packed() {
                  y.flat_view_mut().unwrap().add(x.flat_view().unwrap(), conn);
                } else {
                  // TODO
                  unimplemented!();
                }
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = ReshapeLikeOp::build_device_obatch_1d_to_3d_op(adj_y_, x_.clone());
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(ReshapeOp, ext, x_)))
  }
}

impl ReshapeLikeOp {
  pub fn build_device_obatch_1d_to_3d_op<T>(x_: Val<GPUDeviceOuterBatchArray1d<T>>, target_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>>
  where T: ZeroBits + Copy + 'static,
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        let target_ = target_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          let target_ = target_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let target = target_.get(txn);
            guard._wait(x.async_state());
            guard._wait(target.async_state());
            assert_eq!(x.size(), target.flat_size());
            assert_eq!(x.max_batch_size(), target.max_batch_size());
            let y = GPUDeviceOuterBatchArray3d::zeros(target.size(), target.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let target_ = target_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let target = target_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(target.async_state());
            guard._wait(y.async_state());
            assert_eq!(y.flat_size(), x.flat_size());
            assert_eq!(y.size(), target.size());
            assert_eq!(y.max_batch_size(), x.max_batch_size());
            assert_eq!(y.max_batch_size(), target.max_batch_size());
            match cap {
              WriteCap::Assign => {
                if x.is_packed() && y.is_packed() {
                  y.flat_view_mut().unwrap().copy(x.flat_view().unwrap(), conn);
                } else {
                  // TODO
                  unimplemented!();
                }
              }
              WriteCap::Accumulate => {
                if x.is_packed() && y.is_packed() {
                  y.flat_view_mut().unwrap().add(x.flat_view().unwrap(), conn);
                } else {
                  // TODO
                  unimplemented!();
                }
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = ReshapeOp::build_device_obatch_3d_to_1d_op(adj_y_);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(ReshapeLikeOp, ext, x_, target_)))
  }
}

impl<T: ZeroBits + 'static> SumJoinOpExt<GPUDeviceScalar<T>> for SumJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceScalar<T>>>) -> Val<GPUDeviceScalar<T>> {
    //println!("DEBUG: SumJoinOp: build");
    SumJoinOp::build_device_op(xs_)
  }

  fn build_inplace(xs_: Vec<Val<GPUDeviceScalar<T>>>) -> (Val<GPUDeviceScalar<T>>, Vec<Val<GPUDeviceScalar<T>>>) {
    //println!("DEBUG: SumJoinOp: build inplace");
    SumJoinOp::build_inplace_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> SumJoinOpExt<GPUDeviceArray1d<T>> for SumJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceArray1d<T>>>) -> Val<GPUDeviceArray1d<T>> {
    //println!("DEBUG: SumJoinOp: build");
    SumJoinOp::build_device_op(xs_)
  }

  fn build_inplace(xs_: Vec<Val<GPUDeviceArray1d<T>>>) -> (Val<GPUDeviceArray1d<T>>, Vec<Val<GPUDeviceArray1d<T>>>) {
    //println!("DEBUG: SumJoinOp: build inplace");
    SumJoinOp::build_inplace_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> SumJoinOpExt<GPUDeviceOuterBatchArray1d<T>> for SumJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceOuterBatchArray1d<T>>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    //println!("DEBUG: SumJoinOp: build");
    SumJoinOp::build_device_op(xs_)
  }

  fn build_inplace(xs_: Vec<Val<GPUDeviceOuterBatchArray1d<T>>>) -> (Val<GPUDeviceOuterBatchArray1d<T>>, Vec<Val<GPUDeviceOuterBatchArray1d<T>>>) {
    //println!("DEBUG: SumJoinOp: build inplace");
    SumJoinOp::build_inplace_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> SumJoinOpExt<GPUDeviceOuterBatchArray3d<T>> for SumJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceOuterBatchArray3d<T>>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    //println!("DEBUG: SumJoinOp: build");
    SumJoinOp::build_device_op(xs_)
  }

  fn build_inplace(xs_: Vec<Val<GPUDeviceOuterBatchArray3d<T>>>) -> (Val<GPUDeviceOuterBatchArray3d<T>>, Vec<Val<GPUDeviceOuterBatchArray3d<T>>>) {
    //println!("DEBUG: SumJoinOp: build inplace");
    SumJoinOp::build_inplace_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> SumJoinOpExt<GPUDeviceOuterBatchArray4d<T>> for SumJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceOuterBatchArray4d<T>>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    SumJoinOp::build_device_op(xs_)
  }

  fn build_inplace(xs_: Vec<Val<GPUDeviceOuterBatchArray4d<T>>>) -> (Val<GPUDeviceOuterBatchArray4d<T>>, Vec<Val<GPUDeviceOuterBatchArray4d<T>>>) {
    SumJoinOp::build_inplace_device_op(xs_)
  }
}

impl SumJoinOp {
  pub fn build_device_op<T, A>(xs_: Vec<Val<A>>) -> Val<A>
  where T: ZeroBits + 'static/* + Zero + One*/,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
  {
    let ext = OpExt{
      make_val: {
        let xs_ = xs_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let xs_ = xs_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x0 = xs_[0].get(txn);
            guard._wait(x0.async_state());
            let y = A::zeros_shape(x0.shape(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let xs_ = xs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let mut y = match output.get_mut(txn, token).flat_view_mut() {
              None => panic!(),
              Some(y) => y,
            };
            guard._wait(y.async_state());
            let x0 = match xs_[0].get(txn).flat_view() {
              None => panic!(),
              Some(x) => x,
            };
            guard._wait(x0.async_state());
            match cap {
              WriteCap::Assign => {
                y.copy(x0, conn.clone());
              }
              WriteCap::Accumulate => {
                y.add(x0, conn.clone());
              }
            }
            for i in 1 .. xs_.len() {
              let x = match xs_[i].get(txn).flat_view() {
                None => panic!(),
                Some(x) => x,
              };
              guard._wait(x.async_state());
              y.add(x, conn.clone());
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let xs_ = xs_.clone();
        Box::new(move |_: Pass, y_: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            for i in 0 .. xs_.len() {
              xs_[i].put_adjoint(adj_y_.clone(), sink);
            }
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FJoinOp::new(SumJoinOp, ext, xs_)))
  }

  pub fn build_inplace_device_op<T, A>(old_xs_: Vec<Val<A>>) -> (Val<A>, Vec<Val<A>>)
  where T: ZeroBits + 'static,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
  {
    let mut new_xs_ = Vec::with_capacity(old_xs_.len());
    let new_x0_ = old_xs_[0].accumulate();
    //let new_x0_ = old_xs_[0].clone().pass().accumulate();
    new_xs_.push(new_x0_.clone());
    for i in 1 .. old_xs_.len() {
      new_xs_.push(old_xs_[i].accumulate_value(new_x0_._static_value()));
      //new_xs_.push(old_xs_[i].clone().pass().accumulate_value(new_x0_._static_value()));
    }
    let ext = OpExt{
      make_val: {
        let new_x0_ = new_x0_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          //unreachable!();
          new_x0_._make_value()
        })
      },
      apply: {
        // FIXME(peter, 20180622): may need to do a "join pass" here.
        let section = GPULazyAsyncSection::default();
        let xs_ = new_xs_.clone();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<A>| {
          for x_ in xs_.iter() {
            if !output._valref().is_none() && x_._valref() != output._valref() {
              println!("WARNING: SumJoinAccumulateOp: apply: possibly incorrect result");
            }
          }
          //if let Some((_, token)) = output.write(txn) {
          output.write(txn, |_, token| {
            output.finish_write(txn, token);
          })
        })
        //pass_apply(new_x0_.clone())
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        // FIXME(peter, 20180623): This is generally wrong. The current
        // workaround to get the correct adjoint in some cases is to put
        // the first argument in a `PassOp`.
        //let xs_ = old_xs_.clone();
        let xs_ = new_xs_.clone();
        Box::new(move |_: Pass, y_: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          println!("WARNING: SumJoinAccumulateOp: attempting to build in-place adjoint; please exercise care as correctness is not yet guaranteed.");
          if let Some(adj_y_) = y_.adjoint(sink) {
            for i in 0 .. xs_.len() {
              xs_[i].put_adjoint(adj_y_.clone(), sink);
            }
          }
        })
      }),
      inplace: None,
    };
    (Val::with_value_mode(Rc::new(FJoinOp::new(SumJoinAccumulateOp, ext, new_xs_.clone())), new_x0_._static_value(), WriteMode::Accumulate), new_xs_)
  }

  /*pub fn build_device_batch_op<T, A>(inputs_: Vec<Val<A>>) -> Val<A>
  where T: Copy /*+ Zero + One*/ + 'static,
        //A: GPUDeviceBatchArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
        A: GPUDeviceBatchArrayZeros<T> + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>> + 'static,
        //A: GPUDeviceBatchArrayZeros + GPUFlatViewMut<T> + 'static,
  {
    let ext = OpExt{
      make_val: {
        let inputs_ = inputs_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          //let x0 = inputs_[0].value();
          let inputs_ = inputs_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let x0_size = inputs_[0].get(txn).size();
            let x0_batch_sz = inputs_[0].get(txn).batch_size();
            A::zeros(x0_size, x0_batch_sz, conn)
          }))
        })
      },
      apply: {
        //let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        let inputs_ = inputs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //let inputs_ = inputs_.clone();
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let batch_sz0 = inputs_[0].get(txn).batch_size();
            output.get_mut(txn, token).set_batch_size(batch_sz0);
            let mut y = match output.get_mut(txn, token).flat_view_mut() {
              None => panic!(),
              Some(y) => y,
            };
            let x0 = match inputs_[0].get(txn).flat_view() {
              None => panic!(),
              Some(x) => x,
            };
            match cap {
              WriteCap::Assign => {
                y.copy(x0, conn.clone());
              }
              WriteCap::Accumulate => {
                y.add(x0, conn.clone());
              }
            }
            for i in 1 .. inputs_.len() {
              let batch_sz = inputs_[i].get(txn).batch_size();
              assert_eq!(batch_sz, batch_sz0);
              let x = match inputs_[i].get(txn).flat_view() {
                None => panic!(),
                Some(x) => x,
              };
              y.add(x, conn.clone());
            }
          }
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let inputs_ = inputs_.clone();
        Box::new(move |_: Pass, y_: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            for i in 0 .. inputs_.len() {
              inputs_[i].put_adjoint(adj_y_.clone(), sink);
            }
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_)))
  }*/
}

impl<T: ZeroBits + 'static> ProductJoinOpExt<GPUDeviceScalar<T>> for ProductJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceScalar<T>>>) -> Val<GPUDeviceScalar<T>> {
    ProductJoinOp::build_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> ProductJoinOpExt<GPUDeviceArray1d<T>> for ProductJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceArray1d<T>>>) -> Val<GPUDeviceArray1d<T>> {
    ProductJoinOp::build_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> ProductJoinOpExt<GPUDeviceOuterBatchArray1d<T>> for ProductJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceOuterBatchArray1d<T>>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    ProductJoinOp::build_device_op(xs_)
  }
}

impl<T: ZeroBits + 'static> ProductJoinOpExt<GPUDeviceOuterBatchArray3d<T>> for ProductJoinOp
{
  fn build(xs_: Vec<Val<GPUDeviceOuterBatchArray3d<T>>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    ProductJoinOp::build_device_op(xs_)
  }
}

impl ProductJoinOp {
  pub fn build_device_op<T, A>(xs_: Vec<Val<A>>) -> Val<A>
  where T: ZeroBits + 'static/* + Zero + One*/,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
  {
    let ext = OpExt{
      make_val: {
        let xs_ = xs_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let xs_ = xs_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x0 = xs_[0].get(txn);
            guard._wait(x0.async_state());
            let y = A::zeros_shape(x0.shape(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let xs_ = xs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let mut y = match output.get_mut(txn, token).flat_view_mut() {
              None => panic!(),
              Some(y) => y,
            };
            guard._wait(y.async_state());
            match cap {
              WriteCap::Assign => {
                let x0 = match xs_[0].get(txn).flat_view() {
                  None => panic!(),
                  Some(x) => x,
                };
                guard._wait(x0.async_state());
                y.copy(x0, conn.clone());
                for i in 1 .. xs_.len() {
                  let x = match xs_[i].get(txn).flat_view() {
                    None => panic!(),
                    Some(x) => x,
                  };
                  guard._wait(x.async_state());
                  y.mult(x, conn.clone());
                }
              }
              WriteCap::Accumulate => {
                let x0 = xs_[0].get(txn);
                guard._wait(x0.async_state());
                let mut workspace = A::zeros_shape_with_alloc(conn.burst_arena(), x0.shape(), conn.clone());
                guard._wait(workspace.async_state());
                workspace.flat_view_mut().unwrap().copy(x0.flat_view().unwrap(), conn.clone());
                for i in 1 .. xs_.len() {
                  let x = xs_[i].get(txn);
                  guard._wait(x.async_state());
                  workspace.flat_view_mut().unwrap().mult(x.flat_view().unwrap(), conn.clone());
                }
                y.add(workspace.flat_view().unwrap(), conn.clone());
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let xs_ = xs_.clone();
        Box::new(move |_: Pass, y_: Val<A>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            for i in 0 .. xs_.len() {
              let mut adj_x_i_factors = Vec::with_capacity(xs_.len());
              for j in 0 .. xs_.len() {
                if i == j {
                  adj_x_i_factors.push(adj_y_.clone());
                } else {
                  adj_x_i_factors.push(xs_[j].clone());
                }
              }
              xs_[i].put_adjoint(ProductJoinOp::build_device_op(adj_x_i_factors), sink);
            }
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FJoinOp::new(ProductJoinOp, ext, xs_)))
  }
}

impl BatchMean2dOpExt<f32, GPUDeviceOuterBatchArray3d<f32>, GPUDeviceArray1d<f32>> for BatchMean2dOp {
  fn build(axes: [isize; 2], x_: Val<GPUDeviceOuterBatchArray3d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    BatchMean2dOp::build_device_f32_op(axes, x_)
  }
}

impl BatchVariance2dOpExt<f32, GPUDeviceOuterBatchArray3d<f32>, GPUDeviceArray1d<f32>> for BatchVariance2dOp {
  fn build(axes: [isize; 2], epsilon: f32, x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    BatchVariance2dOp::build_device_f32_op(axes, epsilon, x_, mean_)
  }
}

impl BatchNormalize2dOpExt<f32, GPUDeviceOuterBatchArray3d<f32>, GPUDeviceArray1d<f32>> for BatchNormalize2dOp {
  fn build(axes: [isize; 2], x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>, var_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    BatchNormalize2dOp::build_device_f32_op(axes, x_, mean_, var_)
  }
}

impl BatchMean2dOp {
  pub fn build_device_f32_op(axes: [isize; 2], x_: Val<GPUDeviceOuterBatchArray3d<f32>>) -> Val<GPUDeviceArray1d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x_size = x_.get(txn).size();
            // TODO: assuming NCHW layout.
            let y = GPUDeviceArray1d::zeros(x_size[2], conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mut mean = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(mean.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = x.size();
            let x_bsz = x.batch_size();
            match cap {
              WriteCap::Assign => {
                // TODO
                /*let max_block_ct = conn.cuda_kernel_config().max_block_ct as usize;
                if x_size[2] < max_block_ct {
                  /*println!("WARNING: BatchMean2dOp: operation may be slow: {} {}",
                      x_size[2], max_block_ct);*/
                  let tile_sz = (max_block_ct + x_size[2] - 1) / x_size[2];
                  assert!(tile_sz >= 1);
                  mean.as_view_mut().set_zeros(conn.clone());
                  let mut stream = conn.cuda_stream();
                  unsafe { anode_gpu_batch_mean_3d1_packed_accumulate_tiledatomic_f32(
                      sz2uint(x_size[0] * x_size[1]),
                      sz2uint(x_size[2]),
                      sz2uint(tile_sz),
                      sz2uint(x_bsz),
                      x.as_view().as_dptr(),
                      mean.as_view_mut().as_mut_dptr(),
                      conn.cuda_kernel_config() as *const _,
                      stream.as_mut_ptr(),
                  ) };
                } else {*/
                /*}*/
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_mean_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    x.as_view().as_dptr(),
                    mean.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, mean_: Val<GPUDeviceArray1d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_mean_) = mean_.adjoint(sink) {
            //println!("DEBUG: BatchMean2dOp: found adjoint for primal: {:?} => {:?}", mean_._graph_key(), adj_mean_._graph_key());
            let adj_x_ = BatchMean2dBwdOp::build_device_f32_op(axes, adj_mean_.clone(), x_.clone());
            x_.put_adjoint(adj_x_, sink);
          //} else {
            //println!("WARNING: BatchMean2dOp: missing adjoint for primal: {:?}", mean_._graph_key());
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(BatchMean2dOp, ext, x_)))
  }
}

impl BatchMean2dBwdOp {
  pub fn build_device_f32_op(axes: [isize; 2], dmean_: Val<GPUDeviceArray1d<f32>>, x_: Val<GPUDeviceOuterBatchArray3d<f32>>) -> Val<GPUDeviceOuterBatchArray3d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_size = x.size();
            let x_max_bsz = x.max_batch_size();
            let y = GPUDeviceOuterBatchArray3d::zeros(x_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dmean_ = dmean_.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dmean = dmean_.get(txn);
            let x = x_.get(txn);
            let mut dx = output.get_mut(txn, token);
            guard._wait(dmean.async_state());
            guard._wait(x.async_state());
            guard._wait(dx.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = x.size();
            let x_bsz = x.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_mean_bwd_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dmean.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_mean_bwd_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dmean.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<_>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(BatchMean2dBwdOp, ext, dmean_, x_)))
  }
}

impl BatchVariance2dOp {
  pub fn build_device_f32_op(axes: [isize; 2], epsilon: f32, x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x_size = x_.get(txn).size();
            // TODO: assuming NCHW layout.
            let y = GPUDeviceArray1d::zeros(x_size[2], conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mean = mean_.get(txn);
            let mut var = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(mean.async_state());
            guard._wait(var.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = x.size();
            let x_bsz = x.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_var_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    epsilon,
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    var.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        Box::new(move |_: Pass, var_: Val<GPUDeviceArray1d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_var_) = var_.adjoint(sink) {
            let adj_x_ = BatchVariance2dBwdOp::build_device_f32_op(axes, adj_var_.clone(), x_.clone(), mean_.clone());
            let adj_mean_ = BatchVariance2dBwdMeanOp::build_device_f32_op(axes, adj_var_.clone(), x_.clone(), mean_.clone());
            x_.put_adjoint(adj_x_, sink);
            mean_.put_adjoint(adj_mean_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(BatchVariance2dOp, ext, x_, mean_)))
  }
}

impl BatchVariance2dBwdOp {
  pub fn build_device_f32_op(axes: [isize; 2], dvar_: Val<GPUDeviceArray1d<f32>>, x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceOuterBatchArray3d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_size = x.size();
            let x_max_bsz = x.max_batch_size();
            let y = GPUDeviceOuterBatchArray3d::zeros(x_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dvar_ = dvar_.clone();
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dvar = dvar_.get(txn);
            let x = x_.get(txn);
            let mean = mean_.get(txn);
            let mut dx = output.get_mut(txn, token);
            guard._wait(dvar.async_state());
            guard._wait(x.async_state());
            guard._wait(mean.async_state());
            guard._wait(dx.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = x.size();
            let x_bsz = x.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_var_bwd_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dvar.as_view().as_dptr(),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_var_bwd_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dvar.as_view().as_dptr(),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<_>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(BatchVariance2dBwdOp, ext, dvar_, x_, mean_)))
  }
}

impl BatchVariance2dBwdMeanOp {
  pub fn build_device_f32_op(axes: [isize; 2], dvar_: Val<GPUDeviceArray1d<f32>>, x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x_size = x_.get(txn).size();
            // TODO: assuming NCHW layout.
            let y = GPUDeviceArray1d::zeros(x_size[2], conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dvar_ = dvar_.clone();
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dvar = dvar_.get(txn);
            let x = x_.get(txn);
            let mean = mean_.get(txn);
            let mut dmean = output.get_mut(txn, token);
            guard._wait(dvar.async_state());
            guard._wait(x.async_state());
            guard._wait(mean.async_state());
            guard._wait(dmean.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = x.size();
            let x_bsz = x.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_var_bwd_mean_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dvar.as_view().as_dptr(),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    dmean.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_var_bwd_mean_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dvar.as_view().as_dptr(),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    dmean.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray1d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(BatchVariance2dBwdMeanOp, ext, dvar_, x_, mean_)))
  }
}

impl BatchNormalize2dOp {
  pub fn build_device_f32_op(axes: [isize; 2], x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>, var_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceOuterBatchArray3d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x_size = x_.get(txn).size();
            let x_max_batch_sz = x_.get(txn).max_batch_size();
            let y = GPUDeviceOuterBatchArray3d::zeros(x_size, x_max_batch_sz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        let var_ = var_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mean = mean_.get(txn);
            let var = var_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(mean.async_state());
            guard._wait(var.async_state());
            guard._wait(y.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = x.size();
            let x_bsz = x.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        let var_ = var_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray3d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            //println!("DEBUG: BatchNormalize2dOp: found adjoint for primal: {:?} => {:?}", y_._graph_key(), adj_y_._graph_key());
            let adj_x_ = BatchNormalize2dBwdOp::build_device_f32_op(axes, adj_y_.clone(), var_.clone());
            let adj_mean_ = BatchNormalize2dBwdMeanOp::build_device_f32_op(axes, adj_y_.clone(), var_.clone());
            let adj_var_ = BatchNormalize2dBwdVarianceOp::build_device_f32_op(axes, adj_y_.clone(), x_.clone(), mean_.clone(), var_.clone());
            x_.put_adjoint(adj_x_, sink);
            mean_.put_adjoint(adj_mean_, sink);
            var_.put_adjoint(adj_var_, sink);
          //} else {
            //println!("WARNING: BatchNormalize2dOp: missing adjoint for primal: {:?}", y_._graph_key());
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(BatchNormalize2dOp, ext, x_, mean_, var_)))
  }
}

impl BatchNormalize2dBwdOp {
  pub fn build_device_f32_op(axes: [isize; 2], dy_: Val<GPUDeviceOuterBatchArray3d<f32>>, var_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceOuterBatchArray3d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let dy_ = dy_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let dy_ = dy_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            guard._wait(dy.async_state());
            let x_size = dy.size();
            let x_max_bsz = dy.max_batch_size();
            let y = GPUDeviceOuterBatchArray3d::zeros(x_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = dy_.clone();
        let var_ = var_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            let var = var_.get(txn);
            let mut dx = output.get_mut(txn, token);
            guard._wait(dy.async_state());
            guard._wait(var.async_state());
            guard._wait(dx.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = dy.size();
            let x_bsz = dy.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<_>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(BatchNormalize2dBwdOp, ext, dy_, var_)))
  }
}

impl BatchNormalize2dBwdMeanOp {
  pub fn build_device_f32_op(axes: [isize; 2], dy_: Val<GPUDeviceOuterBatchArray3d<f32>>, var_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let dy_ = dy_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let dy_ = dy_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            guard._wait(dy.async_state());
            let x_size = dy.size();
            // TODO: assuming NCHW layout.
            let y = GPUDeviceArray1d::zeros(x_size[2], conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = dy_.clone();
        let var_ = var_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            let var = var_.get(txn);
            let mut dmean = output.get_mut(txn, token);
            guard._wait(dy.async_state());
            guard._wait(var.async_state());
            guard._wait(dmean.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = dy.size();
            let x_bsz = dy.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_mean_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dmean.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_mean_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dmean.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray1d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(BatchNormalize2dBwdMeanOp, ext, dy_, var_)))
  }
}

impl BatchNormalize2dBwdVarianceOp {
  pub fn build_device_f32_op(axes: [isize; 2], dy_: Val<GPUDeviceOuterBatchArray3d<f32>>, x_: Val<GPUDeviceOuterBatchArray3d<f32>>, mean_: Val<GPUDeviceArray1d<f32>>, var_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let dy_ = dy_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let dy_ = dy_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            guard._wait(dy.async_state());
            let x_size = dy.size();
            // TODO: assuming NCHW layout.
            let y = GPUDeviceArray1d::zeros(x_size[2], conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = dy_.clone();
        let x_ = x_.clone();
        let mean_ = mean_.clone();
        let var_ = var_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            let x = x_.get(txn);
            let mean = mean_.get(txn);
            let var = var_.get(txn);
            let mut dvar = output.get_mut(txn, token);
            guard._wait(dy.async_state());
            guard._wait(x.async_state());
            guard._wait(mean.async_state());
            guard._wait(var.async_state());
            guard._wait(dvar.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = dy.size();
            let x_bsz = dy.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_var_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dvar.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_var_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    x.as_view().as_dptr(),
                    mean.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dvar.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
            /*let max_block_ct = conn.cuda_kernel_config().max_block_ct as usize;
            if x_size[2] < max_block_ct {
              /*println!("WARNING: BatchMean2dOp: operation may be slow: {} {}",
                  x_size[2], max_block_ct);*/
              let tile_sz = (max_block_ct + x_size[2] - 1) / x_size[2];
              assert!(tile_sz >= 1);
              mean.as_view_mut().set_zeros(conn.clone());
              let mut stream = conn.cuda_stream();
              unsafe { anode_gpu_batch_norm_bwd_var_3d1_packed_accumulate_tiledatomic_f32(
                  sz2uint(x_size[0] * x_size[1]),
                  sz2uint(x_size[2]),
                  sz2uint(tile_sz),
                  sz2uint(x_bsz),
                  dy.as_view().as_dptr(),
                  x.as_view().as_dptr(),
                  mean.as_view().as_dptr(),
                  var.as_view().as_dptr(),
                  dvar.as_view_mut().as_mut_dptr(),
                  conn.cuda_kernel_config() as *const _,
                  stream.as_mut_ptr(),
              ) };
            } else {
              // TODO
            }*/
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray1d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F4Op::new(BatchNormalize2dBwdVarianceOp, ext, dy_, x_, mean_, var_)))
  }

  pub fn build_device_f32_op_v2(axes: [isize; 2], dy_: Val<GPUDeviceOuterBatchArray3d<f32>>, y_: Val<GPUDeviceOuterBatchArray3d<f32>>, var_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>>
  {
    let ext = OpExt{
      make_val: {
        let dy_ = dy_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let dy_ = dy_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            guard._wait(dy.async_state());
            let x_size = dy.size();
            // TODO: assuming NCHW layout.
            let y = GPUDeviceArray1d::zeros(x_size[2], conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = dy_.clone();
        let y_ = y_.clone();
        let var_ = var_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            let y = y_.get(txn);
            let var = var_.get(txn);
            let mut dvar = output.get_mut(txn, token);
            guard._wait(dy.async_state());
            guard._wait(y.async_state());
            guard._wait(var.async_state());
            guard._wait(dvar.async_state());
            // TODO: assuming NCHW layout.
            assert_eq!([0, 1], axes);
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = dy.size();
            let x_bsz = dy.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_var_v2_3d1_packed_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    y.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dvar.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_batch_norm_bwd_var_v2_3d1_packed_accumulate_f32(
                    sz2uint(x_size[0] * x_size[1]),
                    sz2uint(x_size[2]),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    y.as_view().as_dptr(),
                    var.as_view().as_dptr(),
                    dvar.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray1d<f32>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(BatchNormalize2dBwdVarianceOp, ext, dy_, y_, var_)))
  }
}

impl OnlineAddOpExt<f32, GPUDeviceScalar<f32>> for OnlineAddOp {
  fn build(scalar: Val<f32>, x_: Val<GPUDeviceScalar<f32>>, y_: Val<GPUDeviceScalar<f32>>) -> Val<GPUDeviceScalar<f32>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          unreachable!();
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let scalar = scalar.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceScalar<f32>>| {
          output.propose(txn, |mut y| {
            //println!("DEBUG: OnlineAverageOp: apply: writing...");
            let mut pool = implicit_ctx().gpu().pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            let r = *scalar.get(txn);
            y.as_view_mut().online_add(r, x.as_view(), conn);
          })
        })
      },
      build: Some({
        Box::new(move |_args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    let y = y_._static_value();
    Val::with_value(Rc::new(F1WrapOp::new(OnlineAddOp, ext, x_, y_)), y)
  }
}

impl OnlineAddOpExt<f32, GPUDeviceArray1d<f32>> for OnlineAddOp {
  fn build(scalar: Val<f32>, x_: Val<GPUDeviceArray1d<f32>>, y_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          unreachable!();
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let scalar = scalar.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          output.propose(txn, |mut y| {
            //println!("DEBUG: OnlineAverageOp: apply: writing...");
            let mut pool = implicit_ctx().gpu().pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            let r = *scalar.get(txn);
            y.as_view_mut().online_add(r, x.as_view(), conn);
          })
        })
      },
      build: Some({
        Box::new(move |_args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    let y = y_._static_value();
    Val::with_value(Rc::new(F1WrapOp::new(OnlineAddOp, ext, x_, y_)), y)
  }
}

impl OnlineDiscountOpExt<f32, GPUDeviceScalar<f32>> for OnlineDiscountOp {
  fn build(scalar: Val<f32>, x_: Val<GPUDeviceScalar<f32>>, y_: Val<GPUDeviceScalar<f32>>) -> Val<GPUDeviceScalar<f32>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          unreachable!();
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let scalar = scalar.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceScalar<f32>>| {
          output.propose(txn, |mut y| {
            //println!("DEBUG: OnlineAverageOp: apply: writing...");
            let mut pool = implicit_ctx().gpu().pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            let r = *scalar.get(txn);
            y.as_view_mut().online_discount(r, x.as_view(), conn);
          })
        })
      },
      build: Some({
        Box::new(move |_args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    let y = y_._static_value();
    Val::with_value(Rc::new(F1WrapOp::new(OnlineDiscountOp, ext, x_, y_)), y)
  }
}

impl OnlineDiscountOpExt<f32, GPUDeviceArray1d<f32>> for OnlineDiscountOp {
  fn build(scalar: Val<f32>, x_: Val<GPUDeviceArray1d<f32>>, y_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          unreachable!();
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let scalar = scalar.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          output.propose(txn, |mut y| {
            //println!("DEBUG: OnlineAverageOp: apply: writing...");
            let mut pool = implicit_ctx().gpu().pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            let r = *scalar.get(txn);
            y.as_view_mut().online_discount(r, x.as_view(), conn);
          })
        })
      },
      build: Some({
        Box::new(move |_args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    let y = y_._static_value();
    Val::with_value(Rc::new(F1WrapOp::new(OnlineDiscountOp, ext, x_, y_)), y)
  }
}

impl OnlineAverageOpExt<f32, GPUDeviceArray1d<f32>> for OnlineAverageOp {
  //fn build(avg_rate: TCell<f32>, x_: Val<GPUDeviceArray1d<f32>>, y_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
  fn build(avg_rate: Val<f32>, x_: Val<GPUDeviceArray1d<f32>>, y_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          unreachable!();
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let avg_rate = avg_rate.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<f32>>| {
          output.propose(txn, |mut y| {
            //println!("DEBUG: OnlineAverageOp: apply: writing...");
            let mut pool = implicit_ctx().gpu().pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            let r = *avg_rate.get(txn);
            y.as_view_mut().online_average(r, x.as_view(), conn);
          })
        })
      },
      build: Some({
        Box::new(move |_args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: None,
      adjoint: None,
      inplace: None,
    };
    //Val::from(Rc::new(F1WrapOp::new(OnlineAverageOp, ext, x_, y_)))
    let y = y_._static_value();
    Val::with_value(Rc::new(F1WrapOp::new(OnlineAverageOp, ext, x_, y_)), y)
  }
}

impl SoftmaxOpExt<GPUDeviceOuterBatchArray1d<f32>> for SoftmaxOp {
  fn build(x_: Val<GPUDeviceOuterBatchArray1d<f32>>) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    SoftmaxOp::build_device_obatch_1d_op(x_)
  }
}

impl SoftmaxOp {
  fn build_device_obatch_1d_op(x_: Val<GPUDeviceOuterBatchArray1d<f32>>) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = GPUDeviceOuterBatchArray1d::zeros(x.size(), x.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            assert_eq!(x.size(), y.size());
            y.set_batch_size(x.batch_size());
            match cap {
              WriteCap::Assign => {
                /*// TODO: assumes NCHW layout.
                let xsoftmax_shape = XSoftmaxFullShape::Softmax0d(Softmax0dFullShape{
                  src_feature_axis: 0,
                  src_batch_axis:   1,
                  src_size:         [x_size, x_bsz],
                  dst_feature_axis: 0,
                  dst_batch_axis:   1,
                  dst_size:         [x_size, x_bsz],
                });
                let mut state_cache = state_cache.borrow_mut();
                let state = state_cache.entry(xsoftmax_shape.clone()).or_insert_with(|| {
                  match query_gpu_softmax_state(conn.device(), xsoftmax_shape, conn.clone()) {
                    None => panic!("invalid softmax config"),
                    Some(state) => state,
                  }
                });
                y.as_view_mut().batch_softmax(
                    state,
                    x.as_view(),
                    conn.clone(),
                );*/
                if x.size() <= conn.cuda_kernel_config().block_sz as _ {
                  // TODO: assumes NCHW layout.
                  let mut stream = conn.cuda_stream();
                  unsafe { anode_gpu_softmax_packed_block_f32(
                      sz2uint(x.size()),
                      sz2uint(x.batch_size()),
                      x.as_view().as_dptr(),
                      y.as_view_mut().as_mut_dptr(),
                      conn.cuda_kernel_config() as *const _,
                      stream.as_mut_ptr(),
                  ) };
                } else {
                  // TODO: assumes NCHW layout.
                  let mut stream = conn.cuda_stream();
                  unsafe { anode_gpu_softmax_packed_deterministic_f32(
                      sz2uint(x.size()),
                      sz2uint(x.batch_size()),
                      x.as_view().as_dptr(),
                      y.as_view_mut().as_mut_dptr(),
                      conn.cuda_kernel_config() as *const _,
                      stream.as_mut_ptr(),
                  ) };
                }
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, y_: Val<_>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(SoftmaxOp, ext, x_)))
  }
}

impl SoftmaxCategoricalNLLOpExt<f32, GPUDeviceOuterBatchArray1d<f32>, GPUDeviceOuterBatchScalar<u32>, GPUDeviceOuterBatchScalar<f32>> for SoftmaxCategoricalNLLOp {
  fn build(x_: Val<GPUDeviceOuterBatchArray1d<f32>>, fixed_softmax_: Val<GPUDeviceOuterBatchArray1d<f32>>, category_data_: Val<GPUDeviceOuterBatchScalar<u32>>) -> Val<GPUDeviceOuterBatchScalar<f32>> {
    SoftmaxCategoricalNLLOp::build_device_obatch_1d_op(x_, fixed_softmax_, category_data_)
  }
}

impl SoftmaxCategoricalNLLOp {
  fn build_device_obatch_1d_op(x_: Val<GPUDeviceOuterBatchArray1d<f32>>, fixed_softmax_: Val<GPUDeviceOuterBatchArray1d<f32>>, category_data_: Val<GPUDeviceOuterBatchScalar<u32>>) -> Val<GPUDeviceOuterBatchScalar<f32>> {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = GPUDeviceOuterBatchScalar::zeros((), x.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        //let x_ = x_.clone();
        let prob_ = fixed_softmax_.clone();
        let data_ = category_data_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            //let x = x_.get(txn);
            let prob = prob_.get(txn);
            let data = data_.get(txn);
            let mut y = output.get_mut(txn, token);
            //guard._wait(x.async_state());
            guard._wait(prob.async_state());
            guard._wait(data.async_state());
            guard._wait(y.async_state());
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = prob.size();
            let x_bsz = prob.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_softmax_cat_nll_packed_f32(
                    sz2uint(x_size),
                    sz2uint(x_bsz),
                    prob.as_view().as_dptr(),
                    data.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                // TODO
                unimplemented!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        let prob_ = fixed_softmax_.clone();
        let data_ = category_data_.clone();
        Box::new(move |_: Pass, y_: Val<_>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = SoftmaxCategoricalNLLBwdOp::build_device_obatch_1d_op(adj_y_, prob_.clone(), data_.clone());
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(SoftmaxCategoricalNLLOp, ext, x_, fixed_softmax_, category_data_)))
  }
}

impl SoftmaxCategoricalNLLBwdOp {
  fn build_device_obatch_1d_op(adj_y_: Val<GPUDeviceOuterBatchScalar<f32>>, /*x_: Val<GPUDeviceOuterBatchArray1d<f32>>,*/ fixed_softmax_: Val<GPUDeviceOuterBatchArray1d<f32>>, category_data_: Val<GPUDeviceOuterBatchScalar<u32>>) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    let ext = OpExt{
      make_val: {
        let x_ = fixed_softmax_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = GPUDeviceOuterBatchArray1d::zeros(x.size(), x.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = adj_y_.clone();
        let prob_ = fixed_softmax_.clone();
        let data_ = category_data_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn);
            let prob = prob_.get(txn);
            let data = data_.get(txn);
            let mut dx = output.get_mut(txn, token);
            guard._wait(dy.async_state());
            guard._wait(prob.async_state());
            guard._wait(data.async_state());
            guard._wait(dx.async_state());
            // FIXME: size checks.
            // FIXME: set batch size.
            let x_size = prob.size();
            let x_bsz = prob.batch_size();
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_softmax_cat_nll_bwd_packed_f32(
                    sz2uint(x_size),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    prob.as_view().as_dptr(),
                    data.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                unsafe { anode_gpu_softmax_cat_nll_bwd_packed_accumulate_f32(
                    sz2uint(x_size),
                    sz2uint(x_bsz),
                    dy.as_view().as_dptr(),
                    prob.as_view().as_dptr(),
                    data.as_view().as_dptr(),
                    dx.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        //let dy_ = adj_y_.clone();
        //let prob_ = fixed_softmax_.clone();
        //let data_ = category_data_.clone();
        Box::new(move |_: Pass, dx_: Val<_>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(_) = dx_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(SoftmaxCategoricalNLLBwdOp, ext, adj_y_, fixed_softmax_, category_data_)))
  }
}

// TODO: need more trait bounds.
impl<V> ConstantOpsExt<f32, V> for Val<V> {
  default fn set_constant(self, c: f32) -> Val<V> {
    // TODO
    unimplemented!();
  }

  default fn add_constant(self, c: f32) -> Val<V> {
    // TODO
    unimplemented!();
  }

  default fn mult_constant(self, c: f32) -> Val<V> {
    // TODO
    unimplemented!();
  }
}

impl<T> Add<T> for Val<GPUDeviceArray1d<T>>
where T: Copy,
      Val<GPUDeviceArray1d<T>>: ConstantOpsExt<T, GPUDeviceArray1d<T>>,
{
  type Output = Val<GPUDeviceArray1d<T>>;

  fn add(self, c: T) -> Val<GPUDeviceArray1d<T>> {
    self.add_constant(c)
  }
}

impl Add<Val<GPUDeviceArray1d<f32>>> for f32 {
  type Output = Val<GPUDeviceArray1d<f32>>;

  fn add(self, x_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    x_.add_constant(self)
  }
}

impl<T> Mul<T> for Val<GPUDeviceArray1d<T>>
where T: Copy,
      Val<GPUDeviceArray1d<T>>: ConstantOpsExt<T, GPUDeviceArray1d<T>>,
{
  type Output = Val<GPUDeviceArray1d<T>>;

  fn mul(self, c: T) -> Val<GPUDeviceArray1d<T>> {
    self.mult_constant(c)
  }
}

impl Mul<Val<GPUDeviceArray1d<f32>>> for f32 {
  type Output = Val<GPUDeviceArray1d<f32>>;

  fn mul(self, x_: Val<GPUDeviceArray1d<f32>>) -> Val<GPUDeviceArray1d<f32>> {
    x_.mult_constant(self)
  }
}

impl<T> Add<Val<GPUDeviceArray1d<T>>> for Val<GPUDeviceOuterBatchArray1d<T>>
where T: Copy,
{
  type Output = Val<GPUDeviceOuterBatchArray1d<T>>;

  fn add(self, y_: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    // TODO
    unimplemented!();
  }
}

impl BatchSumOpExt<GPUDeviceOuterBatchScalar<f32>, GPUDeviceScalar<f32>> for BatchSumOp {
  fn build(x_: Val<GPUDeviceOuterBatchScalar<f32>>) -> Val<GPUDeviceScalar<f32>> {
    //println!("DEBUG: build batch sum op...");
    BatchSumOp::build_device_f32_op(x_)
  }
}

impl BatchSumOp {
  pub fn build_device_f32_op(x_: Val<GPUDeviceOuterBatchScalar<f32>>) -> Val<GPUDeviceScalar<f32>> {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = GPUDeviceScalar::zeros((), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //println!("DEBUG: BatchSumOp: apply: this is {}", output.xvar._raw());
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: BatchSumOp: apply:   nontrivial write");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                // TODO: should use a higher level wrapper for this.
                unsafe { gpudevicemem_sum_packed_deterministic_f32(
                    sz2uint(x.batch_size()),
                    x.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                // TODO: should use a higher level wrapper for this.
                unsafe { gpudevicemem_sum_packed_accumulate_deterministic_f32(
                    sz2uint(x.batch_size()),
                    x.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = adj_y_.batch_broadcast_like(x_.clone());
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(BatchSumOp, ext, x_)))
  }
}

impl BatchBroadcastOpExt<GPUDeviceScalar<f32>, GPUDeviceOuterBatchScalar<f32>> for BatchBroadcastOp {
  fn build(x_: Val<GPUDeviceScalar<f32>>, target: usize) -> Val<GPUDeviceOuterBatchScalar<f32>> {
    //println!("DEBUG: build batch broadcast like op...");
    BatchBroadcastOp::build_device_f32_op(x_, target)
  }
}

impl BatchBroadcastOp {
  pub fn build_device_f32_op(x_: Val<GPUDeviceScalar<f32>>, target: usize) -> Val<GPUDeviceOuterBatchScalar<f32>> {
    let ext = OpExt{
      make_val: {
        //let target_ = target_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          //let target_ = target_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            //let target = target_.get(txn);
            //guard._wait(target.async_state());
            let y = GPUDeviceOuterBatchScalar::zeros((), target, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        //let target_ = target_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: BatchBroadcastOp: apply");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            //let target = target_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            //guard._wait(target.async_state());
            guard._wait(y.async_state());
            y.set_batch_size(target);
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                // TODO: should use a higher level wrapper for this.
                unsafe { gpudevicemem_bcast_packed_f32(
                    sz2uint(y.batch_size()),
                    x.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                // TODO: should use a higher level wrapper for this.
                unsafe { gpudevicemem_bcast_packed_accumulate_f32(
                    sz2uint(y.batch_size()),
                    x.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = adj_y_.batch_sum();
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(BatchBroadcastOp, ext, x_)))
  }
}

impl BatchBroadcastLikeOpExt<GPUDeviceScalar<f32>, GPUDeviceOuterBatchScalar<f32>> for BatchBroadcastLikeOp {
  fn build(x_: Val<GPUDeviceScalar<f32>>, target_: Val<GPUDeviceOuterBatchScalar<f32>>) -> Val<GPUDeviceOuterBatchScalar<f32>> {
    //println!("DEBUG: build batch broadcast like op...");
    BatchBroadcastLikeOp::build_device_f32_op(x_, target_)
  }
}

impl BatchBroadcastLikeOp {
  pub fn build_device_f32_op(x_: Val<GPUDeviceScalar<f32>>, target_: Val<GPUDeviceOuterBatchScalar<f32>>) -> Val<GPUDeviceOuterBatchScalar<f32>> {
    let ext = OpExt{
      make_val: {
        let target_ = target_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let target_ = target_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let target = target_.get(txn);
            guard._wait(target.async_state());
            let y = GPUDeviceOuterBatchScalar::zeros((), target.max_batch_size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let target_ = target_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //println!("DEBUG: BatchBroadcastLikeOp: apply");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            let target = target_.get(txn);
            let mut y = output.get_mut(txn, token);
            guard._wait(x.async_state());
            guard._wait(target.async_state());
            guard._wait(y.async_state());
            y.set_batch_size(target.batch_size());
            match cap {
              WriteCap::Assign => {
                let mut stream = conn.cuda_stream();
                // TODO: should use a higher level wrapper for this.
                unsafe { gpudevicemem_bcast_packed_f32(
                    sz2uint(y.batch_size()),
                    x.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
              WriteCap::Accumulate => {
                let mut stream = conn.cuda_stream();
                // TODO: should use a higher level wrapper for this.
                unsafe { gpudevicemem_bcast_packed_accumulate_f32(
                    sz2uint(y.batch_size()),
                    x.as_view().as_dptr(),
                    y.as_view_mut().as_mut_dptr(),
                    conn.cuda_kernel_config() as *const _,
                    stream.as_mut_ptr(),
                ) };
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      // TODO
      tangent: None,
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = adj_y_.batch_sum();
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    //Val::from(Rc::new(F1Op::new(BatchBroadcastLikeOp, ext, x_)))
    //Val::from(Rc::new(F2Op::new(BatchBroadcastLikeOp, ext, x_, target_)))
    Val::from(Rc::new(F2Op::new(BatchBroadcastLikeOp, ext, target_, x_)))
  }
}

// FIXME: impl the other `PositiveClipExt` trait, see below.
/*impl PositiveClipFlatMapExt<GPUDeviceOuterBatchArray1d<f32>> for Val<GPUDeviceOuterBatchArray1d<f32>> {
  fn positive_clip(self) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    FlatMapOp::<PositiveClipFlatMapF>::build_gpu_obatch_val(PositiveClipFlatMapF, self)
  }
}

impl PositiveClipFlatMapExt<GPUDeviceOuterBatchArray3d<f32>> for Val<GPUDeviceOuterBatchArray3d<f32>> {
  fn positive_clip(self) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    FlatMapOp::<PositiveClipFlatMapF>::build_gpu_obatch_val(PositiveClipFlatMapF, self)
  }
}*/

impl TanhFlatMapExt<GPUDeviceOuterBatchArray1d<f32>> for Val<GPUDeviceOuterBatchArray1d<f32>> {
  fn tanh(self) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    // TODO
    unimplemented!();
  }
}

pub trait ApplyGPUFlatMap<T> where T: Copy {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<T>, y: GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

pub trait ApplyGPUFlatMapBwd<T> where T: Copy {
  fn apply_gpu_flat_map_bwd(&self, adj_y: GPUDeviceArrayView1d<T>, y: GPUDeviceArrayView1d<T>, adj_x: GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

pub trait ApplyGPUFlatMapInplace<T> where T: Copy {
  fn apply_gpu_flat_map_inplace(&self, x: GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

pub trait ApplyGPUFlatMapBwdInplace<T> where T: Copy {
  fn apply_gpu_flat_map_bwd_inplace(&self, adj_y: GPUDeviceArrayViewMut1d<T>, y: GPUDeviceArrayView1d<T>, conn: GPUDeviceConn);
}

pub trait BuildGPUFlatMapAdj<T, A> where T: Copy {
  fn build_gpu_adj(&self, adj_y_: Val<A>, y_: Val<A>) -> Val<A> where A: GPUDeviceArrayZeros<T> { unimplemented!(); }
  //fn build_gpu_obatch_adj(&self, adj_y_: Val<GPUDeviceOuterBatchArray<Idx, T>>, y_: Val<GPUDeviceOuterBatchArray<Idx, T>>) -> Val<GPUDeviceOuterBatchArray<Idx, T>> { unimplemented!(); }
  //fn build_gpu_adj2(&self, adj_y_: Val<A>, x_: Val<A>, y_: Val<A>) -> Val<A> { unimplemented!(); }
}

pub trait BuildGPUBatchFlatMapAdj<T, Idx> where T: Copy {
  fn build_gpu_obatch_adj(&self, adj_y_: Val<GPUDeviceOuterBatchArray<Idx, T>>, y_: Val<GPUDeviceOuterBatchArray<Idx, T>>) -> Val<GPUDeviceOuterBatchArray<Idx, T>> { unimplemented!(); }
}

impl ApplyGPUFlatMap<f32> for ModulusFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_modulus_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for ModulusFlatMapF
where T: Copy,
{
}

impl ApplyGPUFlatMap<f32> for SquareFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_square_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for SquareFlatMapF
where T: Copy,
{
}

impl ApplyGPUFlatMap<f32> for PositiveClipFlatMap<f32> {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.is_packed());
    assert!(y.is_packed());
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_positive_clip_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMapBwd<f32> for PositiveClipFlatMap<f32> {
  fn apply_gpu_flat_map_bwd(&self, adj_y: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayView1d<f32>, adj_x: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(adj_y.is_packed());
    assert!(y.is_packed());
    assert!(adj_x.is_packed());
    assert!(adj_y.size() <= u32::max_value() as _);
    assert_eq!(adj_y.size(), y.size());
    assert_eq!(adj_y.size(), adj_x.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_positive_clip_flat_map_bwd_f32(
        adj_y.size() as _,
        adj_y.as_dptr(),
        y.as_dptr(),
        adj_x.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMapInplace<f32> for PositiveClipFlatMap<f32> {
  fn apply_gpu_flat_map_inplace(&self, x: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.is_packed());
    assert!(x.size() <= u32::max_value() as _);
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_positive_clip_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        x.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMapBwdInplace<f32> for PositiveClipFlatMap<f32> {
  fn apply_gpu_flat_map_bwd_inplace(&self, adj_y: GPUDeviceArrayViewMut1d<f32>, y: GPUDeviceArrayView1d<f32>, conn: GPUDeviceConn) {
    assert!(adj_y.is_packed());
    assert!(y.is_packed());
    assert!(adj_y.size() <= u32::max_value() as _);
    assert_eq!(adj_y.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_positive_clip_flat_map_bwd_f32(
        adj_y.size() as _,
        adj_y.as_dptr(),
        y.as_dptr(),
        adj_y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMap<f32> for PositiveClipFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_positive_clip_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for PositiveClipFlatMapF
where T: Copy + 'static,
      A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
          + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
          + 'static,
      UnitStepFlatMapF: ApplyGPUFlatMap<T>,
{
  fn build_gpu_adj(&self, adj_y_: Val<A>, y_: Val<A>) -> Val<A> where A: GPUDeviceArrayZeros<T> {
    // TODO: use fused kernel to avoid an extra allocation.
    let dy_dx_ = FlatMapOp::<UnitStepFlatMapF>::build_gpu_val::<T, A>(UnitStepFlatMapF, y_);
    //let adj_x = dy_dx_.flat_mult(adj_y);
    //adj_x
    unimplemented!();
  }
}

impl<T, Idx> BuildGPUBatchFlatMapAdj<T, Idx> for PositiveClipFlatMapF
where T: Copy + 'static,
      Idx: ArrayIndex,
      UnitStepFlatMapF: ApplyGPUFlatMap<T>,
{
  fn build_gpu_obatch_adj(&self, adj_y_: Val<GPUDeviceOuterBatchArray<Idx, T>>, y_: Val<GPUDeviceOuterBatchArray<Idx, T>>) -> Val<GPUDeviceOuterBatchArray<Idx, T>> {
    // TODO: use fused kernel to avoid an extra allocation.
    //let dy_dx_ = FlatMapOp::<UnitStepFlatMapF>::build_gpu_obatch_val::<T, A>(UnitStepFlatMapF, y_);
    //let adj_x = dy_dx_.flat_mult(adj_y);
    //adj_x
    unimplemented!();
  }
}

impl ApplyGPUFlatMap<f32> for UnitStepFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_unit_step_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for UnitStepFlatMapF
where T: Copy,
      A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
          + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
          + 'static,
{
  fn build_gpu_adj(&self, adj_y_: Val<A>, y_: Val<A>) -> Val<A> where A: GPUDeviceArrayZeros<T> {
    // TODO
    //let adj_x = zeros_like(adj_y_);
    //adj_x
    unimplemented!();
  }
}

impl ApplyGPUFlatMap<f32> for TanhFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_tanh_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for TanhFlatMapF
where T: Copy,
{
}

impl ApplyGPUFlatMap<f32> for RCosh2FlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_rcosh2_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for RCosh2FlatMapF
where T: Copy,
{
}

impl<F> FlatMapOp<F> where F: Clone + 'static {
  pub fn build_gpu_val<T, A>(f_config: F, x_: Val<A>) -> Val<A>
  where T: Copy,
        A: GPUDeviceArrayZeros<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        F: ApplyGPUFlatMap<T> + BuildGPUFlatMapAdj<T, A>,
  {
    // FIXME
    unimplemented!();
  }

  pub fn build_gpu_obatch_val<T, Idx>(f_config: F, x_: Val<GPUDeviceOuterBatchArray<Idx, T>>) -> Val<GPUDeviceOuterBatchArray<Idx, T>>
  where T: Copy + 'static,
        Idx: ArrayIndex + 'static,
        GPUDeviceOuterBatchArray<Idx, T>: GPUDeviceBatchArrayZeros<T>,
            /*+ FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,*/
        F: ApplyGPUFlatMap<T> + BuildGPUBatchFlatMapAdj<T, Idx>,
  {
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let x_size = x_.get(txn).size();
            let x_max_batch_sz = x_.get(txn).max_batch_size();
            GPUDeviceOuterBatchArray::zeros(x_size, x_max_batch_sz, conn)
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let f_config = f_config.clone();
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray<Idx, T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            //implicit_ctx()._debug_print();
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let x = x_.get(txn);
                let flat_x = x.flat_view().unwrap();
                let mut y = output.get_mut(txn, token);
                let mut flat_y = y.flat_view_mut().unwrap();
                guard._wait(flat_x.async_state());
                guard._wait(flat_y.async_state());
                f_config.apply_gpu_flat_map(flat_x, flat_y, conn);
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        let f_config = f_config.clone();
        Box::new(move |args| {
          let f_config = f_config.clone();
          let x_ = match args[0].downcast_ref::<Val<GPUDeviceOuterBatchArray<Idx, T>>>() {
            None => panic!(),
            Some(x_) => x_.clone(),
          };
          FlatMapOp::<F>::build_gpu_obatch_val::<T, Idx>(f_config, x_)
        })
      }),
      tangent: None,
      adjoint: Some({
        let f_config = f_config.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray<Idx, T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = f_config.build_gpu_obatch_adj(adj_y_, y_);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: Some({
        let f_config = f_config.clone();
        Box::new(move |x_: Val<GPUDeviceOuterBatchArray<Idx, T>>| {
          // FIXME
          //FlatMapInplaceOp::<F>::build_gpu_val::<T, A>(f_config.clone(), x_)
          unimplemented!();
        })
      }),
    };
    Val::from(Rc::new(F1Op::new(FlatMapOp{f: f_config}, ext, x_)))
  }
}

impl<F> FlatMapInplaceOp<F> {
  pub fn build_gpu_val<T, A>(f_config: F, x_: Val<A>) -> Val<A>
  where T: Copy,
        A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
  {
    // FIXME
    //let value = x_.value().clobber();
    //Rc::new(F1Op::new(FlatMapInplaceOp{f: f_config}, ext, x_, value))
    unimplemented!();
  }
}

pub trait ApplyGPUFlatJoin<T> where T: Copy {
  fn apply_gpu_flat_join(&self, xs: Vec<GPUDeviceArrayView1d<T>>, y: GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

impl ApplyGPUFlatJoin<f32> for Map2FlatJoin<IdentityFlatMapF, UnitStepFlatMapF, ProductReduce> {
  fn apply_gpu_flat_join(&self, xs: Vec<GPUDeviceArrayView1d<f32>>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert_eq!(xs.len(), 2);
    for x in xs.iter() {
      assert!(x.size() <= u32::max_value() as _);
      assert_eq!(x.size(), y.size());
    }
    let mut stream = conn.cuda_stream();
    unsafe { anode_gpu_M1_copy_map_M2_unit_step_map_R_product_reduce_flat_join_f32(
        sz2uint(y.size()),
        xs[0].as_dptr(),
        xs[1].as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        stream.as_mut_ptr(),
    ) };
  }
}

impl<F> FlatJoinOp<F> where F: Clone + 'static {
  pub fn build_gpu_val<T, A>(f_config: F, xs_: Vec<Val<A>>) -> Val<A>
  where T: Copy,
        A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        F: ApplyGPUFlatJoin<T> /*+ BuildGPUFlatJoinAdj<T, A>,*/
  {
    let ext = OpExt{
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<_>| {
          // TODO
          unimplemented!();
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let f_config = f_config.clone();
        let xs_ = xs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let mut pool = implicit_ctx().gpu().pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut flat_xs = vec![];
                for x_ in xs_.iter() {
                  let x = x_.get(txn);
                  let flat_x = x.flat_view().unwrap();
                  guard._wait(flat_x.async_state());
                  flat_xs.push(flat_x);
                }
                let mut y = output.get_mut(txn, token);
                let mut flat_y = y.flat_view_mut().unwrap();
                guard._wait(flat_y.async_state());
                f_config.apply_gpu_flat_join(flat_xs, flat_y, conn);
              }
              _ => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        let f_config = f_config.clone();
        Box::new(move |args| {
          let f_config = f_config.clone();
          let xs_ = match args[0].downcast_ref::<Vec<Val<A>>>() {
            None => panic!(),
            Some(xs_) => xs_.clone(),
          };
          FlatJoinOp::<F>::build_gpu_val::<T, A>(f_config, xs_)
        })
      }),
      tangent: None,
      adjoint: None,
      inplace: None,
      /*inplace: Some({
        let f_config = f_config.clone();
        Box::new(move |x_: Val<A>| {
          FlatMapInplaceOp::<F>::build_gpu_val::<T, A>(f_config.clone(), x_)
        })
      }),*/
    };
    Val::from(Rc::new(FJoinOp::new(FlatJoinOp{f: f_config}, ext, xs_)))
  }
}

impl PositiveClipExt<GPUDeviceArray1d<f32>> for Val<GPUDeviceArray1d<f32>> {
  fn positive_clip(self) -> Val<GPUDeviceArray1d<f32>> {
    PositiveClipOp::build_device_op(self)
  }
}

impl PositiveClipExt<GPUDeviceOuterBatchArray1d<f32>> for Val<GPUDeviceOuterBatchArray1d<f32>> {
  fn positive_clip(self) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    PositiveClipOp::build_device_op(self)
  }
}

impl PositiveClipExt<GPUDeviceOuterBatchArray3d<f32>> for Val<GPUDeviceOuterBatchArray3d<f32>> {
  fn positive_clip(self) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    PositiveClipOp::build_device_op(self)
  }
}

impl PositiveClipInplaceExt<GPUDeviceArray1d<f32>> for Val<GPUDeviceArray1d<f32>> {
  fn positive_clip_inplace(self) -> Val<GPUDeviceArray1d<f32>> {
    PositiveClipClobberOp::build_device_op(self)
  }
}

impl PositiveClipInplaceExt<GPUDeviceOuterBatchArray1d<f32>> for Val<GPUDeviceOuterBatchArray1d<f32>> {
  fn positive_clip_inplace(self) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    PositiveClipClobberOp::build_device_op(self)
  }
}

impl PositiveClipInplaceExt<GPUDeviceOuterBatchArray3d<f32>> for Val<GPUDeviceOuterBatchArray3d<f32>> {
  fn positive_clip_inplace(self) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    PositiveClipClobberOp::build_device_op(self)
  }
}

impl PositiveClipInplaceExt<GPUDeviceOuterBatchArray4d<f32>> for Val<GPUDeviceOuterBatchArray4d<f32>> {
  fn positive_clip_inplace(self) -> Val<GPUDeviceOuterBatchArray4d<f32>> {
    PositiveClipClobberOp::build_device_op(self)
  }
}

impl PositiveClipOp {
  pub fn build_device_op<T, A>(old_x_: Val<A>) -> Val<A>
  where T: ZeroBits + Default + 'static,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        PositiveClipFlatMap<T>: ApplyGPUFlatMap<T>,
        PositiveClipFlatMap<T>: ApplyGPUFlatMapBwd<T>,
  {
    let new_x_ = old_x_.clone();
    let ext = OpExt{
      make_val: {
        let x_ = new_x_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = A::zeros_shape(x.shape(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = new_x_.clone();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<_>| {
          // TODO: check valrefs.
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let x = x_.get(txn);
                let mut y = output.get_mut(txn, token);
                guard._wait(x.async_state());
                guard._wait(y.async_state());
                let fmap: PositiveClipFlatMap<T> = Default::default();
                fmap.apply_gpu_flat_map(x.flat_view().unwrap(), y.flat_view_mut().unwrap(), conn);
              }
              WriteCap::Accumulate => {
                panic!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        let x_ = new_x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = PositiveClipBwdOp::build_device_op(adj_y_, y_);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::new(Rc::new(F1Op::new(PositiveClipOp, ext, new_x_)))
  }
}

impl PositiveClipBwdOp {
  pub fn build_device_op<T, A>(old_adj_y_: Val<A>, y_: Val<A>) -> Val<A>
  where T: ZeroBits + Default + 'static,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        PositiveClipFlatMap<T>: ApplyGPUFlatMapBwd<T>,
  {
    let new_adj_y_ = old_adj_y_.clone();
    let ext = OpExt{
      make_val: {
        let adj_y_ = new_adj_y_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let adj_y_ = adj_y_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let adj_y = adj_y_.get(txn);
            guard._wait(adj_y.async_state());
            let adj_x = A::zeros_shape(adj_y.shape(), conn);
            guard._wait(adj_x.async_state());
            adj_x
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let adj_y_ = new_adj_y_.clone();
        let y_ = y_.clone();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<_>| {
          // TODO: check valrefs.
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let adj_y = adj_y_.get(txn);
                let y = y_.get(txn);
                let mut adj_x = output.get_mut(txn, token);
                guard._wait(adj_y.async_state());
                guard._wait(y.async_state());
                guard._wait(adj_x.async_state());
                let fmap: PositiveClipFlatMap<T> = Default::default();
                fmap.apply_gpu_flat_map_bwd(adj_y.flat_view().unwrap(), y.flat_view().unwrap(), adj_x.flat_view_mut().unwrap(), conn);
              }
              WriteCap::Accumulate => {
                panic!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<_>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::new(Rc::new(F2Op::new(PositiveClipBwdOp, ext, new_adj_y_, y_)))
  }
}

impl PositiveClipClobberOp {
  pub fn build_device_op<T, A>(old_x_: Val<A>) -> Val<A>
  where T: ZeroBits + Default + 'static,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        PositiveClipFlatMap<T>: ApplyGPUFlatMapInplace<T>,
        PositiveClipFlatMap<T>: ApplyGPUFlatMapBwd<T>,
        PositiveClipFlatMap<T>: ApplyGPUFlatMapBwdInplace<T>,
  {
    //let new_x_ = old_x_.clone();
    let new_x_ = old_x_.clobber();
    let ext = OpExt{
      make_val: {
        let x_ = new_x_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let y = A::zeros_shape(x.shape(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<_>| {
          // TODO: check valrefs.
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                let fmap: PositiveClipFlatMap<T> = Default::default();
                fmap.apply_gpu_flat_map_inplace(y.flat_view_mut().unwrap(), conn);
              }
              WriteCap::Accumulate => {
                panic!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        let x_ = new_x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = PositiveClipBwdOp::build_device_op(adj_y_, y_);
            //let adj_x_ = PositiveClipBwdClobberOp::build_device_op(adj_y_, y_);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    let new_x_value = new_x_._static_value();
    assert!(new_x_value.is_some());
    Val::with_value_mode(Rc::new(F1Op::new(PositiveClipClobberOp, ext, new_x_)), new_x_value, WriteMode::Clobber)
  }
}

impl PositiveClipBwdClobberOp {
  pub fn build_device_op<T, A>(old_adj_y_: Val<A>, y_: Val<A>) -> Val<A>
  where T: ZeroBits + Default + 'static,
        A: GPUDeviceAsync
            + GPUDeviceZerosShape<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        PositiveClipFlatMap<T>: ApplyGPUFlatMapBwdInplace<T>,
  {
    //let new_adj_y_ = old_adj_y_.clone();
    let new_adj_y_ = old_adj_y_.clobber();
    let ext = OpExt{
      make_val: {
        let adj_y_ = new_adj_y_.clone();
        //Box::new(move || {
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let adj_y_ = adj_y_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let adj_y = adj_y_.get(txn);
            guard._wait(adj_y.async_state());
            let adj_x = A::zeros_shape(adj_y.shape(), conn);
            guard._wait(adj_x.async_state());
            adj_x
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let y_ = y_.clone();
        Box::new(move |txn: Txn, _state: RefMut<_>, output: OVal<_>| {
          // TODO: check valrefs.
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let y = y_.get(txn);
                let mut adj_x = output.get_mut(txn, token);
                guard._wait(y.async_state());
                guard._wait(adj_x.async_state());
                let fmap: PositiveClipFlatMap<T> = Default::default();
                fmap.apply_gpu_flat_map_bwd_inplace(adj_x.flat_view_mut().unwrap(), y.flat_view().unwrap(), conn);
              }
              WriteCap::Accumulate => {
                panic!();
              }
            }
          })
        })
      },
      build: None,
      tangent: None,
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<_>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    let new_adj_y_value = new_adj_y_._static_value();
    assert!(new_adj_y_value.is_some());
    Val::with_value_mode(Rc::new(F2Op::new(PositiveClipBwdClobberOp, ext, new_adj_y_, y_)), new_adj_y_value, WriteMode::Clobber)
  }
}

impl BroadcastLinearOp {
  pub fn build_device_obatch_3d_1d_op_f32(axis: isize, a_: Val<GPUDeviceArray1d<f32>>, x_: Val<GPUDeviceOuterBatchArray3d<f32>>)
      -> Val<GPUDeviceOuterBatchArray3d<f32>>
  // TODO: `ZeroBits` should not be necessary here.
  //where T: Zero + One + ZeroBits + Copy + 'static,
  where GPUDeviceArrayViewMut4d<f32>: GPUTensorMutOps<f32>,
  {
    // TODO
    unimplemented!();
  }
}

impl BroadcastAffineOp {
  pub fn build_device_obatch_3d_1d_op_f32(axis: isize, a_: Val<GPUDeviceArray1d<f32>>, x_: Val<GPUDeviceOuterBatchArray3d<f32>>, b_: Val<GPUDeviceArray1d<f32>>)
      -> Val<GPUDeviceOuterBatchArray3d<f32>>
  // TODO: `ZeroBits` should not be necessary here.
  //where T: Zero + One + ZeroBits + Copy + 'static,
  where GPUDeviceArrayViewMut4d<f32>: GPUTensorMutOps<f32>,
  {
    // TODO
    unimplemented!();
  }
}

impl ReduceSumOp {
  pub fn build_device_obatch_3d_1d_op_f32(axis: isize, x_: Val<GPUDeviceOuterBatchArray3d<f32>>)
      -> Val<GPUDeviceArray1d<f32>>
  // TODO: `ZeroBits` should not be necessary here.
  //where T: Zero + One + ZeroBits + Copy + 'static,
  where GPUDeviceArrayView4d<f32>: GPUTensorOps<f32>,
  {
    // TODO
    unimplemented!();
  }
}

impl LinearReduceSumOp {
  pub fn build_device_obatch_3d_1d_op_f32(axis: isize, x1_: Val<GPUDeviceOuterBatchArray3d<f32>>, x2_: Val<GPUDeviceOuterBatchArray3d<f32>>)
      -> Val<GPUDeviceArray1d<f32>>
  // TODO: `ZeroBits` should not be necessary here.
  //where T: Zero + One + ZeroBits + Copy + 'static,
  where GPUDeviceArrayView4d<f32>: GPUTensorOps<f32>,
  {
    // TODO
    unimplemented!();
  }
}

impl<T> LinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut1d<T>: GPUVectorMutOps<T>,
{
  fn mult(self, x: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    LinearOp::build_device_val(self, x)
  }
}

impl<T> LinearExt<GPUDeviceArray2d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn mult(self, x: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    LinearOp::build_device_obatch_val(self, x)
  }
}

impl<T> AffineExt<GPUDeviceArray2d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn mult_add(self, x: Val<GPUDeviceOuterBatchArray1d<T>>, b: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    AffineOp::build_device_obatch_val(self, x, b)
  }
}

impl<T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut1d<T>: GPUVectorMutOps<T>,
{
  fn left_transpose_mult(self, y: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    // TODO
    unimplemented!();
  }
}

impl<T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn left_transpose_mult(self, y: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    LinearOp::build_device_obatch_ltrans_val(self, y)
  }
}

impl<T> OuterLinearExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>> for Val<GPUDeviceArray1d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut1d<T>: GPUVectorMutOps<T>,
{
  fn outer_mult(self, x: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray2d<T>> {
    // TODO
    unimplemented!();
  }
}

impl<T> OuterLinearExt<GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray2d<T>> for Val<GPUDeviceOuterBatchArray1d<T>>
where T: Zero + One + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn outer_mult(self, x: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceArray2d<T>> {
    LinearOp::build_device_obatch_rtrans_val(self, x)
  }
}

impl LinearOp {
  pub fn build_device_val<T>(map_: Val<GPUDeviceArray2d<T>>, input_: Val<GPUDeviceArray1d<T>>)
      -> Val<GPUDeviceArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: Zero + One + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut1d<T>: GPUVectorMutOps<T>,
  {
    let ext = OpExt{
      make_val: {
        let map_ = map_.clone();
        Box::new(move |state: RefMut<_>| {
          let map_ = map_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let a_size = map_.get(txn).size();
            GPUDeviceArray1d::zeros(a_size[0], conn)
          }))
        })
      },
      apply: {
        let input_ = input_.clone();
        let map_ = map_.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Assign => {
                let a = map_.get(txn).as_view();
                let x = input_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                gpu_matrix_vector_mult(a, x, y, conn);
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, feedfwd: &mut FeedFwd| {
          let input_ = input_.clone();
          let map_ = map_.clone();
          let tng_input_ = input_.tangent(feedfwd);
          let tng_map_ = map_.tangent(feedfwd);
          // FIXME
          unimplemented!();
          //let y_ = map_.mult(tng_input_).add(tng_map_.mult(input_));
          //(y_.clone(), y_)
        })
      }),
      adjoint: Some({
        let x_ = input_.clone();
        let a_ = map_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          //if let Some(adj_y_) = sink.get_adj::<GPUDeviceArray1d<T>>(y_.var()) {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_a_ = adj_y_.clone().outer_mult(x_.clone());
            let adj_x_ = a_.clone().left_transpose_mult(adj_y_);
            a_.put_adjoint(adj_a_, sink);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LinearOp, ext, map_, input_)))
  }

  pub fn build_device_obatch_val<T>(w_: Val<GPUDeviceArray2d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>)
      -> Val<GPUDeviceOuterBatchArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: Zero + One + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let w_size = w_.get(txn).size();
            let x_max_bsz = x_.get(txn).max_batch_size();
            GPUDeviceOuterBatchArray1d::zeros(w_size[0], x_max_bsz, conn)
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let w = w_.get(txn);
                let x = x_.get(txn);
                let mut y = output.get_mut(txn, token);
                guard._wait(w.async_state());
                guard._wait(x.async_state());
                guard._wait(y.async_state());
                assert_eq!(w.size()[1], x.size());
                assert_eq!(w.size()[0], y.size());
                // TODO: set batch size.
                assert_eq!(x.batch_size(), y.batch_size());
                y.as_view_mut().matrix_mult(w.as_view(), x.as_view(), conn);
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_w_ = adj_y_.clone().outer_mult(x_.clone());
            let adj_x_ = w_.clone().left_transpose_mult(adj_y_);
            w_.put_adjoint(adj_w_, sink);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LinearOp, ext, w_, x_)))
  }

  pub fn build_device_obatch_ltrans_val<T>(w_: Val<GPUDeviceArray2d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>)
      -> Val<GPUDeviceOuterBatchArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: Zero + One + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let w_size = w_.get(txn).size();
            let x_max_bsz = x_.get(txn).max_batch_size();
            GPUDeviceOuterBatchArray1d::zeros(w_size[1], x_max_bsz, conn)
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let w = w_.get(txn).as_view();
                let x = x_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                guard._wait(w.async_state());
                guard._wait(x.async_state());
                guard._wait(y.async_state());
                y.left_transpose_matrix_mult(w, x, conn);
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LeftTransposeLinearOp, ext, w_, x_)))
  }

  pub fn build_device_obatch_rtrans_val<T>(w_: Val<GPUDeviceOuterBatchArray1d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>)
      -> Val<GPUDeviceArray2d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: Zero + One + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let w_size = w_.get(txn).size();
            let x_size = x_.get(txn).size();
            GPUDeviceArray2d::zeros([w_size, x_size], conn)
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceArray2d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let w = w_.get(txn).as_view();
                let x = x_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                guard._wait(w.async_state());
                guard._wait(x.async_state());
                guard._wait(y.async_state());
                y.right_transpose_matrix_mult(w, x, conn);
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray2d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(OuterLinearOp, ext, w_, x_)))
  }
}

impl AffineOp {
  pub fn build_device_obatch_val<T>(w_: Val<GPUDeviceArray2d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>, b_: Val<GPUDeviceArray1d<T>>)
      -> Val<GPUDeviceOuterBatchArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: Zero + One + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let w_size = w_.get(txn).size();
            let x_max_bsz = x_.get(txn).max_batch_size();
            GPUDeviceOuterBatchArray1d::zeros(w_size[0], x_max_bsz, conn)
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let w = w_.get(txn).as_view();
                let x = x_.get(txn).as_view();
                let b = b_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                guard._wait(w.async_state());
                guard._wait(x.async_state());
                guard._wait(b.async_state());
                guard._wait(y.async_state());
                y.matrix_mult(w, x, conn.clone());
                y.broadcast_add_vector_inplace(b, 0, conn.clone());
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray1d<T>>, state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_w_ = adj_y_.clone().outer_mult(x_.clone());
            let adj_x_ = w_.clone().left_transpose_mult(adj_y_.clone());
            w_.put_adjoint(adj_w_, sink);
            x_.put_adjoint(adj_x_, sink);
            // FIXME: calculate adj of b via a reduction.
            /*let adj_b_ = adj_y_.reduce_sum(1);
            b_.put_adjoint(adj_b_, sink);*/
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(AffineOp, ext, w_, x_, b_)))
  }
}

fn conv2d_to_xconv_shape(conv_shape: Conv2dShape, batch_sz: usize) -> XConvFullShape {
  let src_size = conv_shape.calculate_input_size();
  let dst_size = conv_shape.calculate_output_size();
  XConvFullShape::Conv2d(Conv2dFullShape{
    ker_space_axes:   conv_shape.ker_space_axes,
    ker_output_axis:  conv_shape.ker_output_axis,
    src_space_axes:   conv_shape.src_space_axes,
    src_feature_axis: conv_shape.src_feature_axis,
    src_batch_axis:   conv_shape.src_batch_axis,
    // TODO: assumes NCHW layout.
    src_size:         [
      src_size[0],
      src_size[1],
      src_size[2],
      batch_sz,
    ],
    dst_space_axes:   conv_shape.dst_space_axes,
    dst_feature_axis: conv_shape.dst_feature_axis,
    dst_batch_axis:   conv_shape.dst_batch_axis,
    dst_size:         [
      dst_size[0],
      dst_size[1],
      dst_size[2],
      batch_sz,
    ],
    ker_size: conv_shape.ker_dims,
    dilation: conv_shape.dilation,
    stride:   conv_shape.stride,
    zero_pad: conv_shape.zero_pad,
    groups:   1,
    cross:    true,
  })
}

impl<T> ConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>> for Val<GPUDeviceArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
      GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
      GPUDeviceArrayViewMut4d<T>: GPUTensorMutOps<T> + GPUBatchConvOps<T, T, T>,
      Val<GPUDeviceOuterBatchArray3d<T>>: OuterConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
      Val<GPUDeviceArray4d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
{
  type ConvShape = Conv2dShape;

  fn conv(self, conv_shape: Conv2dShape, x: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    Conv2dLinearOp::build_device_obatch_val(conv_shape, self, x)
  }
}

impl<T> ConvAffineExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      //CudnnHandle: CudnnConvExt<T, T, T>,
      CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
      GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
      GPUDeviceArrayViewMut4d<T>: GPUTensorMutOps<T> + GPUBatchConvOps<T, T, T>,
      Val<GPUDeviceOuterBatchArray3d<T>>: OuterConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
      Val<GPUDeviceArray4d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
      Val<GPUDeviceOuterBatchArray3d<T>>: ConvReduceBwdExt<GPUDeviceOuterBatchArray3d<T>, GPUDeviceArray1d<T>, ConvShape=Conv2dShape>,
{
  fn conv_add(self, conv_shape: Conv2dShape, x: Val<GPUDeviceOuterBatchArray3d<T>>, b: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    Conv2dAffineOp::build_device_obatch_val(conv_shape, self, x, b)
  }
}

impl Conv2dLinearOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv2dShape,
      w_: Val<GPUDeviceArray4d<T>>,
      x_: Val<GPUDeviceOuterBatchArray3d<T>>)
  -> Val<GPUDeviceOuterBatchArray3d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        //CudnnHandle: CudnnConvExt<T, T, T>,
        CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
        GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
        GPUDeviceArrayViewMut4d<T>: GPUTensorMutOps<T> + GPUBatchConvOps<T, T, T>,
        Val<GPUDeviceOuterBatchArray3d<T>>: OuterConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
        Val<GPUDeviceArray4d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    /*{
      let dst_size = conv_shape._calculate_output_size();
      let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
        ker_space_axes:   conv_shape.ker_space_axes,
        ker_output_axis:  conv_shape.ker_output_axis,
        src_space_axes:   conv_shape.src_space_axes,
        src_feature_axis: conv_shape.src_feature_axis,
        src_batch_axis:   conv_shape.src_batch_axis,
        // TODO: assumes NCHW layout.
        src_size:         [
          conv_shape.src_size[0],
          conv_shape.src_size[1],
          conv_shape.src_size[2],
          32,
        ],
        dst_space_axes:   conv_shape.dst_space_axes,
        dst_feature_axis: conv_shape.dst_feature_axis,
        dst_batch_axis:   conv_shape.dst_batch_axis,
        dst_size:         [
          dst_size[0],
          dst_size[1],
          dst_size[2],
          32,
        ],
        ker_size: conv_shape.ker_size,
        dilation: conv_shape.dilation,
        stride:   conv_shape.stride,
        zero_pad: conv_shape.zero_pad,
        groups:   1,
        cross:    true,
      });
      let mut state_cache = state_cache.borrow_mut();
      state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
        let ctx = implicit_ctx().gpu();
        let mut pool = ctx.pool();
        let conn = pool.conn();
        match query_gpu_conv_fwd_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
          None => panic!("invalid conv2d config: {:?}", xconv_shape),
          Some((cfg, state)) => {
            conn.burst_arena().reserve_bytes(cfg.workspace_size());
            (cfg, state)
          }
        }
      });
    }*/
    let ext = OpExt{
      make_val: {
        //let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          //let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            //let w_size = w_.get(txn).size();
            //let x_size = x_.get(txn).size();
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_max_bsz = x.max_batch_size();
            let y_size = conv_shape.calculate_output_size();
            let y = GPUDeviceOuterBatchArray3d::zeros(y_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: set batch size.
                //assert_eq!(x_.get(txn).size(), conv_shape.src_size);
                let w = w_.get(txn).as_view();
                let x = x_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                guard._wait(w.async_state());
                guard._wait(x.async_state());
                guard._wait(y.async_state());
                /*let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
                  ker_space_axes:   conv_shape.ker_space_axes,
                  ker_output_axis:  conv_shape.ker_output_axis,
                  src_space_axes:   conv_shape.src_space_axes,
                  src_feature_axis: conv_shape.src_feature_axis,
                  src_batch_axis:   conv_shape.src_batch_axis,
                  // TODO: assumes NCHW layout.
                  src_size:         [
                    conv_shape.src_size[0],
                    conv_shape.src_size[1],
                    conv_shape.src_size[2],
                    x.size()[3],
                  ],
                  dst_space_axes:   conv_shape.dst_space_axes,
                  dst_feature_axis: conv_shape.dst_feature_axis,
                  dst_batch_axis:   conv_shape.dst_batch_axis,
                  dst_size: y.size(),
                  ker_size: conv_shape.ker_size,
                  dilation: conv_shape.dilation,
                  stride:   conv_shape.stride,
                  zero_pad: conv_shape.zero_pad,
                  groups:   1,
                  cross:    true,
                });*/
                // TODO: set batch size.
                assert_eq!(x.size()[3], y.size()[3]);
                let xconv_shape = conv2d_to_xconv_shape(conv_shape, x.size()[3]);
                let mut state_cache = state_cache.borrow_mut();
                let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
                  match query_gpu_conv_fwd_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                    None => panic!("invalid conv2d config: {:?}", xconv_shape),
                    Some((cfg, state)) => (cfg, state),
                  }
                });
                let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
                guard._wait(workspace.async_state());
                y.batch_conv2d(
                    &cfg,
                    state,
                    w,
                    x,
                    workspace.as_view_mut(),
                    conn.clone(),
                );
                double_check_scalar::<Self, _>(|| y.flat_view().unwrap().sync_vector_norm(conn).into());
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray3d<T>>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_w_ = adj_y_.clone().outer_conv(conv_shape, x_.clone());
            let adj_x_ = w_.clone().left_transpose_conv(conv_shape, adj_y_.clone());
            w_.put_adjoint(adj_w_, sink);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(Conv2dLinearOp{conv_shape}, ext, w_, x_)))
  }
}

impl Conv2dAffineOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv2dShape,
      w_: Val<GPUDeviceArray4d<T>>,
      x_: Val<GPUDeviceOuterBatchArray3d<T>>,
      b_: Val<GPUDeviceArray1d<T>>)
  -> Val<GPUDeviceOuterBatchArray3d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        //CudnnHandle: CudnnConvExt<T, T, T>,
        CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
        GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
        GPUDeviceArrayViewMut4d<T>: GPUTensorMutOps<T> + GPUBatchConvOps<T, T, T>,
        Val<GPUDeviceOuterBatchArray3d<T>>: OuterConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
        Val<GPUDeviceArray4d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>, ConvShape=Conv2dShape>,
        Val<GPUDeviceOuterBatchArray3d<T>>: ConvReduceBwdExt<GPUDeviceOuterBatchArray3d<T>, GPUDeviceArray1d<T>, ConvShape=Conv2dShape>,
  {
    let ext = OpExt{
      make_val: {
        //let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          //let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            //let w_size = w_.get(txn).size();
            //let x_size = x_.get(txn).size();
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_max_bsz = x.max_batch_size();
            let y_size = conv_shape.calculate_output_size();
            let y = GPUDeviceOuterBatchArray3d::zeros(y_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        let state_cache = RefCell::new(HashMap::new());
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: set batch size.
                //assert_eq!(x_.get(txn).size(), conv_shape.src_size);
                let w = w_.get(txn).as_view();
                let x = x_.get(txn).as_view();
                let b = b_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                guard._wait(w.async_state());
                guard._wait(x.async_state());
                guard._wait(b.async_state());
                guard._wait(y.async_state());
                /*let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
                  ker_space_axes:   conv_shape.ker_space_axes,
                  ker_output_axis:  conv_shape.ker_output_axis,
                  src_space_axes:   conv_shape.src_space_axes,
                  src_feature_axis: conv_shape.src_feature_axis,
                  src_batch_axis:   conv_shape.src_batch_axis,
                  // TODO: assumes NCHW layout.
                  src_size:         [
                    conv_shape.src_size[0],
                    conv_shape.src_size[1],
                    conv_shape.src_size[2],
                    x.size()[3],
                  ],
                  dst_space_axes:   conv_shape.dst_space_axes,
                  dst_feature_axis: conv_shape.dst_feature_axis,
                  dst_batch_axis:   conv_shape.dst_batch_axis,
                  dst_size: y.size(),
                  ker_size: conv_shape.ker_size,
                  dilation: conv_shape.dilation,
                  stride:   conv_shape.stride,
                  zero_pad: conv_shape.zero_pad,
                  groups:   1,
                  cross:    true,
                });*/
                // TODO: set batch size.
                assert_eq!(x.size()[3], y.size()[3]);
                let xconv_shape = conv2d_to_xconv_shape(conv_shape, x.size()[3]);
                let mut state_cache = state_cache.borrow_mut();
                let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
                  match query_gpu_conv_fwd_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                    None => panic!("invalid conv2d config: {:?}", xconv_shape),
                    Some((cfg, state)) => (cfg, state),
                  }
                });
                let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
                guard._wait(workspace.async_state());
                // TODO: can use the fused operation.
                y.batch_conv2d(
                    &cfg,
                    state,
                    w,
                    x,
                    workspace.as_view_mut(),
                    conn.clone(),
                );
                y.broadcast_add_1d_inplace(b, conv_shape.dst_feature_axis, conn.clone());
                /*y.batch_conv2d_affine(
                    &cfg,
                    &mut state,
                    w,
                    x,
                    b,
                    workspace.as_view_mut(),
                    conn.clone(),
                );*/
                double_check_scalar::<Self, _>(|| y.flat_view().unwrap().sync_vector_norm(conn).into());
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray3d<T>>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            //println!("DEBUG: Conv2dAffineOp: found adjoint for primal: {:?} => {:?}", y_._graph_key(), adj_y_._graph_key());
            let adj_w_ = adj_y_.clone().outer_conv(conv_shape, x_.clone());
            let adj_x_ = w_.clone().left_transpose_conv(conv_shape, adj_y_.clone());
            let adj_b_ = adj_y_.clone().conv_reduce_bwd(conv_shape);
            w_.put_adjoint(adj_w_, sink);
            x_.put_adjoint(adj_x_, sink);
            b_.put_adjoint(adj_b_, sink);
          //} else {
            //println!("WARNING: Conv2dAffineOp: missing adjoint for primal: {:?}", y_._graph_key());
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(Conv2dAffineOp{conv_shape}, ext, w_, x_, b_)))
  }
}

impl<T> LeftTransposeConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>> for Val<GPUDeviceArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
      GPUDeviceArrayViewMut4d<T>: GPUBatchLTransConvOps<T, T, T>,
{
  type ConvShape = Conv2dShape;

  fn left_transpose_conv(self, conv_shape: Conv2dShape, y: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    LeftTransposeConv2dLinearOp::build_device_obatch_val(conv_shape, self, y)
  }
}

impl LeftTransposeConv2dLinearOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv2dShape,
      w_: Val<GPUDeviceArray4d<T>>,
      y_: Val<GPUDeviceOuterBatchArray3d<T>>)
  -> Val<GPUDeviceOuterBatchArray3d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
        GPUDeviceArrayViewMut4d<T>: GPUBatchLTransConvOps<T, T, T>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    /*{
      let dst_size = conv_shape._calculate_output_size();
      let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
        ker_space_axes:   conv_shape.ker_space_axes,
        ker_output_axis:  conv_shape.ker_output_axis,
        src_space_axes:   conv_shape.src_space_axes,
        src_feature_axis: conv_shape.src_feature_axis,
        src_batch_axis:   conv_shape.src_batch_axis,
        // TODO: assumes NCHW layout.
        src_size:         [
          conv_shape.src_size[0],
          conv_shape.src_size[1],
          conv_shape.src_size[2],
          32,
        ],
        dst_space_axes:   conv_shape.dst_space_axes,
        dst_feature_axis: conv_shape.dst_feature_axis,
        dst_batch_axis:   conv_shape.dst_batch_axis,
        dst_size:         [
          dst_size[0],
          dst_size[1],
          dst_size[2],
          32,
        ],
        ker_size: conv_shape.ker_size,
        dilation: conv_shape.dilation,
        stride:   conv_shape.stride,
        zero_pad: conv_shape.zero_pad,
        groups:   1,
        cross:    true,
      });
      let mut state_cache = state_cache.borrow_mut();
      state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
        let ctx = implicit_ctx().gpu();
        let mut pool = ctx.pool();
        let conn = pool.conn();
        match query_gpu_conv_bwd_x_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
          None => panic!("invalid conv2d config: {:?}", xconv_shape),
          //Some((cfg, state)) => (cfg, state),
          Some((cfg, state)) => {
            conn.burst_arena().reserve_bytes(cfg.workspace_size());
            (cfg, state)
          }
        }
      });
    }*/
    let ext = OpExt{
      make_val: {
        //let w_ = w_.clone();
        let y_ = y_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          //let w_ = w_.clone();
          let y_ = y_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            //let w = w_.get(txn);
            let y = y_.get(txn);
            //guard._wait(w.async_state());
            guard._wait(y.async_state());
            //let y_size = y.size();
            let y_max_bsz = y.max_batch_size();
            let x_size = conv_shape.calculate_input_size();
            let x = GPUDeviceOuterBatchArray3d::zeros(x_size, y_max_bsz, conn);
            guard._wait(x.async_state());
            x
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let y_ = y_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let w = w_.get(txn).as_view();
            let y = y_.get(txn).as_view();
            let mut x = output.get_mut(txn, token).as_view_mut();
            guard._wait(w.async_state());
            guard._wait(y.async_state());
            guard._wait(x.async_state());
            /*assert_eq!(x.size()[0], conv_shape.src_size[0]);
            assert_eq!(x.size()[1], conv_shape.src_size[1]);
            assert_eq!(x.size()[2], conv_shape.src_size[2]);*/
            // TODO: set batch size.
            assert_eq!(x.size()[3], y.size()[3]);
            let xconv_shape = conv2d_to_xconv_shape(conv_shape, x.size()[3]);
            /*let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
              ker_space_axes:   conv_shape.ker_space_axes,
              ker_output_axis:  conv_shape.ker_output_axis,
              src_space_axes:   conv_shape.src_space_axes,
              src_feature_axis: conv_shape.src_feature_axis,
              src_batch_axis:   conv_shape.src_batch_axis,
              // TODO: assumes NCHW layout.
              //src_size:         conv_shape.src_size,
              src_size:         [
                conv_shape.src_size[0],
                conv_shape.src_size[1],
                conv_shape.src_size[2],
                y.size()[3],
              ],
              dst_space_axes:   conv_shape.dst_space_axes,
              dst_feature_axis: conv_shape.dst_feature_axis,
              dst_batch_axis:   conv_shape.dst_batch_axis,
              dst_size:         y.size(),
              ker_size: conv_shape.ker_size,
              dilation: conv_shape.dilation,
              stride:   conv_shape.stride,
              zero_pad: conv_shape.zero_pad,
              groups:   1,
              cross:    true,
            });*/
            let mut state_cache = state_cache.borrow_mut();
            let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
              match query_gpu_conv_bwd_x_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                None => panic!("invalid conv2d config"),
                Some((cfg, state)) => (cfg, state),
              }
            });
            let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
            guard._wait(workspace.async_state());
            match cap {
              WriteCap::Assign => {
                x.batch_left_transpose_conv2d(
                    &cfg,
                    state,
                    w,
                    y,
                    workspace.as_view_mut(),
                    conn.clone(),
                );
              }
              WriteCap::Accumulate => {
                x.batch_left_transpose_conv2d_accumulate(
                    &cfg,
                    state,
                    w,
                    y,
                    workspace.as_view_mut(),
                    conn.clone(),
                );
              }
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let y_ = y_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let y_ = y_.clone();
        Box::new(move |_: Pass, _this: Val<GPUDeviceOuterBatchArray3d<T>>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LeftTransposeConv2dLinearOp{conv_shape}, ext, w_, y_)))
  }
}

impl<T> OuterConvLinearExt<GPUDeviceArray4d<T>, GPUDeviceOuterBatchArray3d<T>, GPUDeviceOuterBatchArray3d<T>> for Val<GPUDeviceOuterBatchArray3d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
      GPUDeviceArrayViewMut4d<T>: GPUBatchOuterConvOps<T, T, T>,
{
  type ConvShape = Conv2dShape;

  fn outer_conv(self, conv_shape: Conv2dShape, x: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceArray4d<T>> {
    OuterConv2dLinearOp::build_device_obatch_val(conv_shape, self, x)
  }
}

impl OuterConv2dLinearOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv2dShape,
      y_: Val<GPUDeviceOuterBatchArray3d<T>>,
      x_: Val<GPUDeviceOuterBatchArray3d<T>>)
  -> Val<GPUDeviceArray4d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
        GPUDeviceArrayViewMut4d<T>: GPUBatchOuterConvOps<T, T, T>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    /*{
      let dst_size = conv_shape._calculate_output_size();
      let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
        ker_space_axes:   conv_shape.ker_space_axes,
        ker_output_axis:  conv_shape.ker_output_axis,
        src_space_axes:   conv_shape.src_space_axes,
        src_feature_axis: conv_shape.src_feature_axis,
        src_batch_axis:   conv_shape.src_batch_axis,
        // TODO: assumes NCHW layout.
        src_size:         [
          conv_shape.src_size[0],
          conv_shape.src_size[1],
          conv_shape.src_size[2],
          32,
        ],
        dst_space_axes:   conv_shape.dst_space_axes,
        dst_feature_axis: conv_shape.dst_feature_axis,
        dst_batch_axis:   conv_shape.dst_batch_axis,
        dst_size:         [
          dst_size[0],
          dst_size[1],
          dst_size[2],
          32,
        ],
        ker_size: conv_shape.ker_size,
        dilation: conv_shape.dilation,
        stride:   conv_shape.stride,
        zero_pad: conv_shape.zero_pad,
        groups:   1,
        cross:    true,
      });
      let mut state_cache = state_cache.borrow_mut();
      state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
        let ctx = implicit_ctx().gpu();
        let mut pool = ctx.pool();
        let conn = pool.conn();
        match query_gpu_conv_bwd_w_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
          None => panic!("invalid conv2d config: {:?}", xconv_shape),
          //Some((cfg, state)) => (cfg, state),
          Some((cfg, state)) => {
            conn.burst_arena().reserve_bytes(cfg.workspace_size());
            (cfg, state)
          }
        }
      });
    }*/
    let ext = OpExt{
      make_val: {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            // TODO: assumes NCHW layout.
            let w_size = [
              conv_shape.ker_dims[0],
              conv_shape.ker_dims[1],
              conv_shape.src_features,
              conv_shape.features,
            ];
            let w = GPUDeviceArray4d::zeros(w_size, conn);
            guard._wait(w.async_state());
            w
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let y_ = y_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceArray4d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: set batch size.
                let x = x_.get(txn);
                //assert_eq!(x.size(), conv_shape.src_size);
                let x = x.as_view();
                let y = y_.get(txn).as_view();
                let mut w = output.get_mut(txn, token).as_view_mut();
                guard._wait(y.async_state());
                guard._wait(x.async_state());
                guard._wait(w.async_state());
                /*let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
                  ker_space_axes:   conv_shape.ker_space_axes,
                  ker_output_axis:  conv_shape.ker_output_axis,
                  src_space_axes:   conv_shape.src_space_axes,
                  src_feature_axis: conv_shape.src_feature_axis,
                  src_batch_axis:   conv_shape.src_batch_axis,
                  // TODO: assumes NCHW layout.
                  //src_size:         conv_shape.src_size,
                  src_size:         [
                    conv_shape.src_size[0],
                    conv_shape.src_size[1],
                    conv_shape.src_size[2],
                    x.size()[3],
                  ],
                  dst_space_axes:   conv_shape.dst_space_axes,
                  dst_feature_axis: conv_shape.dst_feature_axis,
                  dst_batch_axis:   conv_shape.dst_batch_axis,
                  dst_size:         y.size(),
                  ker_size: conv_shape.ker_size,
                  dilation: conv_shape.dilation,
                  stride:   conv_shape.stride,
                  zero_pad: conv_shape.zero_pad,
                  groups:   1,
                  cross:    true,
                });*/
                // TODO: set batch size.
                assert_eq!(x.size()[3], y.size()[3]);
                let xconv_shape = conv2d_to_xconv_shape(conv_shape, x.size()[3]);
                let mut state_cache = state_cache.borrow_mut();
                let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
                  match query_gpu_conv_bwd_w_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                    None => panic!("invalid conv2d config"),
                    Some((cfg, state)) => (cfg, state),
                  }
                });
                let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
                guard._wait(workspace.async_state());
                w.batch_outer_conv2d(
                    &cfg,
                    state,
                    y,
                    x,
                    workspace.as_view_mut(),
                    conn.clone(),
                );
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<GPUDeviceArray4d<T>>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(OuterConv2dLinearOp{conv_shape}, ext, y_, x_)))
  }
}

impl<T> ConvReduceBwdExt<GPUDeviceOuterBatchArray3d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceOuterBatchArray3d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
      GPUDeviceArrayViewMut1d<T>: GPUBatchConvReduceOps<T, T, T>,
{
  type ConvShape = Conv2dShape;

  fn conv_reduce_bwd(self, conv_shape: Conv2dShape) -> Val<GPUDeviceArray1d<T>> {
    Conv2dReduceBwdOp::build_device_obatch_val(conv_shape, self)
  }
}

impl Conv2dReduceBwdOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv2dShape,
      x_: Val<GPUDeviceOuterBatchArray3d<T>>)
  -> Val<GPUDeviceArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnConvExt<T, T, T, HostScalar=T>,
        GPUDeviceArrayViewMut1d<T>: GPUBatchConvReduceOps<T, T, T>,
  {
    let ext = OpExt{
      make_val: {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let b_size = conv_shape.features;
            let b = GPUDeviceArray1d::zeros(b_size, conn);
            guard._wait(b.async_state());
            b
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let state_cache = RefCell::new(HashMap::new());
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let x = x_.get(txn).as_view();
                let mut b = output.get_mut(txn, token).as_view_mut();
                guard._wait(x.async_state());
                guard._wait(b.async_state());
                /*let xconv_shape = XConvFullShape::Conv2d(Conv2dFullShape{
                  ker_space_axes:   conv_shape.ker_space_axes,
                  ker_output_axis:  conv_shape.ker_output_axis,
                  src_space_axes:   conv_shape.src_space_axes,
                  src_feature_axis: conv_shape.src_feature_axis,
                  src_batch_axis:   conv_shape.src_batch_axis,
                  // TODO: assumes NCHW layout.
                  //src_size:         conv_shape.src_size,
                  src_size:         [
                    conv_shape.src_size[0],
                    conv_shape.src_size[1],
                    conv_shape.src_size[2],
                    x.size()[3],
                  ],
                  dst_space_axes:   conv_shape.dst_space_axes,
                  dst_feature_axis: conv_shape.dst_feature_axis,
                  dst_batch_axis:   conv_shape.dst_batch_axis,
                  dst_size:         x.size(),
                  ker_size: conv_shape.ker_size,
                  dilation: conv_shape.dilation,
                  stride:   conv_shape.stride,
                  zero_pad: conv_shape.zero_pad,
                  groups:   1,
                  cross:    true,
                });*/
                let xconv_shape = conv2d_to_xconv_shape(conv_shape, x.size()[3]);
                let mut state_cache = state_cache.borrow_mut();
                let state = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
                  match query_gpu_conv_bwd_b_state(conn.device(), None, None, xconv_shape, conn.clone()) {
                    None => panic!("invalid conv2d config"),
                    Some(state) => state,
                  }
                });
                b.batch_conv2d_reduce_bwd(
                    state,
                    x,
                    conn.clone(),
                );
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<GPUDeviceArray1d<T>>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(Conv2dReduceBwdOp{conv_shape}, ext, x_)))
  }
}

fn conv3d_to_xconv_shape(conv_shape: Conv3dShape, batch_sz: usize) -> XConvFullShape {
  let src_size = conv_shape.calculate_input_size();
  let dst_size = conv_shape.calculate_output_size();
  XConvFullShape::Conv3d(Conv3dFullShape{
    ker_space_axes:   conv_shape.ker_space_axes,
    ker_output_axis:  conv_shape.ker_output_axis,
    src_space_axes:   conv_shape.src_space_axes,
    src_feature_axis: conv_shape.src_feature_axis,
    src_batch_axis:   conv_shape.src_batch_axis,
    // TODO: assumes NCHW layout.
    src_size:         [
      /*conv_shape.src_dims[0],
      conv_shape.src_dims[1],
      conv_shape.src_dims[2],
      conv_shape.src_features,*/
      src_size[0],
      src_size[1],
      src_size[2],
      src_size[3],
      batch_sz,
    ],
    dst_space_axes:   conv_shape.dst_space_axes,
    dst_feature_axis: conv_shape.dst_feature_axis,
    dst_batch_axis:   conv_shape.dst_batch_axis,
    dst_size:         [
      dst_size[0],
      dst_size[1],
      dst_size[2],
      dst_size[3],
      batch_sz,
    ],
    ker_size: conv_shape.ker_dims,
    dilation: conv_shape.dilation,
    stride:   conv_shape.stride,
    zero_pad: conv_shape.zero_pad,
    groups:   1,
    cross:    true,
  })
}

impl<T> ConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>> for Val<GPUDeviceArray5d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      CudnnHandle: CudnnConvExt<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
      //Val<GPUDeviceOuterBatchArray4d<T>>: OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
      //Val<GPUDeviceArray5d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
{
  type ConvShape = Conv3dShape;

  fn conv(self, conv_shape: Conv3dShape, x: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    Conv3dLinearOp::build_device_obatch_val(conv_shape, self, x)
  }
}

impl<T> ConvAffineExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray5d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      CudnnHandle: CudnnConvExt<T, T, T>,
      GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
      GPUDeviceArrayViewMut5d<T>: GPUTensorMutOps<T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
      //Val<GPUDeviceOuterBatchArray4d<T>>: OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
      //Val<GPUDeviceArray5d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
      //Val<GPUDeviceOuterBatchArray4d<T>>: ConvReduceBwdExt<GPUDeviceOuterBatchArray4d<T>, GPUDeviceArray1d<T>, ConvShape=Conv3dShape>,
{
  fn conv_add(self, conv_shape: Conv3dShape, x: Val<GPUDeviceOuterBatchArray4d<T>>, b: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    Conv3dAffineOp::build_device_obatch_val(conv_shape, self, x, b)
  }
}

impl Conv3dLinearOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv3dShape,
      w_: Val<GPUDeviceArray5d<T>>,
      x_: Val<GPUDeviceOuterBatchArray4d<T>>)
  -> Val<GPUDeviceOuterBatchArray4d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        CudnnHandle: CudnnConvExt<T, T, T>,
        //GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
        //GPUDeviceArrayViewMut5d<T>: GPUTensorMutOps<T> + GPUBatchConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
        //Val<GPUDeviceOuterBatchArray4d<T>>: OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
        //Val<GPUDeviceArray5d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    let ext = OpExt{
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            assert_eq!(x.size(), conv_shape.calculate_input_size());
            let x_max_bsz = x.max_batch_size();
            let y_size = conv_shape.calculate_output_size();
            let y = GPUDeviceOuterBatchArray4d::zeros(y_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let w = w_.get(txn).as_view();
            let x = x_.get(txn).as_view();
            let mut y = output.get_mut(txn, token).as_view_mut();
            guard._wait(w.async_state());
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            // TODO: set batch size.
            assert_eq!(x.size()[4], y.size()[4]);
            let xconv_shape = conv3d_to_xconv_shape(conv_shape, x.size()[4]);
            let mut state_cache = state_cache.borrow_mut();
            let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
              match query_gpu_conv_fwd_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                None => panic!("invalid conv2d config: {:?}", xconv_shape),
                Some((cfg, state)) => (cfg, state),
              }
            });
            let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
            guard._wait(workspace.async_state());
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
            y.batch_conv3d(
                &cfg,
                state,
                alpha,
                w,
                x,
                beta,
                workspace.as_view_mut(),
                conn.clone(),
            );
            //double_check_scalar::<Self, _>(|| y.flat_view().unwrap().sync_vector_norm(conn).into());
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_w_ = adj_y_.clone().outer_conv(conv_shape, x_.clone());
            let adj_x_ = w_.clone().left_transpose_conv(conv_shape, adj_y_.clone());
            w_.put_adjoint(adj_w_, sink);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(Conv3dLinearOp{conv_shape}, ext, w_, x_)))
  }
}

impl Conv3dAffineOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv3dShape,
      w_: Val<GPUDeviceArray5d<T>>,
      x_: Val<GPUDeviceOuterBatchArray4d<T>>,
      b_: Val<GPUDeviceArray1d<T>>)
  -> Val<GPUDeviceOuterBatchArray4d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        CudnnHandle: CudnnConvExt<T, T, T>,
        GPUDeviceArrayView1d<T>: GPUVectorOps<T>,
        GPUDeviceArrayViewMut5d<T>: GPUTensorMutOps<T> + GPUBatchConv3dOps<T, T, T>,
        //GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
        //Val<GPUDeviceOuterBatchArray4d<T>>: OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
        //Val<GPUDeviceArray5d<T>>: LeftTransposeConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    let ext = OpExt{
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let w_ = w_.clone();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let w_size = w_.get(txn).size();
            let x_size = x_.get(txn).size();
            let x_max_bsz = x_.get(txn).max_batch_size();
            let y_size = conv_shape.calculate_output_size();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = GPUDeviceOuterBatchArray4d::zeros(y_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let w = w_.get(txn).as_view();
            let x = x_.get(txn).as_view();
            let b = b_.get(txn).as_view();
            let mut y = output.get_mut(txn, token).as_view_mut();
            guard._wait(w.async_state());
            guard._wait(x.async_state());
            guard._wait(b.async_state());
            guard._wait(y.async_state());
            // TODO: set batch size.
            assert_eq!(x.size()[4], y.size()[4]);
            let xconv_shape = conv3d_to_xconv_shape(conv_shape, x.size()[4]);
            let mut state_cache = state_cache.borrow_mut();
            let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
              match query_gpu_conv_fwd_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                None => panic!("invalid conv2d config: {:?}", xconv_shape),
                Some((cfg, state)) => (cfg, state),
              }
            });
            let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
            guard._wait(workspace.async_state());
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
            y.batch_conv3d(
                &cfg,
                state,
                alpha,
                w,
                x,
                beta,
                workspace.as_view_mut(),
                conn.clone(),
            );
            y.broadcast_add_1d_inplace(b, conv_shape.dst_feature_axis, conn.clone());
            double_check_scalar::<Self, _>(|| y.flat_view().unwrap().sync_vector_norm(conn).into());
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        let b_ = b_.clone();
        Box::new(move |_: Pass, y_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_w_ = adj_y_.clone().outer_conv(conv_shape, x_.clone());
            let adj_x_ = w_.clone().left_transpose_conv(conv_shape, adj_y_.clone());
            let adj_b_ = adj_y_.clone().conv_reduce_bwd(conv_shape);
            w_.put_adjoint(adj_w_, sink);
            x_.put_adjoint(adj_x_, sink);
            b_.put_adjoint(adj_b_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(Conv3dAffineOp{conv_shape}, ext, w_, x_, b_)))
  }
}

impl<T> LeftTransposeConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>> for Val<GPUDeviceArray5d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      CudnnHandle: CudnnConvExt<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
      //Val<GPUDeviceOuterBatchArray4d<T>>: OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
      //Val<GPUDeviceArray5d<T>>: ConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
{
  type ConvShape = Conv3dShape;

  fn left_transpose_conv(self, conv_shape: Conv3dShape, y: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    LeftTransposeConv3dLinearOp::build_device_obatch_val(conv_shape, self, y)
  }
}

impl LeftTransposeConv3dLinearOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv3dShape,
      w_: Val<GPUDeviceArray5d<T>>,
      y_: Val<GPUDeviceOuterBatchArray4d<T>>)
  -> Val<GPUDeviceOuterBatchArray4d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        CudnnHandle: CudnnConvExt<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
        //Val<GPUDeviceOuterBatchArray4d<T>>: OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
        //Val<GPUDeviceArray5d<T>>: ConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>, ConvShape=Conv3dShape>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    let ext = OpExt{
      make_val: {
        let y_ = y_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let y_ = y_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = y_.get(txn);
            guard._wait(y.async_state());
            let max_bsz = y.max_batch_size();
            let x_size = conv_shape.calculate_input_size();
            let x = GPUDeviceOuterBatchArray4d::zeros(x_size, max_bsz, conn);
            guard._wait(x.async_state());
            x
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let w_ = w_.clone();
        let y_ = y_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let w = w_.get(txn).as_view();
            let y = y_.get(txn).as_view();
            let mut x = output.get_mut(txn, token).as_view_mut();
            guard._wait(w.async_state());
            guard._wait(y.async_state());
            guard._wait(x.async_state());
            // TODO: set batch size.
            assert_eq!(x.size()[4], y.size()[4]);
            let xconv_shape = conv3d_to_xconv_shape(conv_shape, x.size()[4]);
            let mut state_cache = state_cache.borrow_mut();
            let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
              match query_gpu_conv_bwd_x_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                None => panic!("invalid conv2d config"),
                Some((cfg, state)) => (cfg, state),
              }
            });
            let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
            guard._wait(workspace.async_state());
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
            x.batch_left_transpose_conv3d(
                &cfg,
                state,
                alpha,
                w,
                y,
                beta,
                workspace.as_view_mut(),
                conn.clone(),
            );
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        let w_ = w_.clone();
        let y_ = y_.clone();
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let y_ = y_.clone();
        Box::new(move |_: Pass, x_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_x_) = x_.adjoint(sink) {
            let adj_w_ = y_.clone().outer_conv(conv_shape, adj_x_.clone());
            let adj_y_ = w_.clone().conv(conv_shape, adj_x_.clone());
            w_.put_adjoint(adj_w_, sink);
            y_.put_adjoint(adj_y_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LeftTransposeConv3dLinearOp{conv_shape}, ext, w_, y_)))
  }
}

impl<T> OuterConvLinearExt<GPUDeviceArray5d<T>, GPUDeviceOuterBatchArray4d<T>, GPUDeviceOuterBatchArray4d<T>> for Val<GPUDeviceOuterBatchArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      CudnnHandle: CudnnConvExt<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
{
  type ConvShape = Conv3dShape;

  fn outer_conv(self, conv_shape: Conv3dShape, x: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceArray5d<T>> {
    OuterConv3dLinearOp::build_device_obatch_val(conv_shape, self, x)
  }
}

impl OuterConv3dLinearOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv3dShape,
      y_: Val<GPUDeviceOuterBatchArray4d<T>>,
      x_: Val<GPUDeviceOuterBatchArray4d<T>>)
  -> Val<GPUDeviceArray5d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        CudnnHandle: CudnnConvExt<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
  {
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    let ext = OpExt{
      make_val: {
        Box::new(move |_state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // TODO: assumes NCHW layout.
            let w_size = [
              conv_shape.ker_dims[0],
              conv_shape.ker_dims[1],
              conv_shape.ker_dims[2],
              conv_shape.src_features,
              conv_shape.features,
            ];
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let w = GPUDeviceArray5d::zeros(w_size, conn);
            guard._wait(w.async_state());
            w
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let y_ = y_.clone();
        let x_ = x_.clone();
        //let state_cache = RefCell::new(HashMap::new());
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = y_.get(txn).as_view();
            let x = x_.get(txn).as_view();
            let mut w = output.get_mut(txn, token).as_view_mut();
            guard._wait(y.async_state());
            guard._wait(x.async_state());
            guard._wait(w.async_state());
            // TODO: set batch size.
            assert_eq!(x.size()[4], y.size()[4]);
            let xconv_shape = conv3d_to_xconv_shape(conv_shape, x.size()[4]);
            let mut state_cache = state_cache.borrow_mut();
            let &mut (cfg, ref mut state) = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
              match query_gpu_conv_bwd_w_algo(conn.device(), None, None, xconv_shape, conn.clone()) {
                None => panic!("invalid conv2d config: {:?}", xconv_shape),
                Some((cfg, state)) => (cfg, state),
              }
            });
            let mut workspace = unsafe { GPUDeviceArray1d::alloc(conn.burst_arena(), cfg.workspace_size(), conn.clone()) };
            guard._wait(workspace.async_state());
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
            w.batch_outer_conv3d(
                &cfg,
                state,
                alpha,
                y,
                x,
                beta,
                workspace.as_view_mut(),
                conn.clone(),
            );
            /*double_check_scalar::<Self, _>(|| w.flat_view().unwrap().sync_vector_norm(conn).into());*/
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let y_ = y_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, w_: Val<_>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_w_) = w_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(OuterConv3dLinearOp{conv_shape}, ext, y_, x_)))
  }
}

impl<T> ConvReduceBwdExt<GPUDeviceOuterBatchArray4d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceOuterBatchArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
      CudnnHandle: CudnnConvExt<T, T, T>,
      GPUDeviceArrayViewMut1d<T>: GPUBatchConvReduceOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
{
  type ConvShape = Conv3dShape;

  fn conv_reduce_bwd(self, conv_shape: Conv3dShape) -> Val<GPUDeviceArray1d<T>> {
    Conv3dReduceBwdOp::build_device_obatch_val(conv_shape, self)
  }
}

impl Conv3dReduceBwdOp {
  pub fn build_device_obatch_val<T>(
      conv_shape: Conv3dShape,
      x_: Val<GPUDeviceOuterBatchArray4d<T>>)
  -> Val<GPUDeviceArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static + Into<f32>,
        CudnnHandle: CudnnConvExt<T, T, T>,
        GPUDeviceArrayViewMut1d<T>: GPUBatchConvReduceOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchLTransConv3dOps<T, T, T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchOuterConv3dOps<T, T, T>,
  {
    let ext = OpExt{
      make_val: {
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let b_size = conv_shape.features;
            let b = GPUDeviceArray1d::zeros(b_size, conn);
            guard._wait(b.async_state());
            b
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let state_cache = RefCell::new(HashMap::new());
        Box::new(move |txn, _state: RefMut<_>, output: OVal<_>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
                let x = x_.get(txn).as_view();
                let mut b = output.get_mut(txn, token).as_view_mut();
                guard._wait(x.async_state());
                guard._wait(b.async_state());
                let xconv_shape = conv3d_to_xconv_shape(conv_shape, x.size()[4]);
                let mut state_cache = state_cache.borrow_mut();
                let state = state_cache.entry(xconv_shape.clone()).or_insert_with(|| {
                  match query_gpu_conv_bwd_b_state(conn.device(), None, None, xconv_shape, conn.clone()) {
                    None => panic!("invalid conv2d config"),
                    Some(state) => state,
                  }
                });
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
                b.batch_conv3d_reduce_bwd(
                    state,
                    alpha,
                    x,
                    beta,
                    conn.clone(),
                );
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<_>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(Conv3dReduceBwdOp{conv_shape}, ext, x_)))
  }
}

pub trait PoolOp {
  fn to_xop(&self) -> XPoolOp;
}

impl PoolOp for AveragePool {
  fn to_xop(&self) -> XPoolOp {
    XPoolOp::Average
  }
}

impl PoolOp for MaxPool {
  fn to_xop(&self) -> XPoolOp {
    XPoolOp::Max
  }
}

impl<T> PoolExt<GPUDeviceOuterBatchArray3d<T>> for Val<GPUDeviceOuterBatchArray3d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnPoolExt<T, HostScalar=T>,
      GPUDeviceArrayViewMut4d<T>: GPUBatchPoolOps<T>,
{
  type PoolShape = Pool2dShape;

  fn average_pool(self, pool_shape: Pool2dShape) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    Pool2dOp::<AveragePool>::build_device_obatch_val(AveragePool, pool_shape, self)
  }

  fn max_pool(self, pool_shape: Pool2dShape) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    Pool2dOp::<MaxPool>::build_device_obatch_val(MaxPool, pool_shape, self)
  }
}

impl<T> PoolBwdExt<GPUDeviceOuterBatchArray3d<T>> for Val<GPUDeviceOuterBatchArray3d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnPoolExt<T, HostScalar=T>,
      GPUDeviceArrayViewMut4d<T>: GPUBatchPoolOps<T>,
{
  fn average_pool_bwd(self, pool_shape: Pool2dShape, y_: Val<GPUDeviceOuterBatchArray3d<T>>, x_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    Pool2dBwdOp::<AveragePool>::build_device_obatch_val(AveragePool, pool_shape, self, y_, x_)
  }

  fn max_pool_bwd(self, pool_shape: Pool2dShape, y_: Val<GPUDeviceOuterBatchArray3d<T>>, x_: Val<GPUDeviceOuterBatchArray3d<T>>) -> Val<GPUDeviceOuterBatchArray3d<T>> {
    Pool2dBwdOp::<MaxPool>::build_device_obatch_val(MaxPool, pool_shape, self, y_, x_)
  }
}

impl<Pool: PoolOp + 'static> Pool2dOp<Pool> {
  pub fn build_device_obatch_val<T>(
      pool: Pool,
      pool_shape: Pool2dShape,
      x_: Val<GPUDeviceOuterBatchArray3d<T>>)
  -> Val<GPUDeviceOuterBatchArray3d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnPoolExt<T>,
        GPUDeviceArrayViewMut4d<T>: GPUBatchPoolOps<T>,
        Val<GPUDeviceOuterBatchArray3d<T>>: SomePoolBwdExt<Pool, GPUDeviceOuterBatchArray3d<T>, PoolShape=Pool2dShape>,
  {
    let xpool_op = pool.to_xop();
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_max_bsz = x.max_batch_size();
            let y_size = pool_shape.calculate_output_size(x.size());
            let y = GPUDeviceOuterBatchArray3d::zeros(y_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let state_cache = RefCell::new(HashMap::new());
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let x = x_.get(txn).as_view();
                let mut y = output.get_mut(txn, token).as_view_mut();
                guard._wait(x.async_state());
                guard._wait(y.async_state());
                assert_eq!(x.size()[0], pool_shape.src_size[0]);
                assert_eq!(x.size()[1], pool_shape.src_size[1]);
                assert_eq!(x.size()[2], pool_shape.src_features);
                // TODO: assumes NCHW layout.
                let y_size = pool_shape.calculate_output_size([
                    pool_shape.src_size[0],
                    pool_shape.src_size[1],
                    pool_shape.src_features,
                ]);
                assert_eq!(y.size()[0], y_size[0]);
                assert_eq!(y.size()[1], y_size[1]);
                assert_eq!(y.size()[2], y_size[2]);
                // TODO: set batch size.
                assert_eq!(x.size()[3], y.size()[3]);
                //y.set_batch_size(x.batch_size());
                let xpool_shape = XPoolFullShape::Pool2d(Pool2dFullShape{
                  src_space_axes:   pool_shape.src_space_axes,
                  src_feature_axis: pool_shape.src_feature_axis,
                  src_batch_axis:   pool_shape.src_batch_axis,
                  // TODO: assumes NCHW layout.
                  src_size:         [
                    pool_shape.src_size[0],
                    pool_shape.src_size[1],
                    pool_shape.src_features,
                    x.size()[3],
                  ],
                  dst_space_axes:   pool_shape.dst_space_axes,
                  dst_feature_axis: pool_shape.dst_feature_axis,
                  dst_batch_axis:   pool_shape.dst_batch_axis,
                  // TODO: assumes NCHW layout.
                  dst_size:         [
                    y_size[0],
                    y_size[1],
                    pool_shape.src_features,
                    y.size()[3],
                  ],
                  ker_size:         pool_shape.ker_size,
                  stride:           pool_shape.stride,
                  zero_pad:         pool_shape.zero_pad,
                });
                let mut state_cache = state_cache.borrow_mut();
                let state = state_cache.entry(xpool_shape.clone()).or_insert_with(|| {
                  match query_gpu_pool_state(conn.device(), xpool_op, xpool_shape, conn.clone()) {
                    None => panic!("invalid pool2d config"),
                    Some(state) => state,
                  }
                });
                y.batch_pool2d(
                    state,
                    x,
                    conn.clone(),
                );
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray3d<T>>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = adj_y_.pool_bwd(pool_shape, y_.clone(), x_.clone());
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(Pool2dOp{pool, pool_shape}, ext, x_)))
  }
}

impl<Pool: PoolOp + 'static> Pool2dBwdOp<Pool> {
  pub fn build_device_obatch_val<T>(
      pool: Pool,
      pool_shape: Pool2dShape,
      dy_: Val<GPUDeviceOuterBatchArray3d<T>>,
      y_: Val<GPUDeviceOuterBatchArray3d<T>>,
      x_: Val<GPUDeviceOuterBatchArray3d<T>>)
  -> Val<GPUDeviceOuterBatchArray3d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnPoolExt<T>,
        GPUDeviceArrayViewMut4d<T>: GPUBatchPoolOps<T>,
  {
    let xpool_op = pool.to_xop();
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_shape = x.shape();
            let dx = GPUDeviceOuterBatchArray3d::zeros_shape(x_shape, conn);
            guard._wait(dx.async_state());
            dx
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = dy_.clone();
        let y_ = y_.clone();
        let x_ = x_.clone();
        let state_cache = RefCell::new(HashMap::new());
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray3d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                let dy = dy_.get(txn).as_view();
                let y = y_.get(txn).as_view();
                let x = x_.get(txn).as_view();
                let mut dx = output.get_mut(txn, token).as_view_mut();
                guard._wait(dy.async_state());
                guard._wait(y.async_state());
                guard._wait(x.async_state());
                guard._wait(dx.async_state());
                // TODO: assumes NCHW layout.
                let y_size = pool_shape.calculate_output_size([
                    pool_shape.src_size[0],
                    pool_shape.src_size[1],
                    pool_shape.src_features,
                ]);
                assert_eq!(dy.size()[0], y_size[0]);
                assert_eq!(dy.size()[1], y_size[1]);
                assert_eq!(dy.size()[2], y_size[2]);
                assert_eq!(y.size()[0], y_size[0]);
                assert_eq!(y.size()[1], y_size[1]);
                assert_eq!(y.size()[2], y_size[2]);
                assert_eq!(x.size()[0], pool_shape.src_size[0]);
                assert_eq!(x.size()[1], pool_shape.src_size[1]);
                assert_eq!(x.size()[2], pool_shape.src_features);
                assert_eq!(dx.size()[0], pool_shape.src_size[0]);
                assert_eq!(dx.size()[1], pool_shape.src_size[1]);
                assert_eq!(dx.size()[2], pool_shape.src_features);
                // TODO: set batch size.
                assert_eq!(dy.size()[3], y.size()[3]);
                assert_eq!(dy.size()[3], x.size()[3]);
                assert_eq!(dy.size()[3], dx.size()[3]);
                let xpool_shape = XPoolFullShape::Pool2d(Pool2dFullShape{
                  src_space_axes:   pool_shape.src_space_axes,
                  src_feature_axis: pool_shape.src_feature_axis,
                  src_batch_axis:   pool_shape.src_batch_axis,
                  // TODO: assumes NCHW layout.
                  src_size:         [
                    pool_shape.src_size[0],
                    pool_shape.src_size[1],
                    pool_shape.src_features,
                    dy.size()[3],
                  ],
                  dst_space_axes:   pool_shape.dst_space_axes,
                  dst_feature_axis: pool_shape.dst_feature_axis,
                  dst_batch_axis:   pool_shape.dst_batch_axis,
                  // TODO: assumes NCHW layout.
                  dst_size:         [
                    y_size[0],
                    y_size[1],
                    pool_shape.src_features,
                    dy.size()[3],
                  ],
                  ker_size:         pool_shape.ker_size,
                  stride:           pool_shape.stride,
                  zero_pad:         pool_shape.zero_pad,
                });
                let mut state_cache = state_cache.borrow_mut();
                let state = state_cache.entry(xpool_shape.clone()).or_insert_with(|| {
                  match query_gpu_pool_state(conn.device(), xpool_op, xpool_shape, conn.clone()) {
                    None => panic!("invalid pool2d config"),
                    Some(state) => state,
                  }
                });
                dx.batch_pool2d_bwd(
                    state,
                    y,
                    dy,
                    x,
                    conn.clone(),
                );
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<GPUDeviceOuterBatchArray3d<T>>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(Pool2dBwdOp{pool, pool_shape}, ext, dy_, y_, x_)))
  }
}

impl<T> PoolExt<GPUDeviceOuterBatchArray4d<T>> for Val<GPUDeviceOuterBatchArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnPoolExt<T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchPool3dOps<T>,
{
  type PoolShape = Pool3dShape;

  fn average_pool(self, pool_shape: Pool3dShape) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    Pool3dOp::<AveragePool>::build_device_obatch_val(AveragePool, pool_shape, self)
  }

  fn max_pool(self, pool_shape: Pool3dShape) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    Pool3dOp::<MaxPool>::build_device_obatch_val(MaxPool, pool_shape, self)
  }
}

impl<T> PoolBwdExt<GPUDeviceOuterBatchArray4d<T>> for Val<GPUDeviceOuterBatchArray4d<T>>
where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
      CudnnHandle: CudnnPoolExt<T>,
      GPUDeviceArrayViewMut5d<T>: GPUBatchPool3dOps<T>,
{
  fn average_pool_bwd(self, pool_shape: Pool3dShape, y_: Val<GPUDeviceOuterBatchArray4d<T>>, x_: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    Pool3dBwdOp::<AveragePool>::build_device_obatch_val(AveragePool, pool_shape, self, y_, x_)
  }

  fn max_pool_bwd(self, pool_shape: Pool3dShape, y_: Val<GPUDeviceOuterBatchArray4d<T>>, x_: Val<GPUDeviceOuterBatchArray4d<T>>) -> Val<GPUDeviceOuterBatchArray4d<T>> {
    Pool3dBwdOp::<MaxPool>::build_device_obatch_val(MaxPool, pool_shape, self, y_, x_)
  }
}

fn pool3d_to_xpool_shape(pool_shape: Pool3dShape, batch_sz: usize) -> XPoolFullShape {
  let src_size = pool_shape.calculate_input_size();
  let dst_size = pool_shape.calculate_output_size();
  XPoolFullShape::Pool3d(Pool3dFullShape{
    src_space_axes:   pool_shape.src_space_axes,
    src_feature_axis: pool_shape.src_feature_axis,
    src_batch_axis:   pool_shape.src_batch_axis,
    // TODO: assumes NCHW layout.
    src_size:         [
      src_size[0],
      src_size[1],
      src_size[2],
      src_size[3],
      batch_sz,
    ],
    dst_space_axes:   pool_shape.dst_space_axes,
    dst_feature_axis: pool_shape.dst_feature_axis,
    dst_batch_axis:   pool_shape.dst_batch_axis,
    // TODO: assumes NCHW layout.
    dst_size:         [
      dst_size[0],
      dst_size[1],
      dst_size[2],
      dst_size[3],
      batch_sz,
    ],
    ker_size: pool_shape.ker_dims,
    stride:   pool_shape.stride,
    zero_pad: pool_shape.zero_pad,
  })
}

impl<Pool: PoolOp + 'static> Pool3dOp<Pool> {
  pub fn build_device_obatch_val<T>(
      pool: Pool,
      pool_shape: Pool3dShape,
      x_: Val<GPUDeviceOuterBatchArray4d<T>>)
  -> Val<GPUDeviceOuterBatchArray4d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnPoolExt<T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchPool3dOps<T>,
        Val<GPUDeviceOuterBatchArray4d<T>>: SomePoolBwdExt<Pool, GPUDeviceOuterBatchArray4d<T>, PoolShape=Pool3dShape>,
  {
    let xpool_op = pool.to_xop();
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            assert_eq!(x.size(), pool_shape.calculate_input_size());
            let x_max_bsz = x.max_batch_size();
            let y_size = pool_shape.calculate_output_size();
            let y = GPUDeviceOuterBatchArray4d::zeros(y_size, x_max_bsz, conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let x_ = x_.clone();
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray4d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn).as_view();
            let mut y = output.get_mut(txn, token).as_view_mut();
            guard._wait(x.async_state());
            guard._wait(y.async_state());
            // TODO: set batch size.
            assert_eq!(x.size()[4], y.size()[4]);
            //y.set_batch_size(x.batch_size());
            let xpool_shape = pool3d_to_xpool_shape(pool_shape, x.size()[4]);
            let mut state_cache = state_cache.borrow_mut();
            let state = state_cache.entry(xpool_shape.clone()).or_insert_with(|| {
              match query_gpu_pool_state(conn.device(), xpool_op, xpool_shape, conn.clone()) {
                None => panic!("invalid pool2d config"),
                Some(state) => state,
              }
            });
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
            y.batch_pool3d(
                state,
                alpha,
                x,
                beta,
                conn.clone(),
            );
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray4d<T>>, _state: RefMut<_>, sink: &mut Sink| {
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = adj_y_.pool_bwd(pool_shape, y_.clone(), x_.clone());
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F1Op::new(Pool3dOp{pool, pool_shape}, ext, x_)))
  }
}

impl<Pool: PoolOp + 'static> Pool3dBwdOp<Pool> {
  pub fn build_device_obatch_val<T>(
      pool: Pool,
      pool_shape: Pool3dShape,
      dy_: Val<GPUDeviceOuterBatchArray4d<T>>,
      y_: Val<GPUDeviceOuterBatchArray4d<T>>,
      x_: Val<GPUDeviceOuterBatchArray4d<T>>)
  -> Val<GPUDeviceOuterBatchArray4d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: GPUDataTyped + CudnnDataTypeExt + Zero + One + ZeroBits + Copy + 'static,
        CudnnHandle: CudnnPoolExt<T>,
        GPUDeviceArrayViewMut5d<T>: GPUBatchPool3dOps<T>,
  {
    let xpool_op = pool.to_xop();
    let state_cache = Rc::new(RefCell::new(HashMap::new()));
    let ext = OpExt{
      make_val: {
        let x_ = x_.clone();
        Box::new(move |state: RefMut<_>| {
          let section = GPULazyAsyncSection::default();
          let x_ = x_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x = x_.get(txn);
            guard._wait(x.async_state());
            let x_shape = x.shape();
            let dx = GPUDeviceOuterBatchArray4d::zeros_shape(x_shape, conn);
            guard._wait(dx.async_state());
            dx
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let dy_ = dy_.clone();
        let y_ = y_.clone();
        let x_ = x_.clone();
        let state_cache = state_cache.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceOuterBatchArray4d<T>>| {
          //if let Some((cap, token)) = output.write(txn) {
          output.write(txn, |cap, token| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let dy = dy_.get(txn).as_view();
            let y = y_.get(txn).as_view();
            let x = x_.get(txn).as_view();
            let mut dx = output.get_mut(txn, token).as_view_mut();
            guard._wait(dy.async_state());
            guard._wait(y.async_state());
            guard._wait(x.async_state());
            guard._wait(dx.async_state());
            // TODO: set batch size.
            assert_eq!(dy.size()[4], y.size()[4]);
            assert_eq!(dy.size()[4], x.size()[4]);
            assert_eq!(dy.size()[4], dx.size()[4]);
            let xpool_shape = pool3d_to_xpool_shape(pool_shape, dy.size()[4]);
            let mut state_cache = state_cache.borrow_mut();
            let state = state_cache.entry(xpool_shape.clone()).or_insert_with(|| {
              match query_gpu_pool_state(conn.device(), xpool_op, xpool_shape, conn.clone()) {
                None => panic!("invalid pool2d config"),
                Some(state) => state,
              }
            });
            let (alpha, beta) = match cap {
              WriteCap::Assign => (one(), zero()),
              WriteCap::Accumulate => (one(), one()),
            };
            dx.batch_pool3d_bwd(
                state,
                alpha,
                y,
                dy,
                x,
                beta,
                conn.clone(),
            );
          })
        })
      },
      build: Some({
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      }),
      tangent: Some({
        Box::new(move |_: Pass, _state: RefMut<_>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |_: Pass, _this: Val<GPUDeviceOuterBatchArray4d<T>>, _state: RefMut<_>, _sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F3Op::new(Pool3dBwdOp{pool, pool_shape}, ext, dy_, y_, x_)))
  }
}
