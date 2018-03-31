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

use ::*;
use context::*;
use ffi::routines_gpu::*;
use ops::*;

use arithmetic::*;
use cuda_blas::*;
use cuda_dnn::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;

use std::cell::{RefMut};
use std::marker::{PhantomData};
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::sync::{Arc};

#[inline]
pub fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

pub trait GPUDeviceMemIoReader<'a> {
  fn read_dev_mem(&mut self, src: &'a Any) -> Option<()>;
}

pub trait GPUDeviceMemIoWriter<'a> {
  fn write_dev_mem(&mut self, cap: WriteCap, dst: &'a mut Any) -> Option<()>;
}

pub struct GPUMuxFun<A> {
  pub dev:  GPUDeviceId,
  pub val:  Val<A>,
}

impl<A> GPUMuxFun<A> where A: 'static {
  pub fn build_ext() -> OpExt<GPUMuxFun<A>, A> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        Box::new(move |/*state: RefMut<GPUMuxFun<A>>*/| {
          unimplemented!();
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        Box::new(move |txn: Txn, state: RefMut<GPUMuxFun<A>>, _output: OVal<A>| {
          let ctx = implicit_ctx().multi_gpu().unwrap().gpu(state.dev);
          let guard = push_ctx(ctx);
          state.val._apply(txn);
        })
      },
      tangent: None,
      //adjoint: None,
      adjoint: Some({
        Box::new(move |x_: Val<A>, sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: None,
    };
    ext
  }
}

impl<A, F> SrcOpExt<A, Rc<F>> for SrcOp
where A: 'static, F: (Fn(GPUDeviceStreamPool) -> A) + 'static,
{
  //fn build(init_gpu_val: Rc<F>) -> Rc<FSrcOp<SrcOp, A>> {
  fn build(init_gpu_val: Rc<F>) -> Val<A> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        Box::new(move || {
        //Box::new(move |state: RefMut<SrcOp>| {
          println!("DEBUG: SrcOpExt: init gpu...");
          let init_gpu_val = init_gpu_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_gpu_val(pool)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          if let Some(_) = output.write(txn) {
            panic!("WARNING: SrcOpExt: should never write");
          }
        })
      },
      tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |x_: Val<A>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    let value = (ext.init)();
    Val::from(Rc::new(FSrcOp::new(SrcOp, ext, value)))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      //F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray1d<T>) + 'static,
      F: (Fn(GPUDeviceConn) -> GPUDeviceArray1d<T>) + 'static,
{
  //fn build(init_val: Rc<F>) -> Rc<FSrcOp<ZerosSrcOp, GPUDeviceArray1d<T>>> {
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray1d<T>> {
    let ctx = implicit_ctx().gpu().unwrap();
    let pool = ctx.pool();
    let section = GPUAsyncSection::new(pool.conn());
    let init_section = section.clone();
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        Box::new(move || {
        //Box::new(move |state: RefMut<ZerosSrcOp>| {
          println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: init...");
          let init_val = init_val.clone();
          let section = init_section.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: init: allocating...");
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut guard = section.enter(conn.clone());
            init_val(conn)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            //println!("DEBUG: ZeroSrcOp: zeroing (GPUDeviceArray1d)...");
            println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: apply: writing...");
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Assign => {
                // TODO: Because async sections do not currently use reentrant
                // mutexes for events, we need to make section entries fairly
                // fine-grained to avoid nesting as in this case.
                let mut y = output.get_mut(txn, token);
                let mut guard = section.enter(conn.clone());
                guard._wait(y.async_data());
                y.as_view_mut().set_zeros(conn);
              }
              _ => unimplemented!(),
            }
          }
        })
      },
      tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |x_: Val<GPUDeviceArray1d<T>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    let value = (ext.init)();
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext, value)))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray2d<T>) + 'static,
{
  //fn build(init_val: Rc<F>) -> Rc<FSrcOp<ZerosSrcOp, GPUDeviceArray2d<T>>> {
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray2d<T>> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        Box::new(move || {
        //Box::new(move |state: RefMut<ZerosSrcOp>| {
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray2d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                y.as_view_mut().set_zeros(conn);
              }
              _ => unreachable!(),
            }
          }
        })
      },
      tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |x_: Val<GPUDeviceArray2d<T>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    let value = (ext.init)();
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext, value)))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray4d<T>) + 'static,
{
  //fn build(init_val: Rc<F>) -> Rc<FSrcOp<ZerosSrcOp, GPUDeviceArray4d<T>>> {
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray4d<T>> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        Box::new(move || {
        //Box::new(move |state: RefMut<ZerosSrcOp>| {
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray4d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {}
            }
          }
        })
      },
      tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |x_: Val<GPUDeviceArray4d<T>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    let value = (ext.init)();
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext, value)))
  }
}

pub trait ApplyGPUFlatMap<T> where T: Copy {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<T>, y: GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

impl ApplyGPUFlatMap<f32> for ModulusFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    unsafe { anode_gpu_modulus_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMap<f32> for SquareFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    unsafe { anode_gpu_square_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMap<f32> for PositiveClipFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    unsafe { anode_gpu_positive_clip_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMap<f32> for UnitStepFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    unsafe { anode_gpu_unit_step_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMap<f32> for TanhFlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    unsafe { anode_gpu_tanh_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl ApplyGPUFlatMap<f32> for RCosh2FlatMapF {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<f32>, y: GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    assert!(x.size() <= u32::max_value() as _);
    assert_eq!(x.size(), y.size());
    unsafe { anode_gpu_rcosh2_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl<F> FlatMapFun<F> where F: Clone + 'static {
  pub fn build_gpu_op<T, A>(f_config: F, x_: Val<A>)
      -> Rc<F1Op<Self, A, A>>
  where T: Copy,
        A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>> + 'static,
        F: ApplyGPUFlatMap<T>,
  {
    let ext = OpExt{
      build: {
        let f_config = f_config.clone();
        Box::new(move |args| {
          let f_config = f_config.clone();
          let x_ = match args[0].downcast_ref::<Val<A>>() {
            None => panic!(),
            Some(x_) => x_.clone(),
          };
          let op = FlatMapFun::<F>::build_gpu_op::<T, A>(f_config, x_);
          Val::from(op)
        })
      },
      init: {
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<FlatMapFun<F>>, output: OVal<A>| {
          let x_ = x_.clone();
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Assign => {
                let x = x_.get(txn);
                let mut y = output.get_mut(txn, token);
                state.f.apply_gpu_flat_map(x.flat_view().unwrap(), y.flat_view_mut().unwrap(), conn);
              }
              _ => unimplemented!(),
            }
          }
        })
      },
      tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Box::new(move |x_: Val<A>, sink: &mut Sink| {
          // TODO
          unimplemented!();
        })
      }),
      inplace: Some({
        let f_config = f_config.clone();
        Box::new(move |x_: Val<A>| {
          let op = FlatMapInplaceFun::<F>::build_gpu_op::<T, A>(f_config.clone(), x_);
          Val::from(op)
        })
      }),
    };
    let value = (ext.init)();
    Rc::new(F1Op::new(FlatMapFun{f: f_config}, ext, x_, value))
  }
}

impl<F> FlatMapInplaceFun<F> {
  pub fn build_gpu_op<T, A>(f_config: F, x_: Val<A>)
      -> Rc<F1Op<Self, A, A>>
  where T: Copy,
        A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
  {
    // FIXME
    //let value = x_.value().clobber();
    //Rc::new(F1Op::new(FlatMapInplaceFun{f: f_config}, ext, x_, value))
    unimplemented!();
  }
}

impl SumJoinOp {
  pub fn build_device_op<T, A>(inputs_: Vec<Val<A>>)
      -> Rc<FJoinOp<Self, A, A>>
  where T: Copy + 'static/* + PseudoField*/,
        //A: GPUDeviceArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
        A: GPUDeviceArrayZeros<T> + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>> + 'static,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        let inputs_ = inputs_.clone();
        Box::new(move || {
        //Box::new(move |state: RefMut<SumJoinOp>| {
          //let x0 = inputs_[0].value();
          let inputs_ = inputs_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let x0_size = inputs_[0].get(txn).size();
            A::zeros(x0_size, conn)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        //let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        let inputs_ = inputs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
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
              let x = match inputs_[i].get(txn).flat_view() {
                None => panic!(),
                Some(x) => x,
              };
              y.add(x, conn.clone());
            }
          }
        })
      },
      // TODO
      tangent:  None,
      // TODO
      adjoint:  None,
      inplace: None,
    };
    let output = (ext.init)();
    Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_, output))
  }

  pub fn build_device_batch_op<T, A>(inputs_: Vec<Val<A>>)
      -> Rc<FJoinOp<Self, A, A>>
  where T: Copy /*+ PseudoField*/,
        //A: GPUDeviceBatchArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
        A: GPUDeviceBatchArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>> + 'static,
        //A: GPUDeviceBatchArrayZeros + GPUFlatViewMut<T> + 'static,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        let inputs_ = inputs_.clone();
        Box::new(move || {
        //Box::new(move |state: RefMut<SumJoinOp>| {
          //let x0 = inputs_[0].value();
          let inputs_ = inputs_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let x0_size = inputs_[0].get(txn).size();
            let x0_batch_sz = inputs_[0].get(txn).batch_size();
            A::zeros(x0_size, x0_batch_sz, conn)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        //let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        let inputs_ = inputs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          //let inputs_ = inputs_.clone();
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
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
      // TODO
      tangent:  None,
      // TODO
      adjoint:  None,
      inplace: None,
    };
    let output = (ext.init)();
    Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_, output))
  }
}

/*impl<T, V> SumJoinOpExt<GPUDeviceArray1d<T>> for SumJoinOp
where T: Copy + PseudoField + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V>>>) -> Rc<FJoinOp<Self, V, V>> {
    Self::build_device_op::<T, GPUDeviceArray1d<T>, V>(xs_)
  }
}

impl<T, V> SumJoinOpExt<GPUDeviceOuterBatchArray1d<T>> for SumJoinOp
where T: Copy + PseudoField,
      V: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V>>>) -> Rc<FJoinOp<Self, V, V>> {
    Self::build_device_batch_op::<T, GPUDeviceOuterBatchArray1d<T>, V>(xs_)
  }
}

impl<A, V> SumExt<A, V> for Rc<AOp<V>>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}

impl<A, V, This> SumExt<A, V> for Rc<This>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
      This: AOp<V> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}*/

impl<T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray2d<T>> where T: Copy {
  //fn mult_left_transpose(&self, y: Val<GPUDeviceArray1d<T>>) -> Rc<F2Op<LeftTransposeLinearMapOp, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>>> {
  fn mult_left_transpose(&self, y: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    // TODO
    unimplemented!();
  }
}

/*impl<This, T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Rc<This> where This: AOp<GPUDeviceArray2d<T>>, T: Copy {
  fn mult_left_transpose(&self, y: Val<GPUDeviceArray1d<T>>) -> Rc<F2Op<LeftTransposeLinearMapOp, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>>> {
    // TODO
    unimplemented!();
  }
}*/

impl<T> RightTransposeLinearExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>> for Val<GPUDeviceArray1d<T>> where T: Copy {
  //fn mult_right_transpose(&self, a: Val<GPUDeviceArray1d<T>>) -> Rc<F2Op<RightTransposeLinearMapOp, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>>> {
  fn mult_right_transpose(&self, a: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray2d<T>> {
    // TODO
    unimplemented!();
  }
}

/*impl<This, T> RightTransposeLinearExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>> for Rc<This> where This: AOp<GPUDeviceArray1d<T>>, T: Copy {
  fn mult_right_transpose(&self, a: Val<GPUDeviceArray1d<T>>) -> Rc<F2Op<RightTransposeLinearMapOp, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>>> {
    // TODO
    unimplemented!();
  }
}*/

impl LinearMapOp {
  /*pub fn build_device_op<T, V1, V2, W>(input_: Rc<AOp<V1>>, map_: Rc<AOp<V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>*/
  pub fn build_device_op<T>(map_: Val<GPUDeviceArray2d<T>>, input_: Val<GPUDeviceArray1d<T>>)
      -> Rc<F2Op<Self, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: PseudoField + ZeroBits + Copy + 'static,
        CublasHandle: CublasBlasExt<T>,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      init: {
        let map_ = map_.clone();
        Box::new(move || {
        //Box::new(move |state: RefMut<LinearMapOp>| {
          //let map = map_.value();
          let map_ = map_.clone();
          RWVal::from(Arc::new(move |txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let a_size = map_.get(txn).size();
            GPUDeviceArray1d::zeros(a_size[0], conn)
          }))
        })
      },
      prepare: None,
      cleanup: None,
      apply: {
        //let input = input_.value();
        //let map = map_.value();
        let input_ = input_.clone();
        let map_ = map_.clone();
        Box::new(move |txn, _state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let alpha = T::one();
            let beta = match cap {
              WriteCap::Assign => T::zero(),
              WriteCap::Accumulate => T::one(),
            };
            assert_eq!(input_.get(txn).size(), map_.get(txn).size()[1]);
            assert_eq!(output.get_mut(txn, token).size(), map_.get(txn).size()[0]);
            assert_eq!(1, map_.get(txn).as_view().stride()[0]);
            let a = map_.get(txn).as_view();
            let x = input_.get(txn).as_view();
            let mut y = output.get_mut(txn, token).as_view_mut();
            let res = unsafe { conn.cublas().gemv(
                CublasTranspose::N,
                sz2int(a.size()[0]),
                sz2int(a.size()[1]),
                &alpha,
                a.as_dptr(), sz2int(a.stride()[1]),
                x.as_dptr(), sz2int(x.stride()),
                &beta,
                y.as_mut_dptr(), sz2int(y.stride()),
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
        Box::new(move || {
          let input_ = input_.clone();
          let map_ = map_.clone();
          let tng_input_ = input_.tangent();
          let tng_map_ = map_.tangent();
          // FIXME
          unimplemented!();
          //let y_ = map_.mult(tng_input_).add(tng_map_.mult(input_));
          //(y_.clone(), y_)
        })
      }),
      adjoint: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Box::new(move |y_: Val<GPUDeviceArray1d<T>>, sink: &mut Sink| {
          //let make = make.clone();
          let input_ = input_.clone();
          let map_ = map_.clone();
          let x_var = input_.var();
          let a_var = map_.var();
          if let Some(adj_y_) = sink.get_adj::<GPUDeviceArray1d<T>>(y_.var()) {
            let adj_a_ = adj_y_.mult_right_transpose(input_);
            let adj_x_ = map_.mult_left_transpose(adj_y_);
            sink.put_adj::<GPUDeviceArray2d<T>>(a_var, adj_a_);
            sink.put_adj::<GPUDeviceArray1d<T>>(x_var, adj_x_);
          }
        })
      }),
      inplace: None,
    };
    let output = (ext.init)();
    Rc::new(F2Op::new(LinearMapOp, ext, input_, map_, output))
  }

  /*pub fn build_device_obatch_op<T, V1, V2, W>(input_: Rc<AOp<V1>>, map_: Rc<AOp<V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>
  where T: Copy,
        V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
        W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }*/

  /*pub fn build_device_batch_affine_op<T, V1, V2, V3, W>(input_: Rc<AOp<V1>>, map_: Rc<AOp<V2>>, bias_: Rc<AOp<V3>>)
      -> Rc<F3Op<Self, V1, V2, V3, W>>
  where T: Copy,
        V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
        V3: RWVal<T=GPUDeviceArray1d<T>> + 'static,
        W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }*/
}

/*impl<A, V> SumExt<A, V> for Rc<AOp<V>>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}

impl<A, V, This> SumExt<A, V> for Rc<This>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
      This: AOp<V> + 'static,
{*/

/*impl<T> MultOpExt<GPUDeviceArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>> for Rc<AOp<GPUDeviceArray2d<T>>>
where T: Copy + PseudoField + 'static,
      /*V1: RWVal<T=GPUDeviceArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      W:  RWVal<T=GPUDeviceArray1d<T>> + 'static,*/
      CublasHandle: CublasBlasExt<T>,
{
  fn mult(self, x: Rc<AOp<GPUDeviceArray1d<T>>>) -> Rc<F2Op<LinearMapOp, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>>> {
    LinearMapOp::build_device_op(x, self)
  }
}

/*impl<T, V1, V2, W> MultOpExt<GPUDeviceOuterBatchArray1d<T>, V1, GPUDeviceOuterBatchArray1d<T>, W> for Rc<AOp<V2>>
where T: Copy,
      V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn mult(self, x: Rc<AOp<V1>>) -> Rc<AOp<W>> {
    LinearMapOp::build_device_obatch_op(x, self)
  }
}*/

impl<T> MultAddOpExt<GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceOuterBatchArray1d<T>> for Rc<AOp<GPUDeviceArray2d<T>>>
where T: Copy,
      /*V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      V3: RWVal<T=GPUDeviceArray1d<T>> + 'static,
      W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,*/
{
  fn mult_add(self, x: Rc<AOp<GPUDeviceOuterBatchArray1d<T>>>, shift: Rc<AOp<GPUDeviceArray1d<T>>>) -> Rc<F3Op<LinearMapOp, GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceOuterBatchArray1d<T>>> {
    LinearMapOp::build_device_batch_affine_op(x, self, shift)
  }
}*/
