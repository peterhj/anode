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
use context::*;
use ffi::routines_gpu::*;
use ops::*;

use arithmetic::*;
use cuda_blas::*;
use cuda_dnn::*;
use devicemem_gpu::*;
use devicemem_gpu::array::*;
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

//pub struct GPUDeviceCtxPushOp;
//pub struct GPUDeviceCtxPopOp;

pub struct GPUMuxFun<F, A> {
  pub dev:  GPUDeviceId,
  pub ext:  OpExt<F, A>,
  pub fun:  RefCell<F>,
}

impl<F, A> Clone for GPUMuxFun<F, A> where F: Clone {
  fn clone(&self) -> Self {
    /*GPUMuxFun{
      dev:  self.dev,
      ext:  self.ext.clone(),
      fun:  RefCell::new(self.fun.borrow().clone()),
    }*/
    unimplemented!();
  }
}

impl<F, A> GPUMuxFun<F, A> {
  pub fn build_ext() -> OpExt<GPUMuxFun<F, A>, A> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          unimplemented!();
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<GPUMuxFun<F, A>>, output: RWVal<A>| {
          let ctx = implicit_ctx().multi_gpu().unwrap().gpu(state.dev.0 as _);
          let guard = push_ctx(ctx);
          (state.ext.func)(txn, state.fun.borrow_mut(), output);
        })
      },
      tangent: None,
      //adjoint: None,
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=A>>, sink: &mut Sink| {
          // TODO
        })
      }),
    };
    ext
  }
}

impl<T, F> SrcOpExt<GPUDeviceArray1d<T>, Rc<F>> for SrcOp
where T: Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray1d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<FSrcOp<SrcOp, GPUDeviceArray1d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: SrcOpExt<|| GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt<|| GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray1d<T>>| {
          if let Some(_) = output.write(txn) {
            panic!("WARNING: SrcOpExt: should never write");
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray1d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(SrcOp, ext, value))
  }
}

/*impl<T> SrcOpExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for SrcOp
where T: Copy + 'static,
{
  fn build(init_val: GPUDeviceArray1d<T>) -> Rc<FSrcOp<SrcOp, GPUDeviceArray1d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            // TODO: actually, duplicate the data.
            GPUDeviceArray1d::<T>::zeros(init_val.size(), conn)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray1d<T>>| {
          if let Some((mode, token)) = output.write(txn) {
            unreachable!("should never evaluate a pure source");
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray1d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(SrcOp, ext, value))
  }
}*/

impl<T, F> SrcOpExt<GPUDeviceArray2d<T>, Rc<F>> for SrcOp
where T: Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray2d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<FSrcOp<SrcOp, GPUDeviceArray2d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray2d<T>>| {
          if let Some(_) = output.write(txn) {
            panic!("WARNING: SrcOpExt: should never write");
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray2d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(SrcOp, ext, value))
  }
}

impl<T, F> SrcOpExt<GPUDeviceArray4d<T>, Rc<F>> for SrcOp
where T: Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray4d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<FSrcOp<SrcOp, GPUDeviceArray4d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray4d<T>>| {
          if let Some(_) = output.write(txn) {
            panic!("WARNING: SrcOpExt: should never write");
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray4d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(SrcOp, ext, value))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray1d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<FSrcOp<ZerosSrcOp, GPUDeviceArray1d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            // TODO: actually, duplicate the closure.
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray1d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Overwrite => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                y.as_view().set_zeros(&conn);
              }
              _ => unimplemented!(),
            }
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray1d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(ZerosSrcOp, ext, value))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray2d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<FSrcOp<ZerosSrcOp, GPUDeviceArray2d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray2d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Overwrite => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                y.as_view().set_zeros(&conn);
              }
              _ => unreachable!(),
            }
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray2d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(ZerosSrcOp, ext, value))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray4d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<FSrcOp<ZerosSrcOp, GPUDeviceArray4d<T>>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          let init_val = init_val.clone();
          RWVal::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<GPUDeviceArray4d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            match cap {
              WriteCap::Overwrite => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                y.as_view().set_zeros(&conn);
              }
              WriteCap::Accumulate => {}
            }
          }
        })
      },
      tangent: Some({
        Rc::new(move || {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        Rc::new(move |x_: Rc<AOp<V=GPUDeviceArray4d<T>>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(FSrcOp::new(ZerosSrcOp, ext, value))
  }
}

impl SumJoinOp {
  pub fn build_device_op<T, A>(inputs_: Vec<Rc<AOp<V=A>>>)
      -> Rc<FJoinOp<Self, A, A>>
  where T: Copy /*+ PseudoField*/,
        A: GPUDeviceArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
  {
    let ext = OpExt{
      make: {
        let inputs_ = inputs_.clone();
        Rc::new(move || {
          let x0 = inputs_[0].value();
          RWVal::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let x0_size = x0.get(txn).size();
            A::zeros(x0_size, conn)
          }))
        })
      },
      func: {
        let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<A>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let mut y = match output.get_mut(txn, token).flat_view() {
              None => panic!(),
              Some(y) => y,
            };
            let x0 = match inputs[0].get(txn).flat_view() {
              None => panic!(),
              Some(x) => x,
            };
            match cap {
              WriteCap::Overwrite => {
                y.copy(&x0, &conn);
              }
              WriteCap::Accumulate => {
                y.add(&x0, &conn);
              }
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
      // TODO
      tangent:  None,
      // TODO
      adjoint:  None,
    };
    let output = (ext.make)();
    Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_, output))
  }

  pub fn build_device_batch_op<T, A>(inputs_: Vec<Rc<AOp<V=A>>>)
      -> Rc<FJoinOp<Self, A, A>>
  where T: Copy /*+ PseudoField*/,
        A: GPUDeviceBatchArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
  {
    let ext = OpExt{
      make: {
        let inputs_ = inputs_.clone();
        Rc::new(move || {
          let x0 = inputs_[0].value();
          RWVal::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let x0_size = x0.get(txn).size();
            let x0_batch_sz = x0.get(txn).batch_size();
            A::zeros(x0_size, x0_batch_sz, conn)
          }))
        })
      },
      func: {
        let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        Rc::new(move |txn: Txn, state: RefMut<_>, output: RWVal<A>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
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
            match cap {
              WriteCap::Overwrite => {
                y.copy(&x0, &conn);
              }
              WriteCap::Accumulate => {
                y.add(&x0, &conn);
              }
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
      // TODO
      tangent:  None,
      // TODO
      adjoint:  None,
    };
    let output = (ext.make)();
    Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_, output))
  }
}

/*impl<T, V> SumJoinOpExt<GPUDeviceArray1d<T>> for SumJoinOp
where T: Copy + PseudoField + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<FJoinOp<Self, V, V>> {
    Self::build_device_op::<T, GPUDeviceArray1d<T>, V>(xs_)
  }
}

impl<T, V> SumJoinOpExt<GPUDeviceOuterBatchArray1d<T>> for SumJoinOp
where T: Copy + PseudoField,
      V: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<FJoinOp<Self, V, V>> {
    Self::build_device_batch_op::<T, GPUDeviceOuterBatchArray1d<T>, V>(xs_)
  }
}

impl<A, V> SumExt<A, V> for Rc<AOp<V=V>>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V=V>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}

impl<A, V, This> SumExt<A, V> for Rc<This>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
      This: AOp<V=V> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V=V>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}*/

impl<T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Rc<AOp<V=GPUDeviceArray2d<T>>> where T: Copy {
  fn mult_left_transpose(&self, y: Rc<AOp<V=GPUDeviceArray1d<T>>>) -> Rc<F2Op<LeftTransposeLinearMapOp, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>>> {
    unimplemented!();
  }
}

impl<This, T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Rc<This> where This: AOp<V=GPUDeviceArray2d<T>>, T: Copy {
  fn mult_left_transpose(&self, y: Rc<AOp<V=GPUDeviceArray1d<T>>>) -> Rc<F2Op<LeftTransposeLinearMapOp, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>>> {
    unimplemented!();
  }
}

impl<T> RightTransposeLinearExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>> for Rc<AOp<V=GPUDeviceArray1d<T>>> where T: Copy {
  fn mult_right_transpose(&self, a: Rc<AOp<V=GPUDeviceArray1d<T>>>) -> Rc<F2Op<RightTransposeLinearMapOp, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>>> {
    unimplemented!();
  }
}

impl<This, T> RightTransposeLinearExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>> for Rc<This> where This: AOp<V=GPUDeviceArray1d<T>>, T: Copy {
  fn mult_right_transpose(&self, a: Rc<AOp<V=GPUDeviceArray1d<T>>>) -> Rc<F2Op<RightTransposeLinearMapOp, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>>> {
    unimplemented!();
  }
}

impl LinearMapOp {
  /*pub fn build_device_op<T, V1, V2, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>*/
  pub fn build_device_op<T>(input_: Rc<AOp<V=GPUDeviceArray1d<T>>>, map_: Rc<AOp<V=GPUDeviceArray2d<T>>>)
      -> Rc<F2Op<Self, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>>>
  where T: PseudoField + Copy + 'static,
        CublasHandle: CublasBlasExt<T>,
  {
    let make = {
      let map_ = map_.clone();
      Rc::new(move || {
        let map = map_.value();
        RWVal::from(Rc::new(move |txn| {
          let ctx = implicit_ctx().gpu().unwrap();
          let pool = ctx.pool();
          let conn = pool.conn();
          let a_size = map.get(txn).size();
          GPUDeviceArray1d::zeros(a_size[0], conn)
        }))
      })
    };
    let ext = OpExt{
      make,
      func: {
        let input = input_.value();
        let map = map_.value();
        Rc::new(move |txn, state: RefMut<_>, output: RWVal<GPUDeviceArray1d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let alpha = T::one();
            let beta = match cap {
              WriteCap::Overwrite => T::zero(),
              WriteCap::Accumulate => T::one(),
            };
            assert_eq!(input.get(txn).size(), map.get(txn).size()[1]);
            assert_eq!(output.get_mut(txn, token).size(), map.get(txn).size()[0]);
            assert_eq!(1, map.get(txn).as_view().stride()[0]);
            let res = unsafe { conn.cublas().gemv(
                CublasTranspose::N,
                sz2int(map.get(txn).as_view().size()[0]),
                sz2int(map.get(txn).as_view().size()[1]),
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
          // FIXME
          unimplemented!();
          //let y_ = map_.mult(tng_input_).add(tng_map_.mult(input_));
          //(y_.clone(), y_)
        })
      }),
      adjoint: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Rc::new(move |y_: Rc<AOp<V=GPUDeviceArray1d<T>>>, sink: &mut Sink| {
          //let make = make.clone();
          let input_ = input_.clone();
          let map_ = map_.clone();
          let x_var = input_.var();
          let a_var = map_.var();
          if let Some((_, adj_y_)) = sink.get_adj::<GPUDeviceArray1d<T>>(y_.var()) {
            let adj_a_ = adj_y_.mult_right_transpose(input_);
            let adj_x_ = map_.mult_left_transpose(adj_y_);
            sink.put_adj::<GPUDeviceArray2d<T>, _>(a_var, adj_a_);
            sink.put_adj::<GPUDeviceArray1d<T>, _>(x_var, adj_x_);
          }
        })
      }),
    };
    let output = (ext.make)();
    Rc::new(F2Op::new(LinearMapOp, ext, input_, map_, output))
  }

  /*pub fn build_device_obatch_op<T, V1, V2, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>
  where T: Copy,
        V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
        W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }*/

  /*pub fn build_device_batch_affine_op<T, V1, V2, V3, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>, bias_: Rc<AOp<V=V3>>)
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

/*impl<A, V> SumExt<A, V> for Rc<AOp<V=V>>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
{
  fn sum(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(xs_)
  }

  fn add(self, x_: Rc<AOp<V=V>>) -> Rc<FJoinOp<SumJoinOp, V, V>> {
    SumJoinOp::build(vec![self, x_])
  }
}

impl<A, V, This> SumExt<A, V> for Rc<This>
where SumJoinOp: SumJoinOpExt<A, V>,
      V: RWVal<T=A> + 'static,
      This: AOp<V=V> + 'static,
{*/

/*impl<T> MultOpExt<GPUDeviceArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>> for Rc<AOp<V=GPUDeviceArray2d<T>>>
where T: Copy + PseudoField + 'static,
      /*V1: RWVal<T=GPUDeviceArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      W:  RWVal<T=GPUDeviceArray1d<T>> + 'static,*/
      CublasHandle: CublasBlasExt<T>,
{
  fn mult(self, x: Rc<AOp<V=GPUDeviceArray1d<T>>>) -> Rc<F2Op<LinearMapOp, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>>> {
    LinearMapOp::build_device_op(x, self)
  }
}

/*impl<T, V1, V2, W> MultOpExt<GPUDeviceOuterBatchArray1d<T>, V1, GPUDeviceOuterBatchArray1d<T>, W> for Rc<AOp<V=V2>>
where T: Copy,
      V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn mult(self, x: Rc<AOp<V=V1>>) -> Rc<AOp<V=W>> {
    LinearMapOp::build_device_obatch_op(x, self)
  }
}*/

impl<T> MultAddOpExt<GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceOuterBatchArray1d<T>> for Rc<AOp<V=GPUDeviceArray2d<T>>>
where T: Copy,
      /*V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      V3: RWVal<T=GPUDeviceArray1d<T>> + 'static,
      W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,*/
{
  fn mult_add(self, x: Rc<AOp<V=GPUDeviceOuterBatchArray1d<T>>>, shift: Rc<AOp<V=GPUDeviceArray1d<T>>>) -> Rc<F3Op<LinearMapOp, GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceOuterBatchArray1d<T>>> {
    LinearMapOp::build_device_batch_affine_op(x, self, shift)
  }
}*/
