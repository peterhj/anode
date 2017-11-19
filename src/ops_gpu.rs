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
use context::{ExecutionCtx, implicit_ctx};
use ops::*;

use arithmetic::*;
use cuda_blas::new::*;
use devicemem_gpu::*;
use devicemem_gpu::array::*;

use std::marker::{PhantomData};
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
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

pub trait GPUDeviceMemIoReader<'a> {
  fn read_dev_mem(&mut self, src: &'a Any) -> Option<()>;
}

pub trait GPUDeviceMemIoWriter<'a> {
  fn write_dev_mem(&mut self, mode: WriteMode, dst: &'a mut Any) -> Option<()>;
}

pub struct GPUDeviceCtxPushOp;
pub struct GPUDeviceCtxPopOp;

impl<T, F, V> SrcOpExt<GPUDeviceArray1d<T>, Rc<F>, V> for ()
where T: Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray1d<T>) + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<SrcOp<(), V>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: SrcOpExt<|| GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt<|| GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            // TODO: actually, duplicate the closure.
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, output: V| {
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
        Rc::new(move |x_: Rc<AOp<V=V>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(SrcOp::new((), ext, value))
  }
}

impl<T, V> SrcOpExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, V> for ()
where T: Copy + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(init_val: GPUDeviceArray1d<T>) -> Rc<SrcOp<(), V>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            // TODO: actually, duplicate the data.
            GPUDeviceArray1d::<T>::zeros(init_val.size(), conn)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, output: V| {
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
        Rc::new(move |x_: Rc<AOp<V=V>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(SrcOp::new((), ext, value))
  }
}

impl<T, F, V> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<F>, V> for ZerosSrcOp
where T: Copy + 'static,
      F: (Fn(GPUDeviceStreamPool) -> GPUDeviceArray1d<T>) + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(init_val: Rc<F>) -> Rc<SrcOp<ZerosSrcOp, V>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            // TODO: actually, duplicate the closure.
            init_val(pool)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };
            if overwrite {
              // TODO: zero out the whole thing.
              println!("DEBUG: zeroing...");
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
        Rc::new(move |x_: Rc<AOp<V=V>>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(SrcOp::new(ZerosSrcOp, ext, value))
  }
}

impl<T, V> ZerosSrcOpExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, V> for ZerosSrcOp
where T: Copy + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(init_val: GPUDeviceArray1d<T>) -> Rc<SrcOp<ZerosSrcOp, V>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make...");
          let init_val = init_val.clone();
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            //GPUDeviceArray1d::<T>::zeros(size, conn)
            init_val.clone()
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };
            if overwrite {
              // TODO: zero out the whole thing.
              println!("DEBUG: zeroing...");
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
        Rc::new(move |x_: Rc<AOp<V=V>>, sink: &mut Sink| {
          // Do nothing.
          /*if let Some(adj_x_) = sink.get_adj::<W>(x_.var()) {
          }*/
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(SrcOp::new(ZerosSrcOp, ext, value))
  }
}

impl<T, V> SrcOpExt<GPUDeviceArray1d<T>, usize, V> for ()
where T: Copy + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(size: usize) -> Rc<SrcOp<(), V>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make...");
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt<GPUDeviceArray1d>: make: allocating...");
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            GPUDeviceArray1d::<T>::zeros(size, conn)
          }))
        })
      },
      func: {
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            /*let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };*/
            unreachable!("should never evaluate a source");
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
        Rc::new(move |x_: Rc<AOp<V=V>>, sink: &mut Sink| {
          // Do nothing.
          /*if let Some(adj_x_) = sink.get_adj::<W>(x_.var()) {
          }*/
        })
      }),
    };
    let value = (ext.make)();
    Rc::new(SrcOp::new((), ext, value))
  }
}

impl<T, V> SrcOpExt<GPUDeviceArray2d<T>, [usize; 2], V> for ()
where T: Copy + 'static,
      V: RWVal<T=GPUDeviceArray2d<T>> + 'static,
{
  fn build(size: [usize; 2]) -> Rc<SrcOp<(), V>> {
    let ext = OpExt{
      make: {
        Rc::new(move || {
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            GPUDeviceArray2d::<T>::zeros(size, conn)
          }))
        })
      },
      func: {
        unimplemented!();
      },
      adjoint: {
        unimplemented!();
      },
      tangent: {
        unimplemented!();
      },
    };
    // TODO
    unimplemented!();
  }
}

impl<T, V> SrcOpExt<GPUDeviceArray4d<T>, [usize; 4], V> for ()
where T: Copy,
      V: RWVal<T=GPUDeviceArray4d<T>> + 'static,
{
  fn build(size: [usize; 4]) -> Rc<SrcOp<(), V>> {
    // TODO
    unimplemented!();
  }
}

impl<T, V> SrcOpExt<GPUDeviceOuterBatchArray1d<T>, [usize; 2], V> for ()
where T: Copy,
      V: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn build(size: [usize; 2]) -> Rc<SrcOp<(), V>> {
    // TODO
    unimplemented!();
  }
}

impl<T, V> SrcOpExt<GPUDeviceOuterBatchArray3d<T>, [usize; 4], V> for ()
where T: Copy,
      V: RWVal<T=GPUDeviceOuterBatchArray3d<T>> + 'static,
{
  fn build(size: [usize; 4]) -> Rc<SrcOp<(), V>> {
    // TODO
    unimplemented!();
  }
}

impl SumJoinOp {
  pub fn build_device_op<T, A, V>(inputs_: Vec<Rc<AOp<V=V>>>)
      -> Rc<JoinOp<Self, V, V>>
  where T: Copy + PseudoField,
        A: GPUDeviceArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>,
        V: RWVal<T=A> + 'static,
  {
    let ext = OpExt{
      make: {
        let inputs_ = inputs_.clone();
        Rc::new(move || {
          let x0 = inputs_[0].value();
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            //let pool = GPUDeviceStreamPool::implicit();
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let x0_size = x0.get(txn).size();
            A::zeros(x0_size, conn)
          }))
        })
      },
      func: {
        let inputs: Vec<_> = inputs_.iter().map(|x_| x_.value()).collect();
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            //let pool = GPUDeviceStreamPool::implicit();
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };
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
    };
    let output = (ext.make)();
    Rc::new(JoinOp::new(SumJoinOp, ext, inputs_, output))
  }

  pub fn build_device_batch_op<T, A, V>(inputs_: Vec<Rc<AOp<V=V>>>)
      -> Rc<JoinOp<Self, V, V>>
  where T: Copy + PseudoField,
        A: GPUDeviceBatchArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>,
        V: RWVal<T=A> + 'static,
  {
    let ext = OpExt{
      make: {
        let inputs_ = inputs_.clone();
        Rc::new(move || {
          let x0 = inputs_[0].value();
          <V as RWVal>::from(Rc::new(move |txn: Txn| {
            //let pool = GPUDeviceStreamPool::implicit();
            let ctx = implicit_ctx().gpu_device().unwrap();
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
        Rc::new(move |txn: Txn, output: V| {
          if let Some((mode, token)) = output.write(txn) {
            //let pool = GPUDeviceStreamPool::implicit();
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let overwrite = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => false,
              (_, true) => true,
              _ => unreachable!(),
            };
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
    };
    let output = (ext.make)();
    Rc::new(JoinOp::new(SumJoinOp, ext, inputs_, output))
  }
}

impl<T, V> SumJoinOpExt<GPUDeviceArray1d<T>, V> for SumJoinOp
where T: Copy + PseudoField + 'static,
      V: RWVal<T=GPUDeviceArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<Self, V, V>> {
    Self::build_device_op::<T, GPUDeviceArray1d<T>, V>(xs_)
  }
}

impl<T, V> SumJoinOpExt<GPUDeviceOuterBatchArray1d<T>, V> for SumJoinOp
where T: Copy + PseudoField,
      V: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn build(xs_: Vec<Rc<AOp<V=V>>>) -> Rc<JoinOp<Self, V, V>> {
    Self::build_device_batch_op::<T, GPUDeviceOuterBatchArray1d<T>, V>(xs_)
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
  where T: Copy + PseudoField + 'static,
        V1: RWVal<T=GPUDeviceArray1d<T>> + 'static,
        V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
        W:  RWVal<T=GPUDeviceArray1d<T>> + 'static,
        CublasHandle: CublasBlasExt<T>,
  {
    let make = {
      let map_ = map_.clone();
      Rc::new(move || {
        let map = map_.value();
        <W as RWVal>::from(Rc::new(move |txn| {
          //let pool = GPUDeviceStreamPool::implicit();
          let ctx = implicit_ctx().gpu_device().unwrap();
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
        Rc::new(move |txn, output: W| {
          if let Some((mode, token)) = output.write(txn) {
            //let pool = GPUDeviceStreamPool::implicit();
            let ctx = implicit_ctx().gpu_device().unwrap();
            let pool = ctx.pool();
            let conn = pool.conn();
            let alpha = T::one();
            let beta = match (mode, token.first_write()) {
              (WriteMode::Accumulate, false) => T::one(),
              (_, true) => T::zero(),
              _ => unreachable!(),
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
    };
    let output = (ext.make)();
    Rc::new(F2Op::new(LinearMapOp, ext, input_, map_, output))
  }

  pub fn build_device_obatch_op<T, V1, V2, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>)
      -> Rc<F2Op<Self, V1, V2, W>>
  where T: Copy,
        V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
        W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }

  pub fn build_device_batch_affine_op<T, V1, V2, V3, W>(input_: Rc<AOp<V=V1>>, map_: Rc<AOp<V=V2>>, bias_: Rc<AOp<V=V3>>)
      -> Rc<F3Op<Self, V1, V2, V3, W>>
  where T: Copy,
        V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
        V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
        V3: RWVal<T=GPUDeviceArray1d<T>> + 'static,
        W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
  {
    // TODO
    unimplemented!();
  }
}

impl<T, V1, V2, W> MultOpExt<GPUDeviceArray1d<T>, V1, GPUDeviceArray2d<T>, V2, GPUDeviceArray1d<T>, W> for Rc<AOp<V=V2>>
where T: Copy + PseudoField + 'static,
      V1: RWVal<T=GPUDeviceArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      W:  RWVal<T=GPUDeviceArray1d<T>> + 'static,
      CublasHandle: CublasBlasExt<T>,
{
  fn mult(self, x: Rc<AOp<V=V1>>) -> Rc<F2Op<LinearMapOp, V1, V2, W>> {
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

impl<T, V1, V2, V3, W> MultAddOpExt<GPUDeviceOuterBatchArray1d<T>, V1, GPUDeviceArray2d<T>, V2, GPUDeviceArray1d<T>, V3, GPUDeviceOuterBatchArray1d<T>, W> for Rc<AOp<V=V2>>
where T: Copy,
      V1: RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
      V2: RWVal<T=GPUDeviceArray2d<T>> + 'static,
      V3: RWVal<T=GPUDeviceArray1d<T>> + 'static,
      W:  RWVal<T=GPUDeviceOuterBatchArray1d<T>> + 'static,
{
  fn mult_add(self, x: Rc<AOp<V=V1>>, shift: Rc<AOp<V=V3>>) -> Rc<F3Op<LinearMapOp, V1, V2, V3, W>> {
    LinearMapOp::build_device_batch_affine_op(x, self, shift)
  }
}
