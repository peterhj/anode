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
use ops::*;

use arithmetic::*;
use arrayidx::*;
//use cuda_blas::*;
use cuda_dnn::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use gpudevicemem::array::linalg::*;
use memarray::*;
use rand::{Rng};

use std::cell::{RefMut};
//use std::marker::{PhantomData};
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

impl<T> FlatIO<MemArray1d<T>> where T: Copy {
  pub fn len(&self) -> usize {
    self.buffer.size()
  }

  pub fn next_slice(&mut self, size: usize) -> MemArrayView1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view().view(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }

  pub fn next_slice_mut(&mut self, size: usize) -> MemArrayViewMut1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view_mut().view_mut(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray1d<T>> where T: Copy {
  pub fn size(&self) -> usize {
    self.array.size()
  }

  pub fn next_view(&mut self, size: usize) -> MemArrayView1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    let slice = self.array.as_view().view(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: usize) -> MemArrayViewMut1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    let slice = self.array.as_view_mut().view_mut(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray2d<T>> where T: Copy {
  pub fn size(&self) -> [usize; 2] {
    self.array.size()
  }

  pub fn next_view(&mut self, size: [usize; 2]) -> MemArrayView2d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
    );
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 2]) -> MemArrayViewMut2d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
    );
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray3d<T>> where T: Copy {
  pub fn size(&self) -> [usize; 3] {
    self.array.size()
  }

  pub fn next_view(&mut self, size: [usize; 3]) -> MemArrayView3d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
    );
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 3]) -> MemArrayViewMut3d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
    );
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray4d<T>> where T: Copy {
  pub fn size(&self) -> [usize; 4] {
    self.array.size()
  }

  pub fn next_view(&mut self, size: [usize; 4]) -> MemArrayView4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_offset[3] .. next_offset[3],
    );
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 4]) -> MemArrayViewMut4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_offset[3] .. next_offset[3],
    );
    self.offset = next_offset;
    slice
  }
}

/*impl<T> BatchArrayIO<OuterBatchMemArray3d<T>> where T: Copy {
  pub fn next_view(&mut self, size: [usize; 3], batch_sz: usize) -> MemArrayView4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let prev_eloffset = self.eloffset;
    let next_eloffset = self.eloffset + batch_sz;
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_eloffset .. next_eloffset,
    );
    self.offset = next_offset;
    self.eloffset = next_eloffset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 3], batch_sz: usize) -> MemArrayViewMut4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let prev_eloffset = self.eloffset;
    let next_eloffset = self.eloffset + batch_sz;
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_eloffset .. next_eloffset,
    );
    self.offset = next_offset;
    self.eloffset = next_eloffset;
    slice
  }
}*/

impl<T> FlatIO<GPUDeviceArray1d<T>> where T: Copy {
  pub fn len(&self) -> usize {
    self.buffer.size()
  }

  pub fn next_slice(&mut self, size: usize) -> GPUDeviceArrayView1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view().view(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }

  pub fn next_slice_mut(&mut self, size: usize) -> GPUDeviceArrayViewMut1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view_mut().view_mut(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray1d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<FlatIO<()>>() {
      // TODO
      unimplemented!();
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.size());
      x.as_view().sync_dump_mem(dst_slice, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_view = dst.next_view_mut(x.size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.as_view(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<FlatIO<()>>() {
      // TODO
      unimplemented!();
    }
    if let Some(src) = src.downcast_mut::<FlatIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.size());
            x.as_view_mut().sync_copy_mem(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_view = src.next_view(x.size());
            x.as_view_mut().sync_copy_mem(src_view, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.size());
            guard._wait(src_slice.async_state());
            x.as_view_mut().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray2d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_view = dst.next_view_mut(x.size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.flat_size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.flat_view().unwrap(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      // TODO
      unimplemented!();
    }
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.flat_size());
            guard._wait(src_slice.async_state());
            x.flat_view_mut().unwrap().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray3d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray3d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_view = dst.next_view_mut(x.size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.flat_size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.flat_view().unwrap(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray3d<T>>>() {
      // TODO
      unimplemented!();
    }
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.flat_size());
            guard._wait(src_slice.async_state());
            x.flat_view_mut().unwrap().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray4d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.flat_size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.flat_view().unwrap(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.flat_size());
            guard._wait(src_slice.async_state());
            x.flat_view_mut().unwrap().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceOuterBatchArray1d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.as_view().async_state());
      let mut dst_view = dst.next_view_mut(x.as_view().size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.as_view().async_state());
            x.set_batch_size(src.size()[1]);
            let src_view = src.next_view(x.as_view().size());
            x.as_view_mut().sync_copy_mem(src_view, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceOuterBatchArray3d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray4d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.as_view().async_state());
      let mut dst_view = dst.next_view_mut(x.as_view().size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    println!("DEBUG: GPUDeviceOuterBatchArray3d<T>::_deserialize");
    /*if let Some(src) = src.downcast_mut::<ArrayIO<MemArray3d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.as_view().async_state());
            x.set_batch_size(1);
            // TODO: either upgrade the src view or downgrade the dst view.
            /*let src_view = src.next_view(x.as_view().size());
            x.as_view_mut().sync_copy_mem(src_view, conn);*/
            unimplemented!();
          }
          _ => unimplemented!(),
        }
      }
      return;
    }*/
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray4d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.as_view().async_state());
            x.set_batch_size(src.size()[3]);
            let src_view = src.next_view(x.as_view().size());
            x.as_view_mut().sync_copy_mem(src_view, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

pub struct GPUMuxOp<A> {
  pub dev:  GPUDeviceId,
  pub val:  Val<A>,
}

impl<A> GPUMuxOp<A> where A: 'static {
  pub fn build_ext() -> OpExt<GPUMuxOp<A>, A> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        Box::new(move |state: RefMut<Self>| {
          let ctx = implicit_ctx().multi_gpu().gpu(state.dev);
          let guard = push_ctx(ctx);
          state.val._make_value()
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<Self>, _output: OVal<A>| {
          let ctx = implicit_ctx().multi_gpu().gpu(state.dev);
          let guard = push_ctx(ctx);
          state.val._apply(txn);
        })
      },
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

impl<A> SrcOpExt<A, Rc<Fn(GPUDeviceConn) -> A>> for SrcOp
where A: GPUDeviceAsync + 'static,
{
  fn build(init_val: Rc<Fn(GPUDeviceConn) -> A>) -> Val<A> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<SrcOp>| {
          println!("DEBUG: SrcOpExt: init gpu...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            println!("DEBUG: SrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          if let Some(_) = output.write(txn) {
            panic!("WARNING: SrcOpExt: should never write");
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, state: RefMut<Self>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(SrcOp, ext)))
  }
}

impl<A> TouchSrcOpExt<A, Rc<Fn(GPUDeviceConn) -> A>> for TouchSrcOp
where A: GPUDeviceAsync + 'static,
{
  fn build(init_val: Rc<Fn(GPUDeviceConn) -> A>) -> Val<A> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<TouchSrcOp>| {
          println!("DEBUG: TouchSrcOpExt: init gpu...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            println!("DEBUG: TouchSrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          if let Some((_, token)) = output.write(txn) {
            // No-op, do nothing.
            let _ = output.get_mut(txn, token);
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, state: RefMut<Self>, sink: &mut Sink| {
          // Do nothing.
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
      F: (Fn(GPUDeviceConn) -> A) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<A> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<SrcOp>| {
          println!("DEBUG: RandomBitsSrcOpExt: init gpu...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            println!("DEBUG: RandomBitsSrcOpExt: init gpu: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(conn);
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
          if let Some((cap, token)) = output.write(txn) {
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
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<A>, state: RefMut<_>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(SrcOp, ext)))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceConn) -> GPUDeviceArray1d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray1d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(GPUDeviceConn) -> GPUDeviceArray1d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceArray1d<T>, Rc<Fn(GPUDeviceConn) -> GPUDeviceArray1d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(GPUDeviceConn) -> GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<ZerosSrcOp>| {
          println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: init...");
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: make_val: allocating...");
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            // FIXME: this part really requires auto-wait and auto-registration.
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray1d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            println!("DEBUG: ZerosSrcOpExt<|| GPUDeviceArray1d>: apply: writing...");
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
              _ => unimplemented!(),
            }
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray1d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceConn) -> GPUDeviceArray2d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray2d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<Fn(GPUDeviceConn) -> GPUDeviceArray2d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceArray2d<T>, Rc<Fn(GPUDeviceConn) -> GPUDeviceArray2d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(GPUDeviceConn) -> GPUDeviceArray2d<T>>) -> Val<GPUDeviceArray2d<T>> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<ZerosSrcOp>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray2d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              _ => unreachable!(),
            }
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray2d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl<T, F> ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<F>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
      F: (Fn(GPUDeviceConn) -> GPUDeviceArray4d<T>) + 'static,
{
  fn build(init_val: Rc<F>) -> Val<GPUDeviceArray4d<T>> {
    <Self as ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<Fn(GPUDeviceConn) -> GPUDeviceArray4d<T>>>>::build(init_val)
  }
}

impl<T> ZerosSrcOpExt<GPUDeviceArray4d<T>, Rc<Fn(GPUDeviceConn) -> GPUDeviceArray4d<T>>> for ZerosSrcOp
where T: ZeroBits + Copy + 'static,
{
  fn build(init_val: Rc<Fn(GPUDeviceConn) -> GPUDeviceArray4d<T>>) -> Val<GPUDeviceArray4d<T>> {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        //Box::new(move || {
        Box::new(move |state: RefMut<ZerosSrcOp>| {
          let section = GPULazyAsyncSection::default();
          let init_val = init_val.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let y = init_val(conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<GPUDeviceArray4d<T>>| {
          if let Some((cap, token)) = output.write(txn) {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            match cap {
              WriteCap::Assign => {
                // TODO: zero out the whole thing.
                println!("DEBUG: ZeroSrcOp: zeroing...");
                let mut y = output.get_mut(txn, token);
                guard._wait(y.async_state());
                y.as_view_mut().set_zeros(conn);
              }
              WriteCap::Accumulate => {}
            }
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        Box::new(move |_: Pass, this: Val<GPUDeviceArray4d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          // Do nothing.
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(FSrcOp::new(ZerosSrcOp, ext)))
  }
}

impl SumJoinOp {
  pub fn build_device_op<T, A>(inputs_: Vec<Val<A>>)
      -> Rc<FJoinOp<Self, A, A>>
  where T: Copy + 'static/* + PseudoField*/,
        //A: GPUDeviceArrayZeros + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> + 'static,
        A: GPUDeviceAsync
            + GPUDeviceArrayZeros<T>
            + FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        let inputs_ = inputs_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<SumJoinOp>| {
          let section = GPULazyAsyncSection::default();
          let inputs_ = inputs_.clone();
          RWVal::from(Arc::new(move |txn: Txn| {
            let ctx = implicit_ctx().gpu();
            let mut pool = ctx.pool();
            let conn = pool.conn();
            let mut section = section.clone();
            let mut guard = section.enter(conn.clone());
            let x0 = inputs_[0].get(txn);
            guard._wait(x0.async_state());
            let y = A::zeros(x0.size(), conn);
            guard._wait(y.async_state());
            y
          }))
        })
      },
      apply: {
        let section = GPULazyAsyncSection::default();
        let inputs_ = inputs_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          if let Some((cap, token)) = output.write(txn) {
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
            let x0 = match inputs_[0].get(txn).flat_view() {
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
            for i in 1 .. inputs_.len() {
              let x = match inputs_[i].get(txn).flat_view() {
                None => panic!(),
                Some(x) => x,
              };
              guard._wait(x.async_state());
              y.add(x, conn.clone());
            }
          }
        })
      },
      // TODO
      tangent: None,
      // TODO
      adjoint: None,
      inplace: None,
    };
    Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_))
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
      make_val: {
        let inputs_ = inputs_.clone();
        //Box::new(move || {
        Box::new(move |state: RefMut<SumJoinOp>| {
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
      // TODO
      tangent: None,
      // TODO
      adjoint: None,
      inplace: None,
    };
    Rc::new(FJoinOp::new(SumJoinOp, ext, inputs_))
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

impl RectFlatMapExt<GPUDeviceOuterBatchArray1d<f32>> for Val<GPUDeviceOuterBatchArray1d<f32>> {
  fn rect(self) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    // TODO
    FlatMapOp::<PositiveClipFlatMapF>::build_gpu_val(PositiveClipFlatMapF, self)
  }
}

impl RectFlatMapExt<GPUDeviceOuterBatchArray3d<f32>> for Val<GPUDeviceOuterBatchArray3d<f32>> {
  fn rect(self) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
    // TODO
    unimplemented!();
  }
}

impl TanhFlatMapExt<GPUDeviceOuterBatchArray1d<f32>> for Val<GPUDeviceOuterBatchArray1d<f32>> {
  fn tanh(self) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
    // TODO
    unimplemented!();
  }
}

pub trait ApplyGPUFlatMap<T> where T: Copy {
  fn apply_gpu_flat_map(&self, x: GPUDeviceArrayView1d<T>, y: GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

pub trait BuildGPUFlatMapAdj<T, A> {
  fn build_gpu_adj(&self, adj_y_: Val<A>, y_: Val<A>) -> Val<A> { unimplemented!(); }
  fn build_gpu_adj2(&self, adj_y_: Val<A>, x_: Val<A>, y_: Val<A>) -> Val<A> { unimplemented!(); }
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

impl<T, A> BuildGPUFlatMapAdj<T, A> for ModulusFlatMapF {
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

impl<T, A> BuildGPUFlatMapAdj<T, A> for SquareFlatMapF {
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

impl<T, A> BuildGPUFlatMapAdj<T, A> for PositiveClipFlatMapF
where T: Copy,
      A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
          + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
          + 'static,
      UnitStepFlatMapF: ApplyGPUFlatMap<T>,
{
  fn build_gpu_adj(&self, adj_y_: Val<A>, y_: Val<A>) -> Val<A> {
    // TODO: use fused kernel to avoid an extra allocation.
    let dy_dx_ = FlatMapOp::<UnitStepFlatMapF>::build_gpu_val::<T, A>(UnitStepFlatMapF, y_);
    //let adj_x = dy_dx_.flat_mult(adj_y);
    //adj_x
    unimplemented!();
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

impl<T, A> BuildGPUFlatMapAdj<T, A> for UnitStepFlatMapF
where T: Copy,
      A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
          + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
          + 'static,
{
  fn build_gpu_adj(&self, adj_y_: Val<A>, y_: Val<A>) -> Val<A> {
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
    unsafe { anode_gpu_tanh_flat_map_f32(
        x.size() as _,
        x.as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
    ) };
  }
}

impl<T, A> BuildGPUFlatMapAdj<T, A> for TanhFlatMapF {
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

impl<T, A> BuildGPUFlatMapAdj<T, A> for RCosh2FlatMapF {
}

impl<F> FlatMapOp<F> where F: Clone + 'static {
  pub fn build_gpu_val<T, A>(f_config: F, x_: Val<A>) -> Val<A>
  where T: Copy,
        A: FlatView<FlatViewTy=GPUDeviceArrayView1d<T>>
            + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>>
            + 'static,
        F: ApplyGPUFlatMap<T> + BuildGPUFlatMapAdj<T, A>,
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
          //let op = FlatMapOp::<F>::build_gpu_op::<T, A>(f_config, x_);
          //Val::from(op)
          FlatMapOp::<F>::build_gpu_val::<T, A>(f_config, x_)
        })
      },
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
        let x_ = x_.clone();
        Box::new(move |txn: Txn, state: RefMut<_>, output: OVal<A>| {
          let x_ = x_.clone();
          if let Some((cap, token)) = output.write(txn) {
            let mut pool = implicit_ctx().gpu().pool();
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
          }
        })
      },
      tangent: None,
      /*tangent: Some({
        Box::new(move || {
          // TODO
          unimplemented!();
        })
      }),*/
      adjoint: Some({
        let f_config = f_config.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<A>, state: RefMut<Self>, sink: &mut Sink| {
          let x_ = x_.clone();
          if let Some(adj_y_) = y_.adjoint(sink) {
            let adj_x_ = f_config.build_gpu_adj(adj_y_, y_);
            x_.put_adjoint(adj_x_, sink);
          }
        })
      }),
      inplace: Some({
        let f_config = f_config.clone();
        Box::new(move |x_: Val<A>| {
          FlatMapInplaceOp::<F>::build_gpu_val::<T, A>(f_config.clone(), x_)
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
    unsafe { anode_gpu_M1_copy_map_M2_unit_step_map_R_product_reduce_flat_join_f32(
        sz2uint(y.size()),
        xs[0].as_dptr(),
        xs[1].as_dptr(),
        y.as_mut_dptr(),
        conn.cuda_kernel_config() as *const _,
        conn.cuda_stream().as_mut_ptr(),
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
      build: {
        let f_config = f_config.clone();
        Box::new(move |args| {
          let f_config = f_config.clone();
          let xs_ = match args[0].downcast_ref::<Vec<Val<A>>>() {
            None => panic!(),
            Some(xs_) => xs_.clone(),
          };
          FlatJoinOp::<F>::build_gpu_val::<T, A>(f_config, xs_)
        })
      },
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
          let xs_ = xs_.clone();
          if let Some((cap, token)) = output.write(txn) {
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
          }
        })
      },
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

impl<T> LinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: PseudoField + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut1d<T>: GPUVectorOps<T>,
{
  fn mult(self, x: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    LinearMapOp::build_device_val(self, x)
  }
}

impl<T> LinearExt<GPUDeviceArray2d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: PseudoField + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn mult(self, x: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    LinearMapOp::build_device_obatch_val(self, x)
  }
}

impl<T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceArray1d<T>, GPUDeviceArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: PseudoField + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut1d<T>: GPUVectorOps<T>,
{
  fn left_transpose_mult(self, y: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray1d<T>> {
    // TODO
    unimplemented!();
  }
}

impl<T> LeftTransposeLinearExt<GPUDeviceArray2d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>> for Val<GPUDeviceArray2d<T>>
where T: PseudoField + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn left_transpose_mult(self, y: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceOuterBatchArray1d<T>> {
    LinearMapOp::build_device_obatch_ltrans_val(self, y)
  }
}

impl<T> OuterLinearExt<GPUDeviceArray1d<T>, GPUDeviceArray1d<T>, GPUDeviceArray2d<T>> for Val<GPUDeviceArray1d<T>>
where T: PseudoField + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut1d<T>: GPUVectorOps<T>,
{
  fn outer_mult(self, x: Val<GPUDeviceArray1d<T>>) -> Val<GPUDeviceArray2d<T>> {
    // TODO
    unimplemented!();
  }
}

impl<T> OuterLinearExt<GPUDeviceOuterBatchArray1d<T>, GPUDeviceOuterBatchArray1d<T>, GPUDeviceArray2d<T>> for Val<GPUDeviceOuterBatchArray1d<T>>
where T: PseudoField + ZeroBits + Copy + 'static,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
{
  fn outer_mult(self, x: Val<GPUDeviceOuterBatchArray1d<T>>) -> Val<GPUDeviceArray2d<T>> {
    LinearMapOp::build_device_obatch_rtrans_val(self, x)
  }
}

impl LinearMapOp {
  pub fn build_device_val<T>(map_: Val<GPUDeviceArray2d<T>>, input_: Val<GPUDeviceArray1d<T>>)
      -> Val<GPUDeviceArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: PseudoField + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut1d<T>: GPUVectorOps<T>,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        let map_ = map_.clone();
        Box::new(move |state: RefMut<LinearMapOp>| {
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
          if let Some((cap, token)) = output.write(txn) {
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
          }
        })
      },
      tangent: Some({
        let input_ = input_.clone();
        let map_ = map_.clone();
        Box::new(move |_: Pass, _state: RefMut<Self>, feedfwd: &mut FeedFwd| {
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
        let input_ = input_.clone();
        let map_ = map_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray1d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          let x_ = input_.clone();
          let a_ = map_.clone();
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
    Val::from(Rc::new(F2Op::new(LinearMapOp, ext, map_, input_)))
  }

  pub fn build_device_obatch_val<T>(w_: Val<GPUDeviceArray2d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>)
      -> Val<GPUDeviceOuterBatchArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: PseudoField + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<LinearMapOp>| {
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
          if let Some((cap, token)) = output.write(txn) {
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
                y.matrix_mult(w, x, conn);
              }
              WriteCap::Accumulate => unimplemented!(),
            }
          }
        })
      },
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<Self>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray1d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          let x_ = x_.clone();
          let w_ = w_.clone();
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
    Val::from(Rc::new(F2Op::new(LinearMapOp, ext, w_, x_)))
  }

  pub fn build_device_obatch_ltrans_val<T>(w_: Val<GPUDeviceArray2d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>)
      -> Val<GPUDeviceOuterBatchArray1d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: PseudoField + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<LinearMapOp>| {
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
          if let Some((cap, token)) = output.write(txn) {
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
          }
        })
      },
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<Self>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceOuterBatchArray1d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          let x_ = x_.clone();
          let w_ = w_.clone();
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LinearMapOp, ext, w_, x_)))
  }

  pub fn build_device_obatch_rtrans_val<T>(w_: Val<GPUDeviceOuterBatchArray1d<T>>, x_: Val<GPUDeviceOuterBatchArray1d<T>>)
      -> Val<GPUDeviceArray2d<T>>
  // TODO: `ZeroBits` should not be necessary here.
  where T: PseudoField + ZeroBits + Copy + 'static,
        GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>,
  {
    let ext = OpExt{
      build: {
        Box::new(move |args| {
          // TODO
          unimplemented!();
        })
      },
      make_val: {
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |state: RefMut<LinearMapOp>| {
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
          if let Some((cap, token)) = output.write(txn) {
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
          }
        })
      },
      tangent: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, _state: RefMut<Self>, _feedfwd: &mut FeedFwd| {
          // TODO
          unimplemented!();
        })
      }),
      adjoint: Some({
        let w_ = w_.clone();
        let x_ = x_.clone();
        Box::new(move |_: Pass, y_: Val<GPUDeviceArray2d<T>>, state: RefMut<Self>, sink: &mut Sink| {
          let x_ = x_.clone();
          let w_ = w_.clone();
          if let Some(adj_y_) = y_.adjoint(sink) {
            // TODO
            unimplemented!();
          }
        })
      }),
      inplace: None,
    };
    Val::from(Rc::new(F2Op::new(LinearMapOp, ext, w_, x_)))
  }
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
