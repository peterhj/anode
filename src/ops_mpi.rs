pub struct MPIBroadcastOp;
pub struct MPIReduceOp;
pub struct MPIAllreduceOp;
pub struct MPIReduceScatterOp;
pub struct MPIAllgatherOp;

pub trait MPIBroadcastExt<V> {
  fn mpi_broadcast() -> Val<V>
  fn mpi_broadcast_root(self) -> Val<V>;
}

pub fn mpi_broadcast<V: 'static>() -> Val<V> where Val<V>: MPIBroadcastExt<V> {
  <Val<V> as MPIBroadcastExt<V>>::mpi_broadcast()
}

pub trait MPIReduceExt<V> {
  fn mpi_reduce(self);
  fn mpi_reduce_root(self) -> Val<V>;
}

pub trait MPIAllreduceExt<V> {
  fn mpi_allreduce(self) -> Val<V>;
}

pub trait MPIReduceScatterExt<V> {
  fn mpi_reduce_scatter(self) -> Val<V>;
}

pub trait MPIAllgatherExt<V> {
  fn mpi_allgather(self) -> Val<V>;
}
