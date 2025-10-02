from abc import abstractmethod
import asyncio
import ray
from omegaconf import DictConfig

from async_rl.data_buffer import DataBuffer


async def aenumerate(ait, start=0):
    """Asynchronously enumerate an async iterator."""
    n = start
    async for elem in ait:
        yield n, elem
        n += 1


class StageController:
    def __init__(self, config: DictConfig):
        self.config = config
        self.src_data_buffer: DataBuffer = None
        self.dst_data_buffer: DataBuffer = None
        self.val_src_data_buffer: DataBuffer = None
        self.val_dst_data_buffer: DataBuffer = None
        self._terminate_signal = False
        self.dataset_size = -1

    @abstractmethod
    def init_wg(self, world_desc: list[int]):
        raise NotImplementedError()

    async def set_terminate(self):
        """Termination by signal sent outside."""
        self._terminate_signal = True

    #### Data buffer related functions.
    def link_buffer(self, src_data_buffer: DataBuffer, dst_data_buffer: DataBuffer):
        self.src_data_buffer = src_data_buffer
        self.dst_data_buffer = dst_data_buffer

    def link_validation_buffer(self, val_src_data_buffer: DataBuffer, val_dst_data_buffer: DataBuffer):
        self.val_src_data_buffer = val_src_data_buffer
        self.val_dst_data_buffer = val_dst_data_buffer

    def dst_drain_size(self, src_drain_size: int):
        """
        If the dataset size is unknown until the end of the epoch (e.g. streaming dataset),
        it needs to be propagated to the next stage.
        """
        return src_drain_size

    async def iterator(self, batch_size: int):
        """If dataset size is known, no need to react from the get_data_iter."""
        if self.dataset_size > 0:
            async_iterator = self.get_data_iter(batch_size)
            num_full_batches = self.dataset_size // batch_size
            remainder = self.dataset_size % batch_size

            assert (
                self.dataset_size == num_full_batches * batch_size + remainder
            ), f"Dataset size {self.dataset_size} != {num_full_batches} * {batch_size} + {remainder}"

            count = 0
            async for data in async_iterator:
                yield count, data
                count += 1
                if count == num_full_batches + (1 if remainder > 0 else 0):
                    break
        else:
            async for i, data in aenumerate(self.get_data_iter(1)):
                yield i, data

    async def validation_iterator(self, batch_size: int):
        async for i, data in aenumerate(self.get_data_iter(batch_size, is_validation=True)):
            yield i, data

    def set_dataset_size(self, dataset_size: int):
        """Set the dataset size for the rollout controller."""
        self.dataset_size = dataset_size

    async def get_data_iter(self, batch_size: int, is_validation: bool = False):
        while not self._terminate_signal:
            try:
                if is_validation:
                    data, drain_signal = await self.val_src_data_buffer.get_data.remote(batch_size)
                else:
                    data, drain_signal = await self.src_data_buffer.get_data.remote(batch_size)
            except StopAsyncIteration:
                return
            yield data
            # handle the drain signal
            if drain_signal >= 0:
                if is_validation:
                    if self.val_dst_data_buffer is not None:
                        await self.val_dst_data_buffer.set_recv_target.remote(self.dst_drain_size(drain_signal))
                else:
                    if self.dst_data_buffer is not None:
                        # If not the last stage, pass the drain signal.
                        await self.dst_data_buffer.set_recv_target.remote(self.dst_drain_size(drain_signal))
                break
