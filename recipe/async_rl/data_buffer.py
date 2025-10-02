import asyncio
from collections import deque
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from transformers import AutoTokenizer


class DataBuffer:
    """
    Data buffer is responsible receiving data from a worker, and passing it to the server.
    """

    def __init__(self):
        self.data_buffer = deque()
        self._send_counter = 0
        self._target_send_size = -1
        self._add_counter = 0

    async def add_data(self, data: DataProto):
        self.data_buffer.append(data)
        try:
            self._add_counter += len(data)
        except Exception:
            pass

    async def set_recv_target(self, past_stage_send_size: int):
        # When the previous stage is drained, set the current stage's drain target.
        self._target_send_size = past_stage_send_size

    async def reset_counter(self):
        self._send_counter = 0
        self._target_send_size = -1
        self._add_counter = 0

    async def _get_data_impl(self, batch_size: int) -> DataProto:
        needed_size = batch_size
        datas = []
        while needed_size > 0:
            if len(self.data_buffer) > 0:
                data = self.data_buffer.popleft()
                if len(data) > needed_size:
                    datas.append(data[:needed_size])
                    self.data_buffer.appendleft(data[needed_size:])
                    needed_size = 0
                else:
                    datas.append(data)
                    needed_size -= len(data)
            else:
                # TODO: sleep too long?
                await asyncio.sleep(0.1)
        return DataProto.concat(datas)

    async def get_data(self, batch_size: int):
        # If already drained, wait for the reset signal.
        while (
            self._target_send_size >= 0 and self._send_counter == self._target_send_size
        ):
            await asyncio.sleep(0.1)

        # Get data from the buffer.
        data = await self._get_data_impl(batch_size)
        self._send_counter += batch_size

        drain_signal = -1
        if self._target_send_size >= 0 and self._send_counter == self._target_send_size:
            drain_signal = self._send_counter

        return data, drain_signal

    async def get_counters(self):
        return {
            "add_counter": self._add_counter,
            "send_counter": self._send_counter,
            "target_send_size": self._target_send_size,
            "queue_len": len(self.data_buffer),
        }


class DataLoaderBuffer(DataBuffer):
    def __init__(
        self,
        data_config,
        tokenizer_name,
        processor,
        batch_size: int,
        max_samples: int = -1,
        is_validation: bool = False,
    ):
        super().__init__()
        self.data_config = data_config
        self.tokenizer_name = tokenizer_name
        self.processor = processor
        self.data_files = data_config.train_files
        self.max_samples = max_samples
        self.is_validation = is_validation

        if is_validation:
            self.data_files = data_config.val_files

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        dataset = create_rl_dataset(
            self.data_files, self.data_config, self.tokenizer, self.processor
        )
        if is_validation and self.max_samples > 0 and len(dataset) > self.max_samples:
            # Use HuggingFace Dataset.select() instead of pandas DataFrame.iloc[]
            dataset.dataframe = dataset.dataframe.select(range(self.max_samples))
        self.dataset = dataset
        dataset_sampler = create_rl_sampler(self.data_config, self.dataset)
        collate_fn = default_collate_fn
        num_workers = data_config.dataloader_num_workers
        self.dataloader = StatefulDataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=not self.is_validation,  # Don't drop last batch for validation
            collate_fn=collate_fn,
            sampler=dataset_sampler,
        )
        self._dataloader_iter = iter(self.dataloader)

        self.num_data = len(self.dataloader) * batch_size
        # Don't set _target_send_size here - let reset_counter handle it?
        self._target_send_size = self.num_data

    async def reset_counter(self):
        # This buffer is the source to drain all other buffers.
        self._send_counter = 0
        self._dataloader_iter = iter(self.dataloader)
        self._target_send_size = self.num_data

    async def add_data(self, data: DataProto):
        assert RuntimeError("should not add data to the dataloader buffer")

    async def _get_data_impl(self, batch_size: int) -> DataProto:
        # Prevent requesting 0-sized batches
        if batch_size <= 0:
            return DataProto(batch=None, non_tensor_batch={}, meta_info={})

        assert batch_size % self.dataloader.batch_size == 0
        batches = []
        num_batches_needed = batch_size // self.dataloader.batch_size

        for i in range(num_batches_needed):
            try:
                batch = next(self._dataloader_iter)
                batches.append(DataProto.from_single_dict(batch))
            except StopIteration:
                # Dataloader is exhausted, break early
                break

        # Handle case where no batches were collected (dataloader exhausted)
        if len(batches) == 0:
            # Return empty DataProto to signal end of data
            return DataProto(batch=None, non_tensor_batch={}, meta_info={})

        if len(batches) > 1:
            result = DataProto.concat(batches)
        else:
            result = batches[0]
        return result

    def get_dataset_size(self):
        try:
            return len(self.dataset)
        except:
            return -1

    async def save_checkpoint(self, checkpoint_path: str):
        torch.save(self.dataloader.state_dict(), checkpoint_path)

    async def load_checkpoint(self, checkpoint_path: str):
        self.dataloader.load_state_dict(torch.load(checkpoint_path))


class RequestGatherBuffer(DataBuffer):
    """A special buffer that guarantees that data of the same uid are sent together."""

    def __init__(self, num_repeats: int):
        super().__init__()
        self.num_repeats = num_repeats
        self.pending_data: dict[str, list[DataProto]] = {}
        self.total_rewards_received_train: int = 0
        self.total_rewards_received_val: int = 0

    async def add_data(self, data: DataProto):
        # TODO: is there a more efficient way?
        for i in range(len(data)):
            uid = data.non_tensor_batch["uid"][i]
            if uid not in self.pending_data:
                self.pending_data[uid] = []
            # use this slice to keep the batch dimension
            self.pending_data[uid].append(data[i : i + 1])
            if len(self.pending_data[uid]) == self.num_repeats:
                # Handle case where pending_data[uid] might be empty
                if len(self.pending_data[uid]) > 0:
                    grouped = DataProto.concat(self.pending_data[uid])
                    try:
                        self._add_counter += len(grouped)
                    except Exception:
                        pass
                    self.data_buffer.append(grouped)
                del self.pending_data[uid]

    async def _get_data_impl(self, batch_size: int) -> DataProto:
        assert batch_size % self.num_repeats == 0
        return await super()._get_data_impl(batch_size)

    async def get_counters(self):
        base = await super().get_counters()
        base.update(
            {
                "pending_uids": len(self.pending_data),
                "total_rewards_received_train": self.total_rewards_received_train,
                "total_rewards_received_val": self.total_rewards_received_val,
            }
        )
        return base
