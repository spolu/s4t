import torch
import torch.nn as nn


class IOTABase():
    def __init__(
            self,
            sync_dir: str,
            module: nn.Module,
    ):
        self._sync_dir = os.path.expanduser(sync_dir),
        self._tmp_dir = os.path.join(self._sync_dir, 'tmp')

        self._module = module

        assert os.path.isdir(self._sync_dir)

    def module(
            self,
    ):
        return self._module

    def atomic_save(
            self,
            obj,
            name: str,
    ):
        tmp_path = os.path.join(self._tmp_dir, name)
        fnl_path = os.path.join(self._sync_dir, name)
        torch.save(obj, tmp_path)
        os.rename(tmp_path, fnl_path)


class IOTASyn(IOTABase):
    def __init__(
            self,
            sync_dir: str,
            module: nn.Module,
    ):
        super(IOTASyn, self).__init__(self, sync_dir, module)

        assert len(os.listdir(self._sync_dir)) == 0
        os.mkdir(self._tmp_dir)

    def broadcast(
            self,
    ):
        torch.save({
            'state_dict': self._module.state_dict(),
        }, os.path.join(

    def aggregate(
            self,
    ):
        pass



class IOTAAck():
    def __init__(
            self,
            sync_dir: str,
            module: nn.Module,
    ):
        super(IOTAAck, self).__init__(self, sync_dir, module)

        assert os.path.isdir(self._tmp_dir)

    def fetch(
            self,
    ):
        pass

    def push(
            self,
            info,
    ):
        pass
