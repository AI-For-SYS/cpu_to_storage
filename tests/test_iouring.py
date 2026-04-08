"""Quick smoke test for iouring_ext. Delete after use."""
import torch
import iouring_ext
import os

test_file = '/dev/shm/iouring_test.bin'

buf = torch.arange(1024, dtype=torch.float16, device='cpu')
block_size = buf.numel() * buf.element_size()

ok = iouring_ext.iouring_write_blocks(buf, block_size, [0], [test_file])
print(f'Write: {ok}')

if ok:
    buf2 = torch.zeros(1024, dtype=torch.float16, device='cpu')
    ok = iouring_ext.iouring_read_blocks(buf2, block_size, [0], [test_file])
    print(f'Read: {ok}')
    print(f'Data matches: {torch.equal(buf, buf2)}')
    os.remove(test_file)
    print('Cleanup done')
else:
    print('Write failed — skipping read test')
