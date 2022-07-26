# overthrust-tests MMAP Implementation

About Implementation:

In this experiment we are using a MMAP allocator implemented [here] (https://github.com/speglich/devito/commit/ac2b8f60ee8b9faa39b935d0f0dd40c6a9842997)!

There is no C code edited, just used ALLOC_MMAP

* Line 96 simple.py

```python
   u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order,
          allocator=ALLOC_MMAP(),
          initializer=lambda x: None,
          save=solver.geometry.nt)

      fw_op = solver.op_fwd(save=True)
      #rev_op = solver.op_grad(save=True)

      fw_op.apply(rec=rec, src=solver.geometry.src, u=u, dt=dt)
      #rev_op.apply(u=u, dt=dt, rec=rec)
```

Usage:

```
make
```

This should run all the experiments.
