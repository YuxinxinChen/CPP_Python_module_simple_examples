### To compile the CPP module
```
python setup.py build --build-base=/home/yuxin420/tmp/pybind_mpi_module/build
```

### To use the compiled module
```
$MPI_HOME/bin/mpirun -n 2 python test_mpi.py

```