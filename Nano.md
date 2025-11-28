## Comando para correr el minigrid con cpu de prueba :)
```
ts bash run_cpu.sh python3 dreamerv3/main.py --logdir ~/logdir/dreamer/minigrid/size1m/01 --configs minigrid size1m --run.steps 100 --jax.platform cpu
```

## Comando para correr el minigrid con GPU
```
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py --logdir ~/logdir/dreamer/minigrid/size12m/01 --configs minigrid size12m --run.steps 10000
```


### ubicacion claves de git
clave privada
```bash
home/iamonardes/.ssh/deploy_minigrid
```

clave publica
```bash
home/iamonardes/.ssh/deploy_minigrid_pub
```
