## Comando para correr el minigrid con cpu de prueba :)
```
ts bash run_cpu.sh python3 dreamerv3/main.py --logdir ~/logdir/dreamer/minigrid/size1m/01 --configs minigrid size1m --run.steps 100 --jax.platform cpu
```

## Comando para correr el minigrid con GPU
```
ts bash run_cpu.sh python3 dreamerv3/main.py --logdir ~/logdir/dreamer/minigrid/size1m/01 --configs minigrid size1m --run.steps 100 --jax.platform gpu
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


ts bash run.sh python3 dreamerv3/main.py --logdir ~/logdir/dreamer/minigrid/size1m/01 --configs minigrid size1m --run.steps 700 --jax.platform cpu