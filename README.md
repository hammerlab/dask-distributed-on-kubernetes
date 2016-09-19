# Running on kubernetes on google container engine

This small repo gives an example Kubernetes configuration for running [dask.distributed](https://github.com/dask/distributed) on Google Container Engine.

## Start a cluster if needed

If you don't already have a cluster running, use a command like the following to start one (here it is called "daskd-cluster"):

```
gcloud container clusters create daskd-cluster \
    --zone us-east1-b --num-nodes=2 \
    --enable-autoscaling --min-nodes=1 --max-nodes=100 \
    --machine-type=n1-standard-32
```

You should see your cluster:
https://console.cloud.google.com/kubernetes/list

Then run this to set it as the default for your session:
```
gcloud config set container/cluster daskd-cluster
gcloud container clusters get-credentials daskd-cluster
```

## Deploy dask distributed

You will want to edit [spec.yaml](spec.yaml) to use the docker image appropriate for your task. You may also want to customize the CPU and memory thresholds requested based on what's required for your task.

This will launch a dask.distributed scheduler and one worker:

```
kubectl create -f spec.yaml
```

You can check how many workers are running with:

```
kubectl get pods
```

Now, scale up the deployment. Here we request 100 workers:

```
kubectl scale deployment daskd-worker --replicas=100
```

You can now run `kubectl get pods` again to check when the workers are started.

You can check on a worker's stdin/stdout with (replace the name with a pod name from `kubectl get pods`):

```
kubectl logs daskd-scheduler-3680716393-j19xr
```

## Run your analysis

First, get the IP of the scheduler (you want the external ip of daskd-scheduler):

```
$ kubectl get service
NAME              CLUSTER-IP    EXTERNAL-IP       PORT(S)    AGE
daskd-scheduler   10.3.249.60   104.196.185.187   8786/TCP   4m
kubernetes        10.3.240.1    <none>            443/TCP    17h
```

For scripting, here's a one-liner for getting the IP:
```
DASK_IP=$(kubectl get service | grep daskd-scheduler | tr -s ' ' | cut -d ' ' -f 3)
```

When you instantiate your dask Executor, just pass in the IP and port:

```python
from math import sqrt
from dask.distributed import Executor
from dask import delayed

client = Executor("104.196.185.187:8786")
tasks = [dask.delayed(sqrt)(i) for i in range(100)]
results = client.compute(tasks, sync=True)
print(results)
```

## Tearing it down

When you're done, shut down the service and cluster:

```
kubectl delete -f spec.yaml
gcloud container clusters delete daskd-cluster
```

## Running a benchmark

We also include a simple [benchmark](benchmarking/benchmark.py) script that will test performance of the cluster with varying numbers of workers (it issues `kubectl` calls itself to change the number of workers). See the script for details. Here's an example invocation:

```
DASK_IP=$(kubectl get service | grep daskd-scheduler | tr -s ' ' | cut -d ' ' -f 3)
python benchmark.py \
    --tasks 5000 \
    --task-time .05 \
    --dask-scheduler $DASK_IP:8786 \
    --jobs-range 200 800 200 \
    --replicas 1 \
    --out results2.csv
```
