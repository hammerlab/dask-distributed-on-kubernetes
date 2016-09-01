# Running on kubernetes on google container engine

## Start a cluster if needed

I called mine "tim-ml1":

```
$ gcloud container clusters create tim-ml1 \
    --zone us-east1-b --num-nodes=1 \
    --enable-autoscaling --min-nodes=1 --max-nodes=5000 \
    --machine-type=n1-standard-1
```

It should show up here:
https://console.cloud.google.com/kubernetes/list

Run this:
```
gcloud config set container/cluster tim-ml1
gcloud container clusters get-credentials tim-ml1
```

## Deploy dask distributed

This will launch dask scheduler and one worker:

```
kubectl create -f spec.yaml
```

Can check it like this:

```
kubectl get pods
```

Then scale it up:

```
kubectl scale deployment daskd-worker --replicas=100
```

## When finished (important)

Run:
```
kubectl delete -f spec.yaml
```


## Other commands

To resize the cluster (probably not necessary if auto scaling):

```
gcloud container clusters resize CLUSTER_NAME --size SIZE
```

Look at logs (exact name will be different):

```
kubectl logs daskd-scheduler-3680716393-j19xr
```




