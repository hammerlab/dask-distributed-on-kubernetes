# Running on kubernetes on google container engine

## Start a cluster if needed

I called mine "tim-ml1":

```
gcloud container clusters create tim-ml1 \
    --zone us-east1-b --num-nodes=1 \
    --enable-autoscaling --min-nodes=1 --max-nodes=100 \
    --machine-type=n1-standard-32
```

It should show up here:
https://console.cloud.google.com/kubernetes/list

Run this:
```
gcloud config set container/cluster tim-ml1
gcloud container clusters get-credentials tim-ml1
```

## Deploy dask distributed

If you want to use a development checkout of MHCflurry, first build a new MHCflurry docker image. From the MHCflurry checkout, run:

```
docker build .
```

When that completes, tag it, push it to docker hub, and edit `spec.yaml` to point to your image.

This will launch dask scheduler and one worker:

```
kubectl create -f spec.yaml
```

Can check it like this:

```
kubectl get pods
```

Get the IP of the scheduler (you want the external ip of daskd-scheduler):

```
$ kubectl get service
NAME              CLUSTER-IP    EXTERNAL-IP       PORT(S)    AGE
daskd-scheduler   10.3.249.60   104.196.185.187   8786/TCP   4m
kubernetes        10.3.240.1    <none>            443/TCP    17h
```

Then scale it up:

```
kubectl scale deployment daskd-worker --replicas=400
```

## Run analysis

Run mhcflurry-class1-allele-specific-cv-and-train, passing in the host and IP of the scheduler above, e.g. `--dask-scheduler 104.196.185.187:8787`.

## When finished (important)

Run:
```
kubectl delete -f spec.yaml
gcloud container clusters delete tim-ml1
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




