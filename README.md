## What this is for

This repository guides you how to train machine learning model in distributed way using
local kubernetes cluster with single node. Nevertheless aquired templates for deploy are
easy to use for real production kubernetes cluster on multiple machines which will allow
you to distribute workload

## How to achive that

In kubernetes there is specific workload type that allows you to create scheduled tasks
where task in basically any number of pods that might run in parallel from which is what
we need for multiple nodes training setup

For distributed setup we will have single master node and slave nodes. Every slave node has
to know master node address. We will use kubectl inside each slave pod to get information about
running task and thus retrive task with index zero and then retrive its address. To do so we
need to have access to out kubernetes cluster from each pod inside container. To do so we create
kubernetes secret with kubernetes config. We can copy our config from ~/.kube/config but do not
forget that our cluster is local thus we will not be able to use kubectl since it will try to use
application programming interface of cluster with addres of localhost but localhost inside container 
is not the same as localhost on host machine thus we have to change that inside ~/.kube/config inside 
container to https://kubernetes.docker.internal:6443 to make it work. Also we have to install kubectl
inside docker

To make kubernetes able to pull image that we build inside pod we have to upload it to our registry

```bash
docker compose build dependencies

docker login

docker tag how-to-create-distributed-kubernetes-task-dependencies vladislavkruglikov/how-to-create-distributed-kubernetes-task-dependencies

docker push vladislavkruglikov/how-to-create-distributed-kubernetes-task-dependencies
```

Just for your information docker would not allow to push tag such as vladislavkruglikov/how-to-create-distributed-kubernetes-task/dependencies so that you have to push vladislavkruglikov/how-to-create-distributed-kubernetes-task-dependencies to make it work

To specify nodes that we want to run out task on we have to first add label to local node

```
kubectl get nodes --show-labels
kubectl label nodes docker-desktop cluster=local
```

So that now we can select nodes that we want to use based on label cluster which will have different
values for different nodes

At this point we can run multiple pods as a part of single task such that each pod have access to
kubernetes cluster and knows addres of zero pod which is our master pod inside environment variable
