# Demo of hands raised count with mediapipe and Kubeedge

The demo consists of running a Kubeedge node on a Raspberry Pi that counts the number of raised hands using the mediapipe framework.

This repo contains:

- A mock Python app that mimicks a change of value and publishes to the MQTT topic to update Kubeedge.
- A real Python app that will publish to MQTT every time the number of raised hands changes.

## Getting started

### Prerequisites

- KubeEdge cluster with EdgeCore running on edge nodes
- MQTT broker accessible from edge nodes
- kubectl access to the Kubernetes cluster

### 0. Install monitoring stack

```sh
kubectl create namespace monitoring

helm install prometheus oci://ghcr.io/prometheus-community/charts/prometheus -f ./infra/prometheus-helm-values.yaml -n monitoring --version 27.50.1

helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana -f ./infra/grafana-helm-values.yaml -n monitoring
```

### 1. Install the MQTT mapper

The mqtt mapper subscribes to an MQTT topic and updates the kubeedge resources when a new value is published.

```sh
kubectl apply -f ./mqtt-mapper/k8s-manifest.yaml
```

### 2. Create the `deviceModel` and `device` resources

In this case the device instance only specifies a few parameters along with the mqtt topic to subscribe to.

```sh
kubectl apply -f ./crds/mqttdevice-model.yaml
kubectl apply -f ./crds/mqttdevice-instance.yaml
```

### 3. Manual test

At this point you can already test if it works as expected by manually publishing to the mqtt topic.

```sh
mosquitto_pub -h <IP of edge node> -p 1883 -t "sensor/handsraised/update/json"   -m '{"handsraised": "13", "status": "online"}'
```

Then check the value of the device CR in Kubernetes.

```sh
kubectl get device hands -n default -ojsonpath='{.status.twins[0].reported.value}'
13
```

### 4a. Mock hand counter app

This app will cycle through 0, 1, 2, 3 and publish to MQTT accordingly.

```sh
kubectl apply -f ./hand-detection/mock-hand-counter/k8s-manifest.yaml
```

You can then watch the value of the kubeedge device CR change in real time.

```sh
watch "kubectl get device hands -n default -ojsonpath='{.status.twins[0].reported.value}'"
```

### 4b. Mediapipe powered raised hand detection

A camera must be connected to the kubeedge node and mapped to `/dev/video0`.

```sh
kubectl apply -f ./hand-detection/advanced-multi-person-counter/k8s-manifest.yaml
```