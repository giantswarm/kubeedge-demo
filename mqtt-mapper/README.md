# MQTT Device Mapper for KubeEdge

A KubeEdge device mapper that connects MQTT devices to the KubeEdge ecosystem, enabling real-time device twin synchronization and data collection.

## Features

- **Dynamic Property Support**: Works with any property names defined in your device model (not limited to temperature/status)
- **Prometheus Metrics**: Exposes device property values as Prometheus time series for monitoring and alerting
- **Real-time MQTT Device Communication**: Connects to MQTT devices and collects sensor data
- **Device Twin Synchronization**: Automatically updates Kubernetes Device status with live data
- **Multi-format Support**: Handles JSON, YAML, and XML message formats
- **Configurable Data Collection**: Customizable collection cycles and reporting intervals
- **Edge-native**: Designed to run on KubeEdge edge nodes with direct device access

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudCore     │    │   EdgeCore      │    │   MQTT Devices  │
│   (Kubernetes)  │◄───┤   + Mapper      │◄───┤   (Sensors)     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- KubeEdge cluster with EdgeCore running on edge nodes
- MQTT broker accessible from edge nodes
- kubectl access to the Kubernetes cluster

### 0. Build image

```sh
docker buildx build --platform linux/amd64 -t vxav/mqttmapper .

docker push vxav/mqttmapper
```

### 1. Deploy Device Model

**Important**: This mapper now supports any property names you define in your device model. You're not limited to just "temperature" and "status" - you can use any property names that match your device's data structure.

Create a device model that defines the properties your MQTT device exposes:

```yaml
apiVersion: devices.kubeedge.io/v1beta1
kind: DeviceModel
metadata:
  name: temperature-model
  namespace: default
spec:
  properties:
  - name: temperature
    dataType: INT
    description: temperature sensor model
    accessMode: ReadOnly
    minimum: 1
    maximum: 100
    unit: Celsius
```

Apply the model:
```bash
kubectl apply -f mqttdevice-model.yaml
```

### 2. Deploy Device Instance

Create a device instance that links to your device model and specifies MQTT connection details:

```yaml
apiVersion: devices.kubeedge.io/v1beta1
kind: Device
metadata:
  name: beta1-device
  namespace: default
spec:
  deviceModelRef:
    name: temperature-model
  nodeName: your-edge-node-name  # Replace with your edge node
  properties:
  - name: temperature
    collectCycle: 5000  # Collect every 5 seconds
    reportToCloud: true
    desired:
      value: "30"
    visitors:
      protocolName: mqtt
      configData:
        clientID: temperature_client
        topic: sensor/beta1-device/getsinglevalue/json
        qos: 1
        retain: false
        username: ""
        password: ""
        cleanSession: true
        keepAlive: 60
  protocol:
    protocolName: mqtt
    configData:
      clientID: "beta1-device-client"
      brokerURL: "tcp://your-mqtt-broker:1883"  # Replace with your MQTT broker - e.g. localhost for default mosquitto
      topic: "sensor/beta1-device/deviceinfo/json"
      message: '{"temperature": "30", "status": "online"}'
      username: ""
      password: ""
      connectionTTL: 30000000000
```

Apply the device:
```bash
kubectl apply -f mqttdevice-instance.yaml
```

### 3. Deploy the Mapper

Create a ConfigMap with mapper configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mqtt-mapper-config
  namespace: kubeedge
data:
  config.yaml: |
    grpc_server:
      socket_path: /etc/kubeedge/mqtt.sock
    common:
      name: Mqtt-mapper
      version: v1.13.0
      api_version: v1.0.0
      protocol: mqtt
      address: your-mqtt-broker-ip  # Replace with your MQTT broker IP
      edgecore_sock: /etc/kubeedge/dmi.sock
      http_port: ""
```

Deploy the DaemonSet:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: mqtt-mapper
  namespace: kubeedge
spec:
  selector:
    matchLabels:
      app: mqtt-mapper
  template:
    metadata:
      labels:
        app: mqtt-mapper
    spec:
      nodeSelector:
        node-role.kubernetes.io/edge: ""
      hostNetwork: true
      containers:
      - name: mqtt-mapper
        image: vxav/mqttmapper:0.2
        command: ["./main"]
        args: ["--config-file", "/config/config.yaml", "--v", "2"]
        volumeMounts:
        - name: kubeedge-socket
          mountPath: /etc/kubeedge
        - name: config
          mountPath: /config
        resources:
          limits:
            cpu: 300m
            memory: 500Mi
          requests:
            cpu: 100m
            memory: 100Mi
      volumes:
      - name: kubeedge-socket
        hostPath:
          path: /etc/kubeedge
          type: Directory
      - name: config
        configMap:
          name: mqtt-mapper-config
      tolerations:
      - key: node-role.kubernetes.io/edge
        operator: Exists
        effect: NoSchedule
```

Deploy both:
```bash
kubectl apply -f configmap.yaml
kubectl apply -f daemonset.yaml
```

### 4. Verify Deployment

Check that the mapper is running:
```bash
kubectl get pods -n kubeedge -l app=mqtt-mapper
kubectl logs -n kubeedge -l app=mqtt-mapper
```

Check device status:
```bash
kubectl get device beta1-device -o yaml
```

You should see a `status` section with twin data:
```yaml
status:
  twins:
  - propertyName: temperature
    reported:
      value: "30"
    observedDesired:
      value: "30"
```

## Configuration

### Logging Levels

Control mapper verbosity with the `--v` flag:

- `--v=0`: Minimal logs (errors, critical messages)
- `--v=1`: Include warnings 
- `--v=2`: Include MQTT messages and twin updates (recommended)
- `--v=3`: Include detailed debug information
- `--v=4`: Very verbose (all debug information)

### MQTT Topics

The mapper automatically subscribes to these topic patterns based on your device configuration:

- `{device-path}/update/json` - Real-time sensor data updates
- `{device-path}/getsinglevalue/json` - Single value requests  
- `{device-path}/setsinglevalue/json` - Single value commands

Example: For device topic `sensor/beta1-device/deviceinfo/json`, the mapper subscribes to:
- `sensor/beta1-device/update/json`
- `sensor/beta1-device/getsinglevalue/json`  
- `sensor/beta1-device/setsinglevalue/json`

### Message Format

MQTT messages should be in JSON format:

```json
{
  "temperature": "28.3",
  "status": "online"
}
```

### Collection Cycles

- `collectCycle`: How often the mapper reports to cloud (in milliseconds)
- `reportCycle`: How often data is pushed to external systems (in milliseconds)

Example: `collectCycle: 5000` = report every 5 seconds

## Testing

### Simulate Device Data

Use mosquitto_pub to simulate device data:

```bash
# Send temperature update
mosquitto_pub -h your-mqtt-broker \
  -t sensor/beta1-device/update/json \
  -m '{"temperature": "25.7", "status": "online"}'

## Dynamic Property Support

This mapper has been updated to support any property names defined in your device model, not just "temperature" and "status". You can now create device models with properties like:

- `humidity`, `pressure`, `battery_level` 
- `rpm`, `voltage`, `current`
- `co2_level`, `air_quality`, `noise_level`
- Any custom property names your device supports

The mapper will automatically:
1. Parse all properties from the device config message
2. Subscribe to MQTT updates for any property
3. Report twin data for all defined properties
4. Handle any data types (string, numeric, boolean)

### Example: Environmental Sensor

See `examples/sensor-model.yaml` and `examples/sensor-instance.yaml` for a complete example of a multi-property environmental sensor with humidity, pressure, battery level, and status properties.

To test with different properties:

```bash
# Send multi-property update
mosquitto_pub -h your-mqtt-broker \
  -t "sensor/environmental-sensor/update/json" \
  -m '{"humidity": "65.2", "pressure": "1015.3", "battery_level": "78", "status": "online"}'
```

## Prometheus Metrics

The mapper automatically exposes device property values as Prometheus metrics on port **8080**:

### Metrics Format
- **Metric Name**: `twin_{deviceName}_{propertyName}`
- **Value**: The device property's reported value  
- **Labels**: `device_name`, `device_namespace`, `property_name`

### Examples
```prometheus
# Temperature sensor
twin_beta1_device_temperature{device_name="beta1-device",device_namespace="default",property_name="temperature"} 25.7

# Multi-property environmental sensor
twin_environmental_sensor_humidity{device_name="environmental-sensor",device_namespace="default",property_name="humidity"} 65.2
twin_environmental_sensor_pressure{device_name="environmental-sensor",device_namespace="default",property_name="pressure"} 1015.3
twin_environmental_sensor_battery_level{device_name="environmental-sensor",device_namespace="default",property_name="battery_level"} 78
```

### Access Metrics
```bash
# View all metrics
curl http://localhost:8080/metrics

# Filter device metrics only
curl http://localhost:8080/metrics | grep twin_

# Health check
curl http://localhost:8080/health
```

For detailed metrics testing instructions, see [`examples/test-metrics.md`](examples/test-metrics.md).

### Watch Device Updates

Monitor device twin updates:
```bash
kubectl get device beta1-device -w -o jsonpath='{.status.twins[0].reported.value}'
```

## Troubleshooting

### Common Issues

**Pod stuck in ContainerCreating:**
- Check that ConfigMap exists: `kubectl get configmap mqtt-mapper-config -n kubeedge`
- Verify edge node has KubeEdge running: `systemctl status edgecore`

**No twin data updates:**
- Check mapper logs: `kubectl logs -n kubeedge -l app=mqtt-mapper`
- Verify MQTT broker connectivity from edge node
- Ensure `reportToCloud: true` in device spec

**"device model not found" error:**
- Apply DeviceModel before DeviceInstance
- Ensure DeviceModel name matches `deviceModelRef.name`

**MQTT connection failed:**
- Verify broker URL and port in device `protocol.configData.brokerURL`
- Check network connectivity from edge node to MQTT broker
- Verify authentication credentials if required

### Debug Commands

```bash
# Check mapper status
kubectl get pods -n kubeedge -l app=mqtt-mapper -o wide

# View mapper logs
kubectl logs -n kubeedge -l app=mqtt-mapper -f

# Check device status
kubectl describe device beta1-device

# Check EdgeCore logs
sudo journalctl -u edgecore -f

# Test MQTT connectivity
mosquitto_sub -h your-mqtt-broker -t 'sensor/+/+/+'
```

## Architecture Details

### Data Flow

1. **Device Registration**: Mapper registers with EdgeCore via DMI socket
2. **MQTT Subscription**: Mapper subscribes to device data topics  
3. **Data Collection**: Device publishes data to MQTT broker
4. **Twin Update**: Mapper receives data and updates device twin
5. **Cloud Sync**: EdgeCore syncs twin data to Kubernetes Device status

### Components

- **Mapper Pod**: Runs on edge node, communicates with devices and EdgeCore
- **EdgeCore**: KubeEdge edge runtime, manages device lifecycle
- **CloudCore**: KubeEdge cloud component, syncs device state
- **Device CRDs**: Kubernetes custom resources defining devices and models

