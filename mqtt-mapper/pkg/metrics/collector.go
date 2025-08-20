package metrics

import (
	"fmt"
	"strconv"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/klog/v2"
)

// DeviceMetricsCollector manages Prometheus metrics for device properties
type DeviceMetricsCollector struct {
	gauges map[string]prometheus.Gauge
	mutex  sync.RWMutex
}

// NewDeviceMetricsCollector creates a new device metrics collector
func NewDeviceMetricsCollector() *DeviceMetricsCollector {
	return &DeviceMetricsCollector{
		gauges: make(map[string]prometheus.Gauge),
		mutex:  sync.RWMutex{},
	}
}

// UpdateDeviceProperty updates or creates a Prometheus gauge for a device property
// metricName format: "twin_{deviceName}_{propertyName}"
func (dmc *DeviceMetricsCollector) UpdateDeviceProperty(deviceName, deviceNamespace, propertyName, value string) error {
	dmc.mutex.Lock()
	defer dmc.mutex.Unlock()

	// Create metric name in the format: twin_deviceName_propertyName
	metricName := fmt.Sprintf("twin_%s_%s", deviceName, propertyName)

	// Check if gauge already exists
	gauge, exists := dmc.gauges[metricName]
	if !exists {
		// Create new gauge
		gauge = prometheus.NewGauge(prometheus.GaugeOpts{
			Name: metricName,
			Help: fmt.Sprintf("Device twin property value for %s.%s", deviceName, propertyName),
			ConstLabels: prometheus.Labels{
				"device_name":      deviceName,
				"device_namespace": deviceNamespace,
				"property_name":    propertyName,
			},
		})

		// Register the gauge
		if err := prometheus.Register(gauge); err != nil {
			klog.Errorf("Failed to register gauge %s: %v", metricName, err)
			return err
		}

		dmc.gauges[metricName] = gauge
		klog.V(2).Infof("Created new Prometheus gauge: %s", metricName)
	}

	// Convert value to float64 and set the gauge
	numericValue, err := convertToFloat64(value)
	if err != nil {
		klog.V(2).Infof("Could not convert value '%s' to numeric for metric %s, setting to 0: %v", value, metricName, err)
		// For non-numeric values, we could set to 0 or skip the metric
		// For now, let's set to 0 for status-like properties
		numericValue = 0
	}

	gauge.Set(numericValue)
	klog.V(3).Infof("Updated Prometheus gauge %s to value: %f (original: %s)", metricName, numericValue, value)

	return nil
}

// RemoveDeviceMetrics removes all metrics for a specific device
func (dmc *DeviceMetricsCollector) RemoveDeviceMetrics(deviceName string) {
	dmc.mutex.Lock()
	defer dmc.mutex.Unlock()

	prefix := fmt.Sprintf("twin_%s_", deviceName)
	for metricName, gauge := range dmc.gauges {
		if len(metricName) >= len(prefix) && metricName[:len(prefix)] == prefix {
			prometheus.Unregister(gauge)
			delete(dmc.gauges, metricName)
			klog.V(2).Infof("Removed Prometheus gauge: %s", metricName)
		}
	}
}

// GetMetricsCount returns the number of active metrics
func (dmc *DeviceMetricsCollector) GetMetricsCount() int {
	dmc.mutex.RLock()
	defer dmc.mutex.RUnlock()
	return len(dmc.gauges)
}

// convertToFloat64 attempts to convert a string value to float64
// It handles various numeric formats and special cases
func convertToFloat64(value string) (float64, error) {
	// Try direct conversion first
	if numValue, err := strconv.ParseFloat(value, 64); err == nil {
		return numValue, nil
	}

	// Handle boolean values
	switch value {
	case "true", "online", "connected", "active", "enabled":
		return 1.0, nil
	case "false", "offline", "disconnected", "inactive", "disabled":
		return 0.0, nil
	}

	// Try removing units (e.g., "25.5°C" -> "25.5")
	for i, r := range value {
		if !((r >= '0' && r <= '9') || r == '.' || r == '-' || r == '+') {
			if truncated := value[:i]; truncated != "" {
				if numValue, err := strconv.ParseFloat(truncated, 64); err == nil {
					return numValue, nil
				}
			}
			break
		}
	}

	return 0, fmt.Errorf("cannot convert '%s' to numeric value", value)
}
