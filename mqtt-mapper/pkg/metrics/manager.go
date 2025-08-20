package metrics

import (
	"sync"

	"k8s.io/klog/v2"
)

// Manager coordinates metrics collection and HTTP server
type Manager struct {
	collector *DeviceMetricsCollector
	server    *MetricsServer
	enabled   bool
	mutex     sync.RWMutex
}

var (
	globalManager *Manager
	managerOnce   sync.Once
)

// GetManager returns the global metrics manager instance (singleton)
func GetManager() *Manager {
	managerOnce.Do(func() {
		globalManager = &Manager{
			collector: NewDeviceMetricsCollector(),
			enabled:   false,
		}
	})
	return globalManager
}

// Enable enables metrics collection and starts the HTTP server
func (m *Manager) Enable(port int) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.enabled {
		klog.V(2).Info("Metrics already enabled")
		return nil
	}

	// Create and start the metrics server
	m.server = NewMetricsServer(port)
	if err := m.server.Start(); err != nil {
		return err
	}

	m.enabled = true
	klog.Info("Device metrics collection enabled")
	return nil
}

// Disable disables metrics collection and stops the HTTP server
func (m *Manager) Disable() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.enabled {
		return nil
	}

	if m.server != nil {
		if err := m.server.Stop(); err != nil {
			klog.Errorf("Error stopping metrics server: %v", err)
		}
		m.server = nil
	}

	m.enabled = false
	klog.Info("Device metrics collection disabled")
	return nil
}

// IsEnabled returns whether metrics collection is currently enabled
func (m *Manager) IsEnabled() bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.enabled
}

// UpdateDeviceProperty updates a device property metric if metrics are enabled
func (m *Manager) UpdateDeviceProperty(deviceName, deviceNamespace, propertyName, value string) {
	m.mutex.RLock()
	enabled := m.enabled
	collector := m.collector
	m.mutex.RUnlock()

	if !enabled {
		klog.V(4).Infof("Metrics disabled, skipping update for %s.%s", deviceName, propertyName)
		return
	}

	if err := collector.UpdateDeviceProperty(deviceName, deviceNamespace, propertyName, value); err != nil {
		klog.Errorf("Failed to update metrics for %s.%s: %v", deviceName, propertyName, err)
	}
}

// RemoveDeviceMetrics removes all metrics for a device if metrics are enabled
func (m *Manager) RemoveDeviceMetrics(deviceName string) {
	m.mutex.RLock()
	enabled := m.enabled
	collector := m.collector
	m.mutex.RUnlock()

	if !enabled {
		return
	}

	collector.RemoveDeviceMetrics(deviceName)
}

// GetMetricsCount returns the current number of active metrics
func (m *Manager) GetMetricsCount() int {
	m.mutex.RLock()
	collector := m.collector
	m.mutex.RUnlock()

	return collector.GetMetricsCount()
}
